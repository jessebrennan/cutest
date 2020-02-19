import importlib
import inspect
import logging
import sys
from abc import ABC
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import List, Optional, Iterable, Set, Tuple, Mapping

from cutest.util import Stack


"""
Outstanding things:

- What happens if a suite is called inside of another suite?
  Does this break anything?
- Test reporting / logging
- Concurrent fixture
- Skipping tests
- using fixture outside of test
- Calling tests inside of tests?

Next step:
- Write a test that uses a simple fixture. Observe where / how
  it makes sense to access the state.
- Also make a test that contains sub-tests. How are those handled
  with state?

"""

log = logging.getLogger(__name__)


class Model:
    def __init__(self):
        # Used to track the suite when building the graph
        self.current_suite: Optional[Suite] = None
        self.suites: List[Suite] = []

    def suite(self, func):
        suite = Suite(self, func)
        self.suites.append(suite)
        return suite

    def fixture(self, obj):
        if inspect.isclass(obj):
            # FIXME: assert has __enter__ and __exit__
            return Fixture(self, obj)
        elif inspect.isgeneratorfunction(obj):
            cm = contextmanager(obj)
            return Fixture(self, cm)
        else:
            raise CutestError('fixture must decorate a contextmanager or generator')

    def test(self, func):
        return Test(self, func)

    def initialize(self):
        """
        Build the test model graph
        """
        for suite in self.suites:
            suite.initialize()


class Node(ABC):
    """
    Inherit to be allowed in the Suite graph
    """
    def __init__(self, model: Model):
        self.model = model
        # root is set when a node is added to a _Suite
        self.root: Optional[Suite] = None
        # parent is set when a node is added to a node
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

    @property
    def data(self):
        raise NotImplementedError

    def print_graph(self, depth=0):
        print('  ' * depth, self.__class__.__name__, self.data.__name__)
        for node in self.children:
            node.print_graph(depth=depth + 1)


class CallableNode(Node, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = None
        self.args = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _replace_args(self, fixtures) -> Tuple[Iterable, Mapping]:
        """
        If any of the args are Fixtures, we need to substitute them out
        before evaluating
        """
        args = []
        kwargs = {}
        for arg in self.args:
            if isinstance(arg, Fixture):
                assert arg in fixtures
                args.append(arg.context_manager())
            else:
                args.append(arg)
        for key, val in self.kwargs:
            if isinstance(val, Fixture):
                assert val in fixtures
                kwargs[key] = val.context_manager()
            else:
                kwargs[key] = val
        return args, kwargs


# FIXME: Should this inherit from _GraphNode?
class Suite(Node):

    def __init__(self, model: Model, func):
        super().__init__(model)
        self._func = func
        self.fixture_stack: Stack[Fixture] = Stack()

    @property
    def data(self):
        return self._func

    def initialize(self):
        # Reset children to make this method idempotent
        self.children = []
        assert self.fixture_stack.empty()
        self.root = self
        self.parent = None
        self.model.current_suite = self
        self._func()
        self.model.current_suite = None

    def add(self, node: Node):
        node.root = self.root
        if self.fixture_stack.empty():
            node.parent = self
            self.children.append(node)
        else:
            self.fixture_stack.top().add(node)


class Fixture(CallableNode):
    def __init__(self, model: 'Model', cm):
        super().__init__(model)
        self._cm = cm
        self._initialized_cm = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    # FIXME: This seems to cause some problems, at least with Jupyter
    # def __getattr__(self, item):
    #     """
    #     Fixtures need to be substituted with the underlying context manager
    #     before they can be used by the user. This can only happen inside of a
    #     test (or while initializing another fixture).
    #     """
    #     raise CutestError('Fixtures can only be used within tests')

    def initialize(self, fixtures: Set['Fixture']):
        """
        A fixture must be initialized before it's underlying context manager
        can be used
        """
        args, kwargs = self._replace_args(fixtures)
        self._initialized_cm = self._cm(*args, **kwargs)

    def context_manager(self):
        if self._initialized_cm is None:
            raise CutestError('Initialize fixture before using context manager')
        else:
            return self._initialized_cm

    @property
    def data(self):
        return self._cm

    def add(self, node):
        node.parent = self
        self.children.append(node)

    def __enter__(self):
        self.model.current_suite.add(self)
        self.model.current_suite.fixture_stack.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fixture_ = self.model.current_suite.fixture_stack.pop()
        assert fixture_ is self
        return False

    def add_test(self, test_):
        self.children.append(test_)


class Concurrent(Node):
    # TODO: Finish implementing me. Inherit from _Fixture instead?
    # Latest idea is to have this point to a runner class, not take in
    # executor

    def __init__(self, model: Model, executor: Executor):
        # FIXME: What about model for this guy?
        super().__init__(model)
        self.executor = executor

    @property
    def data(self):
        return self.executor


class Test(CallableNode):
    def __init__(self, model: Model, func):
        super().__init__(model)
        self._func = func
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if self.model.current_suite is None:
            raise CutestError(f'Test must be called from within a suite')
        else:
            self.model.current_suite.add(self)

    def run(self, fixtures: Set[Fixture]):
        # TODO: Maybe log something here about the test that's running
        args, kwargs = self._replace_args(fixtures)
        try:
            result = self._func(*args, **kwargs)
        except Exception as e:
            # TODO: Log failure, here or in recursive run? (probably here)
            return False, e
        else:
            # TODO: log success
            return True, result

    @property
    def data(self):
        return self._func


class Runner:

    def __init__(self):
        self.passes: List[Tuple[Test, None]] = []
        self.fails: List[Tuple[Test, Exception]] = []

    def run_suites(self, suites: Iterable[Suite]):
        for suite in suites:
            self.run(suite)

    def run(self, suite: Suite):
        assert suite.root is suite
        print('Running test suite %s', suite)
        suite.print_graph()
        self.recursive_run(suite, fixtures=set())

    def recursive_run(self, node: Node, fixtures: Set[Fixture]):
        if isinstance(node, Test):
            # TODO: What about skipping? Maybe we should trim tree. This would
            # avoid initializing fixtures unnecessarily
            success, result = node.run(fixtures)
            if success:
                assert result is None, 'Tests should not return anything'
                self.passes.append((node, result))
            else:
                # TODO: Should be BaseException?
                assert isinstance(result, Exception)
                self.fails.append((node, result))
            assert len(node.children) == 0
        elif isinstance(node, Fixture):
            node.initialize(fixtures)
            fixtures.add(node)
            for child in node.children:
                self.recursive_run(child, fixtures)
            fixtures.remove(node)
        elif isinstance(node, Concurrent):
            with node.executor as executor:
                # FIXME: This only runs children concurrently. Executor should
                # be passed on recursively and _Sequential should be added
                # which would set recursive exec to None. What about sub-concurrent calls?
                for child in node.children:
                    executor.submit(self.recursive_run, child, fixtures)
        elif isinstance(node, Suite):
            assert node.root is node, "Cannot handle sub-suites yet"
            for child in node.children:
                self.recursive_run(child, fixtures)
        else:
            assert False


def main(argv):
    module_name = argv[0]
    module = importlib.import_module(module_name)
    suites = []
    for member in inspect.getmembers(module):
        if isinstance(member, Suite):
            suites.append(member)
    for suite in suites:
        suite.initialize()
    runner = Runner()
    runner.run_suites(suites)


class CutestError(Exception):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
