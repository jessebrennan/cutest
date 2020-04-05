import argparse
import importlib
import inspect
import logging
import sys
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import Executor, Future
from contextlib import contextmanager, ExitStack
from copy import copy
from typing import List, Optional, Iterable, Set, Tuple, Mapping, Type, Callable, Union, ContextManager

from cutest.util import Stack

from release import read_version

__version__ = read_version(string=True)

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

    def fixture(self, obj) -> 'FixtureDefinition':
        """
        Decorate a class context manager or a generator function to
        make it a test fixture.

        If decorating a generator function, yield once inside of a try
        block. Any cleanup should happen inside of an attached finally
        block.
        """
        if (
                inspect.isclass(obj)
                and hasattr(obj, '__enter__')
                and hasattr(obj, '__exit__')
        ):
            return FixtureDefinition(self, obj)
        elif inspect.isgeneratorfunction(obj):
            cm = contextmanager(obj)
            return FixtureDefinition(self, cm)
        else:
            raise CutestError('fixture must decorate a contextmanager or generator')

    def test(self, func: Callable) -> 'TestDefinition':
        return TestDefinition(self, func)

    def serial(self):
        return self.runner(SerialRunner)

    def threads(self):
        raise NotImplementedError()

    def processes(self):
        raise NotImplementedError()

    def runner(self, runner: Type['Runner']) -> 'RunnerNode':
        return RunnerNode(self, runner)

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
        # root is set when a node is added to a Suite
        self.root: Optional[Suite] = None
        # parent is set when a node is added to a node
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

    @abstractmethod
    @property
    def data(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.data.__name__

    @property
    def type(self):
        return self.__class__.__name__

    def print_graph(self, depth=0):
        log.info('%s%s %s', '  ' * depth, self.type, self.name)
        for node in self.children:
            node.print_graph(depth=depth + 1)

    def prune_all_but(self, nodes: Iterable['Node']) -> bool:
        """
        Prune children unless they are in nodes or ancestors of nodes.

        Return if this node should be saved
        """
        save_children = [c for c in self.children if c.prune_all_but(nodes)]
        self.children = save_children
        return self in nodes or any(save_children)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        else:
            return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    # TODO: evaluate consequences of moving this up
    def add(self, node):
        node.parent = self
        self.children.append(node)


class CallableNode(Node, ABC):

    def __init__(self, model, args, kwargs):
        super().__init__(model)
        self.args = args
        self.kwargs = kwargs

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


@contextmanager
def combine(*context_managers: ContextManager):
    """
    Context manager combinator, used mainly to save indentation

    Context managers are entered in argument-order and exited in the
    opposite order. The return values of enter are returned as a list
    from __enter__()
    """
    with ExitStack() as stack:
        contexts = []
        for cm in context_managers:
            contexts.append(stack.enter_context(cm))
        yield contexts


# FIXME: Should this inherit from Node?
class Suite(Node):

    def __init__(self, model: Model, func):
        super().__init__(model)
        self._func = func
        # FIXME: make the type non-leaf Node
        self.parent_stack: Stack[Union[Fixture, RunnerNode]] = Stack()

    @property
    def data(self):
        return self._func

    def initialize(self):
        # Reset children to make this method idempotent
        self.children = []
        assert self.parent_stack.empty()
        self.root = self
        self.parent = None
        self.model.current_suite = self
        self._func()
        self.model.current_suite = None

    def add(self, node: Node):
        node.root = self.root
        if self.parent_stack.empty():
            super().add(node)
        else:
            self.parent_stack.top().add(node)


class FixtureDefinition:

    def __init__(self, model: Model, cm):
        self.model = model
        self.cm = cm

    def __call__(self, *args, **kwargs):
        return Fixture(self.cm, self.model, args, kwargs)


class Fixture(CallableNode):

    def __init__(self, cm, model: Model, args, kwargs):
        super().__init__(model, args, kwargs)
        self.cm = cm
        self._initialized_cm = None

    def __getattr__(self, item):
        """
        Fixtures need to be substituted with the underlying context manager
        before they can be used by the user. This can only happen inside of a
        test (or while initializing another fixture).
        """
        raise CutestError('Fixtures can only be used within tests or other fixtures')

    def initialize(self, fixtures: Set['Fixture']):
        """
        A fixture must be initialized before it's underlying context manager
        can be used
        """
        args, kwargs = self._replace_args(fixtures)
        self._initialized_cm = self.cm(*args, **kwargs)

    def context_manager(self):
        if self._initialized_cm is None:
            raise CutestError('Initialize fixture before using context manager')
        else:
            return self._initialized_cm

    @property
    def data(self):
        return self.cm

    def __enter__(self):
        self.model.current_suite.add(self)
        self.model.current_suite.parent_stack.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fixture_ = self.model.current_suite.parent_stack.pop()
        assert fixture_ is self
        return False


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


class TestDefinition:

    def __init__(self, model: Model, func):
        self.model = model
        self._func = func
        self.calls: List[Test] = []

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls.append(Test(self._func, self.model, args, kwargs))


class Test(CallableNode):

    def __init__(self, func, model, args, kwargs):
        super().__init__(model, args, kwargs)
        self.func = func
        if self.model.current_suite is None:
            raise CutestError(f'Test must be called from within a suite')
        else:
            self.model.current_suite.add(self)

    def run(self, fixtures: Set[Fixture]):
        # TODO: Maybe log something here about the test that's running
        log.info('Running test %s', self.name)
        args, kwargs = self._replace_args(fixtures)
        try:
            result = self.func(*args, **kwargs)
        except Exception as e:
            # TODO: Log failure, here or in recursive run_suite? (probably here)
            log.info('Test %s failed', self.name)
            return False, e
        else:
            # TODO: log success
            log.info('Test %s passed', self.name)
            return True, result

    @property
    def data(self):
        return self.func

    def add(self, node):
        raise CutestError('Tests should not have child nodes')


class Runner(metaclass=ABCMeta):

    def __init__(self):
        self.passes: List[Tuple[Test, None]] = []
        self.fails: List[Tuple[Test, Exception]] = []
        self.__suites_ran: Set[Suite] = set()

    def run_collection(self, collection: 'Collection'):
        for model in collection.models:
            self.run_model(model)
        for suite in collection.suites:
            self.run_suite(suite)
        self.run_tests(collection.tests)

    def run_model(self, model: Model):
        log.info('Running model %s', model)
        for suite in model.suites:
            self.run_suite(suite)

    def run_tests(self, tests: Iterable[Test]):
        tests = set(tests)
        pruned_suites = self.pruned_suites(tests)
        for suite in pruned_suites:
            self.run_suite(suite)

    def pruned_suites(self, tests: Set[Test]):
        raw_suites: Set[Suite] = set(copy(t.root) for t in tests)
        assert None not in raw_suites
        for suite in raw_suites:
            suite.initialize()
            if suite.prune_all_but(tests):
                yield suite

    def run_suite(self, suite: Suite):
        assert suite.root is suite
        if suite in self.__suites_ran:
            log.warning('Suite %s was already run', suite)
        else:
            log.info('Running test suite %s', suite)
            suite.print_graph()
            self.__suites_ran.add(suite)
            self.run(suite, fixtures=set())

    def run(self, node: Union['RunnerNode', Suite], fixtures: Set[Fixture]):
        """
        All public run methods must go through this one
        """
        self._recursive_run(node, fixtures)

    def _recursive_run(self, node: Node, fixtures: Set[Fixture]):
        if isinstance(node, Test):
            self._run_test(node, fixtures)
        elif isinstance(node, Fixture):
            self._run_fixture(node, fixtures)
        elif isinstance(node, Suite):
            self._run_suite(node, fixtures)
        elif isinstance(node, RunnerNode):
            self._run_runner_node(node, fixtures)
        else:
            assert False

    def _run_test(self, node: Test, fixtures: Set[Fixture]):
        success, result = node.run(fixtures)
        if success:
            assert result is None, 'Tests should not return anything'
            self.passes.append((node, result))
        else:
            # TODO: Should be BaseException?
            assert isinstance(result, Exception)
            self.fails.append((node, result))
        assert len(node.children) == 0

    def _run_fixture(self, node: Fixture, fixtures: Set[Fixture]):
        node.initialize(fixtures)
        fixtures.add(node)
        with node.context_manager():
            for child in node.children:
                self._recursive_run(child, fixtures)
        fixtures.remove(node)

    def _run_suite(self, node: Suite, fixtures: Set[Fixture]):
        assert node.root is node, "Cannot handle sub-suites yet"
        for child in node.children:
            self._recursive_run(child, fixtures)

    def _run_runner_node(self, node: 'RunnerNode', fixtures):
        runner = node.runner_cls()
        # FIXME: this is an infinite loop. Change run to avoid this
        runner.run(node, fixtures)
        self.passes += runner.passes
        self.fails += runner.fails


class SerialRunner(Runner):
    pass


class ExecutorRunner(Runner):

    def __init__(self, executor: Executor):
        super().__init__()
        # TODO: Should this be an executor class??
        self.executor: Executor = executor
        self.futures: List[Future] = []

    def run(self, node: Union['RunnerNode', Suite], fixtures: Set[Fixture]):
        with self.executor:
            super().run(node, fixtures)
        # FIXME: Do we need (to wait for / to even save) futures? Do they return anything?

    def _run_test(self, node: Test, fixtures: Set[Fixture]):
        future = self.executor.submit(super()._run_test, node, fixtures)
        self.futures.append(future)

    def _run_fixture(self, node: Test, fixtures: Set[Fixture]):
        future = self.executor.submit(super()._run_test, node, fixtures)
        self.futures.append(future)


class RunnerNode(Node):

    def __init__(self, model: Model, runner_cls: Type[Runner]):
        super().__init__(model)
        self.runner_cls: Type[Runner] = runner_cls

    # Extract next 2? methods into non-leaf Node class
    def __enter__(self):
        self.model.current_suite.add(self)
        self.model.current_suite.parent_stack.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        node = self.model.current_suite.parent_stack.pop()
        assert node is self
        return False

    @property
    def data(self):
        return self.runner_cls


class Collection:

    def __init__(self):
        self.models: List[Model] = []
        self.suites: List[Suite] = []
        self.tests: List[Test] = []

    def add_tests(self, test_ids: List[str]):
        for test_id in test_ids:
            parts = test_id.split('.')
            module = importlib.import_module(parts[0])
            mod_index = 1
            for i in range(1, len(parts)):
                mod_name = '.'.join(parts[1:i])
                try:
                    module = importlib.import_module(mod_name)
                except (ModuleNotFoundError, ValueError):
                    mod_index = i
                    break
            rest = parts[mod_index:]
            if rest:
                obj, *attrs = rest
                obj = getattr(module, obj)
                if isinstance(obj, Model):
                    assert len(attrs) == 0, f"Specifying tests beneath a Model {obj} is not supported, {test_id}"
                    obj.initialize()
                    self.models.append(obj)
                elif isinstance(obj, Suite):
                    obj.model.initialize()
                    if len(attrs):
                        raise CutestError(f"Specifying tests beneath a Suite {obj} is not YET supported, {test_id}")
                    else:
                        self.suites.append(obj)
                elif isinstance(obj, TestDefinition):
                    obj.model.initialize()
                    self.tests += obj.calls
            else:
                models: List[Model] = [obj for obj in vars(module).values() if isinstance(obj, Model)]
                for model in models:
                    model.initialize()
                    self.models.append(model)


class CutestError(Exception):
    pass


def main(argv=None):
    if argv is not None:
        # If called programmatically (i.e. tests), we don't want to override logging info
        logging.basicConfig(level=logging.INFO)
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Run unit tests with cutest')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('tests', nargs='*',
                        help='a list of any number of test modules, suites, and test methods')
    options = parser.parse_args(argv)
    collection = Collection()
    collection.add_tests(options.tests)
    runner = SerialRunner()
    runner.run_collection(collection)
