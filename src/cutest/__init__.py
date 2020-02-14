import importlib
import inspect
import sys
from abc import ABC
from contextlib import contextmanager
from typing import List

from cutest.util import Stack


class _GraphNode(ABC):
    """
    Inherit to be allowed in the Suite graph
    """
    def __init__(self):
        self.graph: List[_GraphNode] = []

    @property
    def data(self):
        raise NotImplementedError

    def print_graph(self, depth=0):
        print('  ' * depth, self.__class__.__name__, self.data.__name__)
        for node in self.graph:
            node.print_graph(depth=depth + 1)


# FIXME: Should this inherit from _GraphNode?
class _Suite(_GraphNode):

    def __init__(self, func):
        self._func = func
        self.fixture_stack: Stack[_Fixture] = Stack()
        super().__init__()

    @property
    def data(self):
        return self._func

    def build_graph(self):
        global _current_suite
        _current_suite = self
        self._func()
        _current_suite = None

    def add(self, node: _GraphNode):
        if self.fixture_stack.empty():
            self.graph.append(node)
        else:
            self.fixture_stack.top().add(node)


_current_suite = None


def suite(func):
    return _Suite(func)


class _Fixture(_GraphNode):
    def __init__(self, cm):
        self._cm = cm
        self.graph: List[_GraphNode] = []
        self.args = None
        self.kwargs = None
        super().__init__()

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # FIXME this is wrong. We should actually return the result of self._cm(*args, **kwargs)
        return self

    @property
    def data(self):
        return self._cm

    def add(self, node):
        self.graph.append(node)

    def __enter__(self):
        global _current_suite
        _current_suite.add(self)
        _current_suite.fixture_stack.add(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_suite
        fixture_ = _current_suite.fixture_stack.pop()
        assert fixture_ is self
        return False

    def add_test(self, test_):
        self.graph.append(test_)


def fixture(obj):
    if inspect.isclass(obj):
        return _Fixture(obj)
    elif inspect.isgeneratorfunction(obj):
        cm = contextmanager(obj)
        return _Fixture(cm)
    else:
        raise CutestError('fixture must decorate a contextmanager or generator')


class _Test(_GraphNode):
    def __init__(self, func):
        self._func = func
        self.args = None
        self.kwargs = None
        super().__init__()

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        global _current_suite
        if _current_suite is None:
            raise CutestError(f'Test must be called from within a suite')
        else:
            _current_suite.add(self)

    @property
    def data(self):
        return self._func


def test(func):
    return _Test(func)


def main(argv):
    module_name = argv[0]
    module = importlib.import_module(module_name)
    suites = []
    for member in inspect.getmembers(module):
        if isinstance(member, _Suite):
            suites.append(member)
    for suite_ in suites:
        suite_.build_graph()


class CutestError(Exception):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
