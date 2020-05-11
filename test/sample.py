import time

from cutest import Model

cu = Model()


@cu.suite
def my_suite():
    test_1()
    with fix_1():
        test_2()
        with fix_2() as f2:
            test_3(f2)
            with cu.threads():
                test_4()
                test_1()
        test_5()


@cu.suite
def mixed_suite():
    test_1()
    with fix_1():
        test_2()
        test_fail()
        with fix_2() as f2:
            test_3(f2)
            with cu.threads():
                test_4()
                test_1()
                test_fail()
        test_fail()
        test_5()


@cu.test
def test_1():
    print('test 1')


@cu.test
def test_2():
    print('test 2')


@cu.test
def test_3(f2):
    print('test 3 start')
    f2.fix_method()
    print('test 3 end')


@cu.test
def test_4():
    print('test 4 start')
    time.sleep(.1)
    print('test 4 end')


@cu.test
def test_5():
    print('test 5')


@cu.test
def test_fail():
    print('asserting false')
    assert False


@cu.fixture
def fix_1():
    print('enter fix_1')
    yield
    print('exit fix_1')


@cu.fixture
class fix_2:

    def __enter__(self):
        print('enter fix_2')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exit fix_2')
        return False

    def fix_method(self):
        print('fix_2 method')
