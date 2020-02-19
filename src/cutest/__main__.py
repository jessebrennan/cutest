import argparse
import sys


def main(argv):
    # module_name = argv[0]
    # module = importlib.import_module(module_name)
    # suites = []
    # for member in inspect.getmembers(module):
    #     if isinstance(member, Suite):
    #         suites.append(member)
    # for suite in suites:
    #     suite.initialize()
    # runner = Runner()
    # runner.run_suites(suites)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='output verbosity; more vs (up to 3) for more verbosity')
    parser.add_argument('tests', nargs='*',
                        help='a list of any number of test modules, classes, and test methods')
    options = parser.parse_args(argv)


if __name__ == '__main__':
    main(sys.argv[1:])
