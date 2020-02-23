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

    parser = argparse.ArgumentParser(description='Run unit tests with cutest')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('tests', nargs='*',
                       help='a list of any number of test modules, suites, and test methods')
    group.add_argument('-d', '--discover', metavar='DIRECTORY', default='.',
                       help='directory to start test discovery')
    options = parser.parse_args(argv)


if __name__ == '__main__':
    main(sys.argv[1:])
