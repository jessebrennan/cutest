import argparse
import sys

from cutest import Collection, Runner


def main(argv):
    parser = argparse.ArgumentParser(description='Run unit tests with cutest')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('tests', nargs='*',
                        help='a list of any number of test modules, suites, and test methods')
    options = parser.parse_args(argv)
    collection = Collection()
    collection.add_tests(options.tests)
    runner = Runner()
    runner.run_collection(collection)


if __name__ == '__main__':
    main(sys.argv[1:])
