import unittest

import cutest
from test import sample


class RunnerTest(unittest.TestCase):

    def test_prune(self):
        test_output = cutest.default_output_stream()
        runner = cutest.SerialRunner(test_output, verbosity=1)
        sample.my_suite.initialize()
        pruned = list(runner.pruned_suites(sample.test_1.calls))
        self.assertEqual(len(pruned), 1)
