import unittest

import cutest
# FIXME: import should not be global
from test import sample


class SmokeTest(unittest.TestCase):

    def test_sample_module(self):
        runner = cutest.Runner()
        sample.cu.initialize()
        runner.run_model(sample.cu)

    def test_sample_suite(self):
        runner = cutest.Runner()
        sample.my_suite.initialize()
        runner.run_suite(sample.my_suite)

    def test_sample_tests(self):
        runner = cutest.Runner()
        runner.run_tests(sample.test_1.calls)
