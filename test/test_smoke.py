import unittest

import cutest
# FIXME: import should not be global
from test import sample


class SmokeTest(unittest.TestCase):

    def test_sample_module(self):
        runner = cutest.SerialRunner()
        sample.cu.initialize()
        runner.run_model(sample.cu)

    def test_sample_suite(self):
        runner = cutest.SerialRunner()
        sample.my_suite.initialize()
        runner.run_suite(sample.my_suite)

    def test_sample_tests(self):
        runner = cutest.SerialRunner()
        runner.run_tests(sample.test_1.calls)
