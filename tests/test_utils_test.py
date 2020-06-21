from test_utils import *
import unittest


class TestUtilsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_cost(self):
        gt = "test"
        self.assertLess(
            cost("test", gt),
            cost("text", gt))

        self.assertLess(
            cost("ignore some test case ignore", gt),
            cost("top elf saw top", gt))
    

if __name__ == "__main__":
    unittest.main()
