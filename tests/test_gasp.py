"""
Example unit tests for gasp package
"""
import unittest
import desc.gasp

class gaspTestCase(unittest.TestCase):
    def setUp(self):
        self.message = 'Hello, world'

    def tearDown(self):
        pass

    def test_run(self):
        foo = desc.gasp.gasp(self.message)
        self.assertEqual(foo.run(), self.message)

    def test_failure(self):
        self.assertRaises(TypeError, desc.gasp.gasp)
        foo = desc.gasp.gasp(self.message)
        self.assertRaises(RuntimeError, foo.run, True)

if __name__ == '__main__':
    unittest.main()
