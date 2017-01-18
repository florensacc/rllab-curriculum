import unittest
import tensorflow as tf

from misc.testing_utils import are_np_arrays_equal, are_np_array_lists_equal


class TFTestCase(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.get_default_session() or tf.Session()
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()

    def tearDown(self):
        self.sess_context.__exit__(None, None, None)
        self.sess.close()

    def assertNpEqual(self, np_arr1, np_arr2, msg="Numpy arrays not equal."):
        self.assertTrue(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpNotEqual(self, np_arr1, np_arr2, msg="Numpy arrays equal"):
        self.assertFalse(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpArraysEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are not equal."
    ):
        self.assertTrue(are_np_array_lists_equal(np_arrays1, np_arrays2), msg)

    def assertNpArraysNotEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are equal."
    ):
        self.assertFalse(are_np_array_lists_equal(np_arrays1, np_arrays2), msg)

    def assertParamsEqual(self, network1, network2):
        self.assertNpArraysEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are not equal.",
        )

    def assertParamsNotEqual(self, network1, network2):
        self.assertNpArraysNotEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are equal.",
        )

