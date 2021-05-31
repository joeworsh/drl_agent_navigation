# Copyright Joe Worsham 2021

import numpy as np
import unittest

from joe_agents.er_buffer import ExperienceReplayBuffer


class ExperienceReplayBufferTest(unittest.TestCase):
    def test_sample(self):
        buffer = ExperienceReplayBuffer(10)
        for i in range(10):
            buffer.append(i, i, i, i, False)

        # verify that sample returns these entries in random order
        s, a, r, na, d = buffer.sample(5)
        self.assertFalse(np.all(np.diff(s.squeeze()) >= 0))
        self.assertFalse(np.all(np.diff(a.squeeze()) >= 0))
        self.assertFalse(np.all(np.diff(r.squeeze()) >= 0))
        self.assertFalse(np.all(np.diff(na.squeeze()) >= 0))
        self.assertFalse(np.any(d.squeeze()))

        # verify that two samples are not identical
        s2, a2, r2, na2, d2 = buffer.sample(5)
        self.assertFalse(np.array_equal(s, s2))
        self.assertFalse(np.array_equal(a, a2))
        self.assertFalse(np.array_equal(r, r2))
        self.assertFalse(np.array_equal(na, na2))
        self.assertTrue(np.array_equal(d, d2))

        # verify that 10 more samples will push out the old 10
        old_content = np.arange(10)
        for i in range(10, 20):
            buffer.append(i, i, i, i, True)

        s3, a3, r3, na3, d3 = buffer.sample(5)
        self.assertFalse(np.any(np.in1d(old_content, s3.squeeze())))
        self.assertFalse(np.any(np.in1d(old_content, a3.squeeze())))
        self.assertFalse(np.any(np.in1d(old_content, r3.squeeze())))
        self.assertFalse(np.any(np.in1d(old_content, na3.squeeze())))
        self.assertTrue(np.all(d3.squeeze()))

        # check that length works
        self.assertEquals(10, len(buffer))
        buffer.clear()
        self.assertEquals(0, len(buffer))
