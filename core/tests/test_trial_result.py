from unittest import TestCase
import core.trial_result


class TestTrialResult(TestCase):

    def test_no_id(self):
        core.trial_result.TrialResult(1, 2, None, True, {})

    def test_identifier(self):
        trial_result = core.trial_result.TrialResult(1, 2, None, True, {}, id_=123)
        self.assertEquals(trial_result.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a': 1, 'b': 2, 'c': 3}
        trial_result = core.trial_result.TrialResult(1, 2, None, True, {}, **kwargs)
        self.assertEquals(trial_result.identifier, 1234)

    def test_serialize_and_deserialize(self):
        trial_result1 = core.trial_result.TrialResult(1, 2, None, True, {}, id_=12345)
        s_trial_result1 = trial_result1.serialize()

        trial_result2 = core.trial_result.TrialResult.deserialize(s_trial_result1)
        s_trial_result2 = trial_result2.serialize()

        self._assert_models_equal(trial_result1, trial_result2)
        self.assertEquals(s_trial_result1, s_trial_result2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            trial_result2 = core.trial_result.TrialResult.deserialize(s_trial_result2)
            s_trial_result2 = trial_result2.serialize()
            self._assert_models_equal(trial_result1, trial_result2)
            self.assertEquals(s_trial_result1, s_trial_result2)

    def _assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two dataset models are equal
        :param trial_result1: Dataset
        :param trial_result2: Dataset
        :return:
        """
        if (not isinstance(trial_result1, core.trial_result.TrialResult) or
                not isinstance(trial_result2, core.trial_result.TrialResult)):
            self.fail('object was not a TrialResult')
        self.assertEquals(trial_result1.identifier, trial_result2.identifier)
        self.assertEquals(trial_result1.image_source_id, trial_result2.image_source_id)
        self.assertEquals(trial_result1.system_id, trial_result2.system_id)
        self.assertEquals(trial_result1.success, trial_result2.success)
