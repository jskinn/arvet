import copy
import core.trial_result


class LoopClosureTrialResult(core.trial_result.TrialResult):
    """
    The results of running a place recognition system.
    That is, a confusion matrix of place identification probabilities (I think)
    """
    def __init__(self, image_source_id, system_id, trained_state_id, success, system_settings, id_=None,
                 loop_closures=None, dataset_repeats=0, **kwargs):
        if isinstance(dataset_repeats, int):
            self._dataset_repeats = dataset_repeats
        else:
            self._dataset_repeats = 0
        if loop_closures is not None and len(loop_closures) > 0:
            self._loop_closures = loop_closures
        else:
            self._loop_closures = None
            success = False
        super().__init__(image_source_id, system_id, trained_state_id, success, system_settings, id_=id_)

    @property
    def dataset_repeats(self):
        return self._dataset_repeats

    @dataset_repeats.setter
    def dataset_repeats(self, repeats):
        if isinstance(repeats, int):
            self._dataset_repeats = repeats

    @property
    def loop_closures(self):
        return self._loop_closures

    def serialize(self):
        serialized = super().serialize()
        if self._loop_closures is not None:
            serialized['loop_closures'] = copy.deepcopy(self._loop_closures)
        serialized['dataset_repeats'] = self._dataset_repeats
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'loop_closures' in serialized_representation:
            kwargs['loop_closures'] = serialized_representation['loop_closures']
        if 'dataset_repeats' in serialized_representation:
            kwargs['dataset_repeats'] = serialized_representation['dataset_repeats']
        return super().deserialize(serialized_representation, db_client, **kwargs)
