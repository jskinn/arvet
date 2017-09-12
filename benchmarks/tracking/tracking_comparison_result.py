#Copyright (c) 2017, John Skinner
import numpy as np
import pickle
import bson
import trials.slam.tracking_state
import core.trial_comparison


class TrackingComparisonResult(core.trial_comparison.TrialComparisonResult):
    """
    The results of comparing tracking between two trial results.
    """

    def __init__(self, benchmark_id, trial_result_id, reference_id, changes, settings, initializing_is_lost=True,
                 id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id,
                         reference_id=reference_id,  id_=id_, **kwargs)
        self._changes = changes
        self._settings = settings
        self._initializing_is_lost = initializing_is_lost

    @property
    def initializing_is_lost(self):
        """
        This is a configuration parameter, which changes the way
        statistics are counted over the changes. Does not affect the changes themselves.
        Note that it is not saved, always resets to 'True' for newly deserialized objects.
        :return: 
        """
        return self._initializing_is_lost

    @initializing_is_lost.setter
    def initializing_is_lost(self, initializing_is_lost):
        self._initializing_is_lost = bool(initializing_is_lost)

    @property
    def changes(self):
        return self._changes

    @property
    def new_tracking_count(self):
        return np.sum([1 for ref_state, comp_state in self.changes.values()
                       if not self.is_lost(comp_state) and self.is_lost(ref_state)])

    @property
    def new_lost_count(self):
        return np.sum([1 for ref_state, comp_state in self.changes.values()
                       if self.is_lost(comp_state) and not self.is_lost(ref_state)])

    @property
    def settings(self):
        return self._settings

    def is_lost(self, state):
        return state is trials.slam.tracking_state.TrackingState.LOST or (
            self.initializing_is_lost and state is trials.slam.tracking_state.TrackingState.NOT_INITIALIZED)

    def serialize(self):
        output = super().serialize()
        output['changes'] = bson.Binary(pickle.dumps(self.changes, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'changes' in serialized_representation:
            kwargs['changes'] = pickle.loads(serialized_representation['changes'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)