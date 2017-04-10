import core.benchmark


class BenchmarkPrecisionRecallResult(core.benchmark.BenchmarkResult):
    """
    Average Trajectory Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, true_positives, false_positives,
                 true_negatives, false_negatives, settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_)

        self._true_positives = true_positives
        self._false_positives = false_positives
        self._true_negatives = true_negatives
        self._false_negatives = false_negatives
        self._settings = settings

    @property
    def true_positives(self):
        return self._true_positives

    @property
    def false_positives(self):
        return self._false_positives

    @property
    def true_negatives(self):
        return self._true_negatives

    @property
    def false_negatives(self):
        return self._false_negatives

    @property
    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['true_positives'] = self.true_positives
        output['false_positives'] = self.false_positives
        output['true_negatives'] = self.true_negatives
        output['false_negatives'] = self.false_negatives
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'true_positives' in serialized_representation:
            kwargs['true_positives'] = serialized_representation['true_positives']
        if 'false_positives' in serialized_representation:
            kwargs['false_positives'] = serialized_representation['false_positives']
        if 'true_negatives' in serialized_representation:
            kwargs['true_negatives'] = serialized_representation['true_negatives']
        if 'false_negatives' in serialized_representation:
            kwargs['false_negatives'] = serialized_representation['false_negatives']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)
