import numpy as np
import itertools
import core.benchmark


class BoundingBoxOverlapBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    Simple results for a list of bounding box scores.
    Produces summaries of the distribution of scores by class,
    as well as an aggregate summary over all the classes.
    """

    def __init__(self, benchmark_id, trial_result_id, class_bbox_scores, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)
        self._class_bbox_scores = class_bbox_scores
        self._settings = settings

    @property
    def classes(self):
        return list(self._class_bbox_scores.keys())

    @property
    def scores_by_class(self):
        return self._class_bbox_scores

    @property
    def all_bbox_scores(self):
        return list(itertools.chain.from_iterable(self.scores_by_class.values()))

    @property
    def mean_score(self):
        return np.mean(self.all_bbox_scores)

    @property
    def median_score(self):
        return np.median(self.all_bbox_scores)

    @property
    def std_score(self):
        return np.std(self.all_bbox_scores)

    @property
    def min_score(self):
        return np.min(self.all_bbox_scores)

    @property
    def max_score(self):
        return np.max(self.all_bbox_scores)

    @property
    def settings(self):
        return self._settings

    def get_mean_score(self, class_name):
        if class_name in self.scores_by_class:
            return np.mean(self.scores_by_class[class_name])
        return 0

    def get_median_score(self, class_name):
        if class_name in self.scores_by_class:
            return np.median(self.scores_by_class[class_name])
        return 0

    def get_std_score(self, class_name):
        if class_name in self.scores_by_class:
            return np.std(self.scores_by_class[class_name])
        return 0

    def get_min_score(self, class_name):
        if class_name in self.scores_by_class:
            return np.min(self.scores_by_class[class_name])
        return 0

    def get_max_score(self, class_name):
        if class_name in self.scores_by_class:
            return np.max(self.scores_by_class[class_name])
        return 0

    def serialize(self):
        output = super().serialize()
        output['class_bbox_scores'] = self.scores_by_class
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'class_bbox_scores' in serialized_representation:
            kwargs['class_bbox_scores'] = serialized_representation['class_bbox_scores']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
