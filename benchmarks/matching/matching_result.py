# Copyright (c) 2017, John Skinner
import enum
import pickle
import bson
import core.benchmark


class MatchType(enum.Enum):
    """
    An enum for the 4 kinds of match for loop closures
    """
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    TRUE_NEGATIVE = 2
    FALSE_NEGATIVE = 3


class MatchBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    Benchmark results for matching some property in a test.
    For instance, matching class labels, or detecting loop closures.
    There is a lot of different use cases for this result type.
    
    Stores the match for each input over the full trial, so that we can compare.
    Note that this cannot generate precision-recall curves, I need a more general solution for that.
    """

    def __init__(self, benchmark_id, trial_result_id, matches, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)

        self._matches = matches
        self._settings = settings
        # Pre-calculate total matches of each type
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0
        for index, match_type in matches.items():
            if match_type == MatchType.TRUE_POSITIVE:
                self._true_positives += 1
            elif match_type == MatchType.FALSE_POSITIVE:
                self._false_positives += 1
            elif match_type == MatchType.TRUE_NEGATIVE:
                self._true_negatives += 1
            elif match_type == MatchType.FALSE_NEGATIVE:
                self._false_negatives += 1

    @property
    def matches(self):
        return self._matches

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
        output['matches'] = bson.Binary(pickle.dumps(self.matches, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'matches' in serialized_representation:
            kwargs['matches'] = pickle.loads(serialized_representation['matches'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
