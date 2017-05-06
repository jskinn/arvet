import enum
import bson
import pickle
import core.benchmark_comparison


# Can we change from True Positive to False Positive? That implies the ground truth is different.
class MatchChanges(enum.Enum):
    LOST_TO_TRUE_POSITIVE = 0
    TRUE_POSITIVE_TO_LOST = 1
    REMAIN_TRUE_POSITIVE = 2
    TRUE_POSITIVE_TO_FALSE_POSITIVE = 3
    TRUE_POSITIVE_TO_TRUE_NEGATIVE = 4
    TRUE_POSITIVE_TO_FALSE_NEGATIVE = 5

    LOST_TO_FALSE_POSITIVE = 6
    FALSE_POSITIVE_TO_LOST = 7
    REMAIN_FALSE_POSITIVE = 8
    FALSE_POSITIVE_TO_TRUE_POSITIVE = 9
    FALSE_POSITIVE_TO_TRUE_NEGATIVE = 10
    FALSE_POSITIVE_TO_FALSE_NEGATIVE = 11

    LOST_TO_TRUE_NEGATIVE = 12
    TRUE_NEGATIVE_TO_LOST = 13
    REMAIN_TRUE_NEGATIVE = 14
    TRUE_NEGATIVE_TO_TRUE_POSITIVE = 15
    TRUE_NEGATIVE_TO_FALSE_POSITIVE = 16
    TRUE_NEGATIVE_TO_FALSE_NEGATIVE = 17

    LOST_TO_FALSE_NEGATIVE = 18
    FALSE_NEGATIVE_TO_LOST = 19
    REMAIN_FALSE_NEGATIVE = 20
    FALSE_NEGATIVE_TO_TRUE_POSITIVE = 21
    FALSE_NEGATIVE_TO_FALSE_POSITIVE = 22
    FALSE_NEGATIVE_TO_TRUE_NEGATIVE = 23


# Constant sets for different groups of transitions
NEW_FOUND = frozenset((
    MatchChanges.LOST_TO_TRUE_POSITIVE,
    MatchChanges.LOST_TO_FALSE_POSITIVE,
    MatchChanges.LOST_TO_TRUE_NEGATIVE,
    MatchChanges.LOST_TO_FALSE_NEGATIVE
))
NEW_LOST = frozenset((
    MatchChanges.TRUE_POSITIVE_TO_LOST,
    MatchChanges.FALSE_POSITIVE_TO_LOST,
    MatchChanges.TRUE_NEGATIVE_TO_LOST,
    MatchChanges.FALSE_NEGATIVE_TO_LOST
))
UNCHANGED = frozenset((
    MatchChanges.REMAIN_TRUE_POSITIVE,
    MatchChanges.REMAIN_FALSE_POSITIVE,
    MatchChanges.REMAIN_TRUE_NEGATIVE,
    MatchChanges.REMAIN_FALSE_NEGATIVE
))
NEW_TRUE_POSITIVES = frozenset((
    MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE,
    MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE,
    MatchChanges.FALSE_NEGATIVE_TO_TRUE_POSITIVE,
    MatchChanges.LOST_TO_TRUE_POSITIVE
))
NEW_FALSE_POSITIVES = frozenset((
    MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE,
    MatchChanges.TRUE_NEGATIVE_TO_FALSE_POSITIVE,
    MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE,
    MatchChanges.LOST_TO_FALSE_POSITIVE
))
NEW_TRUE_NEGATIVES = frozenset((
    MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE,
    MatchChanges.FALSE_POSITIVE_TO_TRUE_NEGATIVE,
    MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE,
    MatchChanges.LOST_TO_TRUE_NEGATIVE
))
NEW_FALSE_NEGATIVES = frozenset((
    MatchChanges.TRUE_POSITIVE_TO_FALSE_NEGATIVE,
    MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE,
    MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE,
    MatchChanges.LOST_TO_FALSE_NEGATIVE
))
STRANGE_TRANSITIONS = frozenset((
    MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE,
    MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE,
    MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE,
    MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE,
    MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE,
    MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE,
    MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE,
    MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE
))


class MatchingComparisonResult(core.benchmark_comparison.BenchmarkComparisonResult):
    """
    Results of comparing two sets of match records.
    This identifies occasions when something was previously matched one way, and is later matched another.
    """

    def __init__(self, benchmark_comparison_id, benchmark_result, reference_benchmark_result,
                 match_changes, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_comparison_id=benchmark_comparison_id, benchmark_result=benchmark_result,
                         reference_benchmark_result=reference_benchmark_result, id_=id_, **kwargs)
        self._match_changes = match_changes
        self._settings = settings

    @property
    def match_changes(self):
        return self._match_changes

    @property
    def new_true_negatives(self):
        return self.count_changes_of_types(NEW_TRUE_NEGATIVES)

    @property
    def new_false_negatives(self):
        return self.count_changes_of_types(NEW_FALSE_NEGATIVES)

    @property
    def new_true_positives(self):
        return self.count_changes_of_types(NEW_TRUE_POSITIVES)

    @property
    def new_false_positives(self):
        return self.count_changes_of_types(NEW_FALSE_POSITIVES)

    @property
    def new_missing(self):
        return self.count_changes_of_types(NEW_LOST)

    @property
    def new_found(self):
        return self.count_changes_of_types(NEW_FOUND)

    @property
    def num_unchanged(self):
        return self.count_changes_of_types(UNCHANGED)

    @property
    def settings(self):
        return self._settings

    def num_strange_transitions(self):
        """
        Count the number of unexpected changes which imply a difference in ground truth.
        You shouldn't get any of these if your comparison is between two similar datasets.
        :return: The number of unexpected transition types that occurred in the comparison
        """
        return self.count_changes_of_types(STRANGE_TRANSITIONS)

    def count_changes_of_types(self, change_types):
        return sum(change in change_types for idx, change in self.match_changes.items())

    def serialize(self):
        output = super().serialize()
        output['match_changes'] = bson.Binary(pickle.dumps(self.match_changes, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'match_changes' in serialized_representation:
            kwargs['match_changes'] = pickle.loads(serialized_representation['match_changes'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
