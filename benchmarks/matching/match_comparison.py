# Copyright (c) 2017, John Skinner
import util.associate
import core.benchmark_comparison
import benchmarks.matching.matching_result as match_res
import benchmarks.matching.match_comparison_result as match_comp_res


class BenchmarkMatchingComparison(core.benchmark_comparison.BenchmarkComparison):
    """
    A comparison benchmark for identifying changes in matches between similar runs.
    This is useful for comparing runs of many things that produce matching results,
    such as image labelling, or loop closure detection
    """

    def __init__(self, offset=0, max_difference=0.02, id_=None):
        super().__init__(id_=id_)
        self._offset = offset
        self._max_difference = max_difference

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def max_difference(self):
        return self._max_difference

    @max_difference.setter
    def max_difference(self, max_difference):
        self._max_difference = max_difference

    def get_settings(self):
        return {
            'offset': self.offset,
            'max_difference': self.max_difference
        }

    def serialize(self):
        output = super().serialize()
        output['offset'] = self.offset
        output['max_difference'] = self.max_difference
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'offset' in serialized_representation:
            kwargs['offset'] = serialized_representation['offset']
        if 'max_difference' in serialized_representation:
            kwargs['max_difference'] = serialized_representation['max_difference']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_benchmark_requirements(cls):
        return {'success': True, 'match_changes': {'$exists': True, '$ne': []}}

    def is_result_appropriate(self, benchmark_result):
        """
        Can this particular benchmark result be used in the benchmark?
        :param benchmark_result: 
        :return: 
        """
        return hasattr(benchmark_result, 'identifier') and hasattr(benchmark_result, 'matches')

    def compare_results(self, benchmark_result, reference_benchmark_result):
        """
        Compare two matching results, to detect any changes in achieved performance.
        This measures all the different kinds of changes
        
        :param benchmark_result: The test benchmark result
        :param reference_benchmark_result: The reference benchmark result to which the test one is compared for changes
        :return: MatchingComparison
        :rtype BenchmarkResult:
        """
        matching_indexes = util.associate.associate(benchmark_result.matches, reference_benchmark_result.matches,
                                                    self.offset, self.max_difference)

        match_changes = {}
        base_indexes = list(benchmark_result.matches.keys())
        ref_indexes = list(reference_benchmark_result.matches.keys())

        # Go through the matched base and reference indexes, and record the changes in match behviour
        for base_idx, ref_idx in matching_indexes:
            base_match_type = benchmark_result.matches[base_idx]
            ref_match_type = reference_benchmark_result.matches[ref_idx]
            base_indexes.remove(base_idx)
            ref_indexes.remove(ref_idx)
            change = get_change_type(ref_match_type, base_match_type)
            if change is not None:
                match_changes[ref_idx] = change

        # Add additional transitions for missing indexes
        for base_idx in base_indexes:
            base_match_type = benchmark_result.matches[base_idx]
            if base_match_type is match_res.MatchType.TRUE_POSITIVE:
                match_changes[base_idx] = match_comp_res.MatchChanges.LOST_TO_TRUE_POSITIVE
            elif base_match_type is match_res.MatchType.FALSE_POSITIVE:
                match_changes[base_idx] = match_comp_res.MatchChanges.LOST_TO_FALSE_POSITIVE
            elif base_match_type is match_res.MatchType.TRUE_NEGATIVE:
                match_changes[base_idx] = match_comp_res.MatchChanges.LOST_TO_TRUE_NEGATIVE
            elif base_match_type is match_res.MatchType.FALSE_NEGATIVE:
                match_changes[base_idx] = match_comp_res.MatchChanges.LOST_TO_FALSE_NEGATIVE
        for ref_idx in ref_indexes:
            ref_match_type = reference_benchmark_result.matches[ref_idx]
            if ref_match_type is match_res.MatchType.TRUE_POSITIVE:
                match_changes[ref_idx] = match_comp_res.MatchChanges.TRUE_POSITIVE_TO_LOST
            elif ref_match_type is match_res.MatchType.FALSE_POSITIVE:
                match_changes[ref_idx] = match_comp_res.MatchChanges.FALSE_POSITIVE_TO_LOST
            elif ref_match_type is match_res.MatchType.TRUE_NEGATIVE:
                match_changes[ref_idx] = match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_LOST
            elif ref_match_type is match_res.MatchType.FALSE_NEGATIVE:
                match_changes[ref_idx] = match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_LOST

        return match_comp_res.MatchingComparisonResult(benchmark_comparison_id=self.identifier,
                                                       benchmark_result=benchmark_result.identifier,
                                                       reference_benchmark_result=reference_benchmark_result.identifier,
                                                       match_changes=match_changes,
                                                       settings=self.get_settings())


def get_change_type(from_, to):
    """
    Map from one match type to another match type into MatchChanges enum values.
    really, this is just a dirty great mapping function. It might be more efficient
    as a dict or a list or something.
    
    :param from_: What the match was in the reference result
    :param to: What the match is in the test benchmark result
    :return: An enum value indicating what if any changes occurred to the match type.
    """
    if from_ is match_res.MatchType.TRUE_POSITIVE:
        if to is match_res.MatchType.TRUE_POSITIVE:
            return match_comp_res.MatchChanges.REMAIN_TRUE_POSITIVE
        elif to is match_res.MatchType.FALSE_POSITIVE:
            return match_comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE
        elif to is match_res.MatchType.TRUE_NEGATIVE:
            return match_comp_res.MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE
        elif to is match_res.MatchType.FALSE_NEGATIVE:
            return match_comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_NEGATIVE
    elif from_ is match_res.MatchType.FALSE_POSITIVE:
        if to is match_res.MatchType.TRUE_POSITIVE:
            return match_comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE
        elif to is match_res.MatchType.FALSE_POSITIVE:
            return match_comp_res.MatchChanges.REMAIN_FALSE_POSITIVE
        elif to is match_res.MatchType.TRUE_NEGATIVE:
            return match_comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_NEGATIVE
        elif to is match_res.MatchType.FALSE_NEGATIVE:
            return match_comp_res.MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE
    elif from_ is match_res.MatchType.TRUE_NEGATIVE:
        if to is match_res.MatchType.TRUE_POSITIVE:
            return match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE
        elif to is match_res.MatchType.FALSE_POSITIVE:
            return match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_POSITIVE
        elif to is match_res.MatchType.TRUE_NEGATIVE:
            return match_comp_res.MatchChanges.REMAIN_TRUE_NEGATIVE
        elif to is match_res.MatchType.FALSE_NEGATIVE:
            return match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE
    elif from_ is match_res.MatchType.FALSE_NEGATIVE:
        if to is match_res.MatchType.TRUE_POSITIVE:
            return match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_POSITIVE
        elif to is match_res.MatchType.FALSE_POSITIVE:
            return match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE
        elif to is match_res.MatchType.TRUE_NEGATIVE:
            return match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE
        elif to is match_res.MatchType.FALSE_NEGATIVE:
            return match_comp_res.MatchChanges.REMAIN_FALSE_NEGATIVE
    return None
