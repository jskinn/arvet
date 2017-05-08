import numpy as np
import core.benchmark
import benchmarks.matching.matching_result as match_res


class BenchmarkLoopClosure(core.benchmark.Benchmark):
    """
    A tool for benchmarking loop closures as correct or not.
    """

    def __init__(self, distance_threshold, trivial_closure_index_distance=2, id_=None):
        """
        Create an

        The key parameter here is how far geometrically is considered a correct match.
        Larger parameter values are more forgiving of inaccuracy in the algorithm,
        but it all depends on the scale of the dataset, which is hard to quantify.
        TODO: Might need to convert to an actual distance units,
        to handle scale between generated an real world datasets.

        :param distance_threshold: The maximum distance between points that is considered a valid closure
        :param trivial_closure_index_distance: The minimum difference between closure indexes,
        below which closures are considered trivial.
        This will have different scales for datasets with indexes vs timestamps.
        Set to 0 to accept all trivial loop closures.
        """
        super().__init__(id_=id_)
        self._threshold_distance = distance_threshold
        self._trivial_closure_distance = trivial_closure_index_distance

    @property
    def threshold_distance(self):
        return self._threshold_distance

    @property
    def trivial_closure_index_distance(self):
        return self._trivial_closure_distance

    def get_settings(self):
        return {
            'threshold_distance': self.threshold_distance,
            'trivial_closure_distance': self.trivial_closure_index_distance
        }

    def serialize(self):
        output = super().serialize()
        output['threshold_distance'] = self.threshold_distance
        output['trivial_closure_index_distance'] = self.trivial_closure_index_distance
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'threshold_distance' in serialized_representation:
            kwargs['distance_threshold'] = serialized_representation['threshold_distance']
        if 'trivial_closure_index_distance' in serialized_representation:
            kwargs['trivial_closure_index_distance'] = serialized_representation['trivial_closure_index_distance']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    def get_trial_requirements(self):
        return {'success': True, 'loop_closures': {'$exists': True, '$ne': []}}

    def benchmark_results(self, trial_result):
        """

        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        matches = {}
        threshold_distance_squared = self.threshold_distance * self.threshold_distance
        trivial_index_distance_squared = self.trivial_closure_index_distance * self.trivial_closure_index_distance

        poses = trial_result.get_ground_truth_camera_poses()
        indexes = list(poses.keys())
        indexes.sort()
        # First, check all the produced matches to see if they were correct
        for idx, closure_index in trial_result.get_loop_closures().items():
            # TODO: Maybe need to resolve differences between closure indexes and pose indexes
            image_location = poses[idx].location
            match_location = poses[closure_index].location
            indexes.remove(idx)

            diff = match_location - image_location
            square_dist = np.dot(diff, diff)

            index_diff = idx - closure_index
            index_square_dist = index_diff*index_diff

            if (square_dist < threshold_distance_squared and
                    index_square_dist > trivial_index_distance_squared):
                matches[idx] = match_res.MatchType.TRUE_POSITIVE
            else:
                # wasn't close enough to current location, or was trivial.
                matches[idx] = match_res.MatchType.FALSE_POSITIVE

        # Now go through all the remaining indexes to make sure there was not a match there
        for unmatched_idx in indexes:
            unmatched_pose = poses[unmatched_idx].location
            found_closure = False
            for index, pose in poses.items():
                if index < unmatched_idx - self.trivial_closure_index_distance:
                    diff = unmatched_pose - pose.location
                    square_dist = np.dot(diff, diff)

                    if square_dist < threshold_distance_squared:
                        matches[unmatched_idx] = match_res.MatchType.FALSE_NEGATIVE
                        found_closure = True
                        break
            if not found_closure:
                matches[unmatched_idx] = match_res.MatchType.TRUE_NEGATIVE

        return match_res.MatchBenchmarkResult(benchmark_id=self.identifier,
                                              trial_result_id=trial_result.identifier,
                                              matches=matches,
                                              settings=self.get_settings())
