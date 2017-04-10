import core.benchmark
import core.loop_closure_detection
import benchmarks.precision_recall
import util.geometry


class BenchmarkPrecisionRecallLoopClosure(core.benchmark.Benchmark):
    """
    A tool for benchmarking loop closures as correct or not.
    """

    def __init__(self, distance_threshold, allow_trivial_closures=False):
        """
        Create an

        The key parameter here is how far geometrically is considered a correct match.
        Larger parameter values are more forgiving of inaccuracy in the algorithm,
        but it all depends on the scale of the dataset, which is hard to quantify.
        TODO: Might need to convert to an actual distance units,
        to handle scale between generated an real world datasets.

        :param distance_threshold: The maximum distance between points that is considered a valid closure
        :param allow_trivial_closures: Whether we allow trivial loop closures, such as between successive frames.
        """
        self._threshold_distance_squared = distance_threshold * distance_threshold
        self._allow_trivial_loop_closures = allow_trivial_closures

    @property
    def identifier(self):
        return 'LoopClosurePrecisionRecall'

    @property
    def threshold_distance(self):
        return self._threshold_distance_squared

    @property
    def allow_trivial_loop_closures(self):
        return self._allow_trivial_loop_closures

    def get_settings(self):
        return {
            'threshold_distance_squared': self.threshold_distance,
            'trivial_closures': self.allow_trivial_loop_closures
        }

    def get_trial_requirements(self):
        return {'success': True, 'loop_closures': {'$exists': True, '$ne': []}}

    def benchmark_results(self, dataset_images, trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if isinstance(trial_result, core.loop_closure_detection.LoopClosureTrialResult):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for current_idx, closure_index in trial_result.loop_closures:
                current_dataset_idx = current_idx % len(dataset_images)
                current_image = dataset_images[current_dataset_idx]   # Index > len means the dataset looped
                if closure_index >= 0:
                    # Detected a loop closure, was it right?
                    closure_index = closure_index % len(dataset_images)
                    matched_image = dataset_images[closure_index]
                    square_dist = util.geometry.square_length(current_image.camera_location -
                                                              matched_image.camera_location)
                    if square_dist < self._threshold_distance_squared:
                        if self.allow_trivial_loop_closures:
                            true_positives += 1
                        else:
                            # Need to verify that the match is non-trivial, that is it's not close to the source index
                            was_trivial = True
                            for intermediate_idx in range(closure_index, current_dataset_idx):
                                intermediate_image = dataset_images[intermediate_idx]
                                square_intermediate_dist = util.geometry.square_length(
                                    intermediate_image.camera_location - current_image.camera_location)
                                if square_intermediate_dist >= self._threshold_distance_squared:
                                    was_trivial = False
                                    break
                            if was_trivial:
                                # We count trivial matches as false positives
                                false_positives += 1
                            else:
                                true_positives += 1
                    else:
                        # wasn't close enough to current location
                        false_positives += 1
                else:
                    # Did not detect a loop closure, was there one to find?
                    was_loop_closure = False
                    was_trivial = True

                    # Loop backwards
                    for intermediate_idx in range(current_dataset_idx, 0, -1):
                        intermediate_image = dataset_images[intermediate_idx]
                        square_intermediate_dist = util.geometry.square_length(current_image.camera_location -
                                                                               intermediate_image.camera_location)
                        if square_intermediate_dist < self._threshold_distance_squared:
                            if not was_trivial or self._allow_trivial_loop_closures:
                                was_loop_closure = True
                                break
                        elif was_trivial:
                            # We've moved beyond the distance for
                            was_trivial = False
                    if was_loop_closure:
                        false_negatives += 1
                    else:
                        true_negatives += 1

            return benchmarks.precision_recall.BenchmarkPrecisionRecallResult(benchmark_id=self.identifier,
                                                                              trial_result_id=trial_result.identifier,
                                                                              true_positives=true_positives,
                                                                              false_positives=false_positives,
                                                                              true_negatives=true_negatives,
                                                                              false_negatives=false_negatives,
                                                                              settings=self.get_settings())
        else:
            return core.benchmark.FailedBenchmark(benchmark_id=self.identifier,
                                                  trial_result_id=trial_result.identifier,
                                                  reason='Trial was not a loop closure trial')
