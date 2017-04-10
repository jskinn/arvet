import core.benchmark
import core.comparison
import core.loop_closure_detection
import util.geometry as geom

# Quick enums for different results
_TRUE_POSITIVE = 0
_FALSE_POSITIVE = 1
_TRUE_NEGATIVE = 2
_FALSE_NEGATIVE = 3

class BenchmarkMatchingComparisonResult(core.comparison.ComparisonBenchmarkResult):
    """
    Results of changes in false/positive matches


    """

    def __init__(self, benchmark_id, trial_result_id, reference_id, new_true_positives, new_false_positives,
                 new_true_negatives, new_false_negatives, new_missing, new_found, settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, reference_id, True, id_)
        self._new_true_positives = new_true_positives
        self._new_false_positives = new_false_positives
        self._new_true_negatives = new_true_negatives
        self._new_false_negatives = new_false_negatives
        self._new_missing = new_missing
        self._new_found = new_found
        self._settings = settings

    @property
    def new_true_negatives(self):
        return self._new_true_negatives

    @property
    def new_false_negatives(self):
        return self._new_false_negatives

    @property
    def new_true_positives(self):
        return self._new_true_positives

    @property
    def new_false_positives(self):
        return self._new_false_positives

    @property
    def new_missing(self):
        return self._new_missing

    @property
    def new_found(self):
        return self._new_found

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['new_false_negatives'] = self.new_false_negatives
        output['new_true_negatives'] = self.new_true_negatives
        output['new_false_positives'] = self.new_false_positives
        output['new_true_positives'] = self.new_true_positives
        output['new_missing'] = self.new_missing
        output['new_found'] = self.new_found
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'new_false_positives' in serialized_representation:
            kwargs['new_false_positives'] = serialized_representation['new_false_positives']
        if 'new_true_positives' in serialized_representation:
            kwargs['new_true_positives'] = serialized_representation['new_true_positives']
        if 'new_true_negatives' in serialized_representation:
            kwargs['new_true_negatives'] = serialized_representation['new_true_negatives']
        if 'new_false_negatives' in serialized_representation:
            kwargs['new_false_negatives'] = serialized_representation['new_false_negatives']
        if 'new_missing' in serialized_representation:
            kwargs['new_missing'] = serialized_representation['new_missing']
        if 'new_found' in serialized_representation:
            kwargs['new_found'] = serialized_representation['new_found']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkMatchingComparison(core.comparison.ComparisonBenchmark):
    """
    A tool for comparing match performance between two algorithms
    """

    def __init__(self, distance_threshold, allow_trivial_closures=False):
        """
        Create a Mean Time Lost benchmark.
        No configuration at this stage, it is just counting tracking state changes.
        """
        self._threshold_distance_squared = distance_threshold * distance_threshold
        self._allow_trivial_loop_closures = allow_trivial_closures

    @property
    def identifier(self):
        return 'PlaceRecMatchComparison'

    @property
    def threshold_square_distance(self):
        return self._threshold_distance_squared

    @property
    def allow_trivial_loop_closures(self):
        return self._allow_trivial_loop_closures

    def get_settings(self):
        return {
            'threshold_distance_squared': self.threshold_square_distance,
            'trivial_closures': self.allow_trivial_loop_closures
        }

    def get_trial_requirements(self):
        return {'success': True, 'loop_closures': {'$exists': True, '$ne': []}}

    def compare_results(self, trial_result, reference_trial_result, reference_dataset_images):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if not isinstance(trial_result, core.loop_closure_detection.LoopClosureTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Trial was not a loop closure result')
        if not isinstance(reference_trial_result, core.loop_closure_detection.LoopClosureTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Reference trial was not loop closure result')
        new_false_positives = []
        new_true_positives = []
        new_false_negatives = []
        new_true_negatives = []
        new_missing = []
        new_found = []
        ref_idx = 0
        for trial_idx in range(0, len(trial_result.loop_closures)):
            if ref_idx >= len(reference_trial_result.loop_closures):
                break
            trial_img_idx, trial_closure = trial_result.loop_closures[trial_idx]
            ref_img_idx, ref_closure = reference_trial_result.loop_closures[ref_idx]

            if trial_img_idx > ref_img_idx:
                new_missing.append(ref_idx)
                ref_idx += 1
            elif trial_img_idx < ref_img_idx:
                new_found.append(trial_img_idx)
            else:   # trial_img_idx == ref_img_idx
                current_dataset_idx = trial_img_idx % len(reference_dataset_images)
                current_image = reference_dataset_images[current_dataset_idx]  # Index > len means the dataset looped

                trial_match = self.check_loop_closure(trial_closure, current_image.camera_location, current_dataset_idx,
                                                      reference_dataset_images)
                ref_match = self.check_loop_closure(ref_closure, current_image.camera_location, current_dataset_idx,
                                                      reference_dataset_images)

                if trial_match != ref_match:
                    if trial_match == _TRUE_POSITIVE:
                        new_true_positives.append(trial_img_idx)
                    elif trial_match == _FALSE_POSITIVE:
                        new_false_positives.append(trial_img_idx)
                    elif trial_match == _TRUE_NEGATIVE:
                        new_true_negatives.append(trial_img_idx)
                    elif trial_match == _FALSE_NEGATIVE:
                        new_false_negatives.append(trial_img_idx)
                ref_idx += 1

        return BenchmarkMatchingComparisonResult(benchmark_id=self.identifier,
                                                 trial_result_id=trial_result.identifier,
                                                 reference_id=reference_trial_result.identifier,
                                                 new_true_positives=new_true_positives,
                                                 new_false_positives=new_false_positives,
                                                 new_true_negatives=new_true_negatives,
                                                 new_false_negatives=new_false_negatives,
                                                 new_missing=new_missing,
                                                 new_found=new_found,
                                                 settings=self.get_settings())

    def check_loop_closure(self, closure_idx, current_location, current_dataset_idx, dataset_images):
        if closure_idx >= 0:
            # Detected a loop closure, was it right?
            closure_idx = closure_idx % len(dataset_images)
            matched_image = dataset_images[closure_idx]
            square_dist = geom.square_length(current_location - matched_image.camera_location)
            if square_dist < self.threshold_square_distance:
                if self.allow_trivial_loop_closures:
                    return _TRUE_POSITIVE
                else:
                    # Need to verify that the match is non-trivial, that is,
                    # There are intervening frames greater than the distance
                    was_trivial = True
                    for intermediate_idx in range(closure_idx, current_dataset_idx):
                        intermediate_image = dataset_images[intermediate_idx]
                        square_intermediate_dist = geom.square_length(intermediate_image.camera_location - current_location)
                        if square_intermediate_dist >= self._threshold_distance_squared:
                            was_trivial = False
                            break
                    if was_trivial:
                        # We count trivial matches as false positives
                        return _FALSE_POSITIVE
                    else:
                        return _TRUE_POSITIVE
            else:
                # wasn't close enough to current location
                return _FALSE_POSITIVE
        else:
            # Did not detect a loop closure, was there one to find?
            was_loop_closure = False
            was_trivial = True

            # Loop backwards
            for intermediate_idx in range(current_dataset_idx, 0, -1):
                intermediate_image = dataset_images[intermediate_idx]
                square_intermediate_dist = geom.square_length(current_location - intermediate_image.camera_location)
                if square_intermediate_dist < self._threshold_distance_squared:
                    if not was_trivial or self._allow_trivial_loop_closures:
                        was_loop_closure = True
                        break
                elif was_trivial:
                    # We've moved beyond the distance for trivial closures, all future closures are non-trivial
                    was_trivial = False
            if was_loop_closure:
                return _FALSE_NEGATIVE
            else:
                return _TRUE_NEGATIVE