import bson
import core.trial_comparison
import core.benchmark
import trials.feature_detection.feature_detector_result as detector_result


class FeatureDetectionComparison(core.trial_comparison.TrialComparison):

    def __init__(self, acceptable_radius=4, id_=None):
        super().__init__(id_=id_)
        self._acceptable_radius = acceptable_radius

    def is_trial_appropriate(self, trial_result):
        return isinstance(trial_result, detector_result.FeatureDetectorResult)

    def compare_trial_results(self, trial_result, reference_trial_result):
        if not isinstance(trial_result, detector_result.FeatureDetectorResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  "{0} is not a FeatureDetectorResult".format(trial_result.identifier))
        if not isinstance(reference_trial_result, detector_result.FeatureDetectorResult):
            return core.benchmark.FailedBenchmark(self.identifier, reference_trial_result.identifier,
                                                  "{0} is not a FeatureDetectorResult".format(
                                                      reference_trial_result.identifier))
        reference_feature_points = reference_trial_result.keypoints
        base_feature_points = trial_result.keypoints
        image_ids = set(reference_feature_points.keys()) & set(base_feature_points.keys())
        point_matches = {}
        missing_trial = {}
        missing_reference = {}
        for image_id in image_ids:
            points1 = {point.pt for point in trial_result.keypoints[image_id]}
            points2 = {point.pt for point in reference_trial_result.keypoints[image_id]}
            potential_matches = sorted(
                (point_dist(point1, point2), point1, point2)
                for point1 in points1
                for point2 in points2
                if point_dist(point1, point2) < self._acceptable_radius * self._acceptable_radius)
            point_matches[image_id] = []
            for match in potential_matches:
                coords1 = match[1].pt
                coords2 = match[2].pt
                if coords1 in points1 and coords2 in points2:
                    points1.remove(coords1)
                    points2.remove(coords2)
                    point_matches[image_id].append((match[1], match[2]))
            missing_trial[image_id] = list(points1)
            missing_reference[image_id] = list(points2)
        return FeatureDetectionComparisonResult(
            benchmark_id=self.identifier,
            trial_result_id=trial_result.identifier,
            reference_id=reference_trial_result.identifier,
            point_matches=point_matches,
            missing_trial=missing_trial,
            missing_reference=missing_reference)

    def serialize(self):
        serialized = super().serialize()
        serialized['acceptable_radius'] = self._acceptable_radius
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'acceptable_radius' in serialized_representation:
            kwargs['acceptable_radius'] = serialized_representation['acceptable_radius']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_trial_requirements(cls):
        pass


class FeatureDetectionComparisonResult(core.trial_comparison.TrialComparisonResult):

    def __init__(self, benchmark_id, trial_result_id, reference_id, point_matches, missing_trial, missing_reference,
                 id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(
            benchmark_id=benchmark_id,
            trial_result_id=trial_result_id,
            reference_id=reference_id,
            id_=id_, **kwargs)
        self._point_matches = point_matches
        self._missing_trial = missing_trial
        self._missing_reference = missing_reference

    def compute_intersection_over_union(self):
        """
        Provide a score for each image, intersection over union of detected feature points
        between the reference and trial images.
        This will be 1 if every point is re-detected with none left over.
        Detecting extra points, or failing to detect points reduce the ratio.
        :return:
        """
        return {image_id: len(self._point_matches[image_id]) / (len(self._point_matches[image_id]) +
                                                                len(self._missing_trial[image_id]) +
                                                                len(self._missing_reference[image_id]))
                for image_id in self._point_matches.keys()}

    def get_image_ids(self):
        return self._point_matches.keys()

    def get_for_image(self, image_id):
        if image_id in self._point_matches and image_id in self._missing_trial and image_id in self._missing_reference:
            return self._point_matches[image_id], self._missing_trial[image_id], self._missing_reference[image_id]
        return None, None, None

    def serialize(self):
        serialized = super().serialize()
        serialized['point_matches'] = {str(img_id): matches for img_id, matches in self._point_matches.items()}
        serialized['missing_trial'] = {str(img_id): points for img_id, points in self._missing_trial.items()}
        serialized['missing_reference'] = {str(img_id): points for img_id, points in self._missing_reference.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'point_matches' in serialized_representation:
            kwargs['point_matches'] = {bson.ObjectId(str_id): matches for str_id, matches
                                       in serialized_representation['point_matches'].items()}
        if 'missing_trial' in serialized_representation:
            kwargs['missing_trial'] = {bson.ObjectId(str_id): points for str_id, points
                                       in serialized_representation['missing_trial'].items()}
        if 'missing_reference' in serialized_representation:
            kwargs['missing_reference'] = {bson.ObjectId(str_id): points for str_id, points
                                           in serialized_representation['missing_reference'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)


def point_dist(point1, point2):
    diff = (point1[0] - point2[0], point1[1] - point2[1])
    return diff[0] * diff[0] + diff[1] * diff[1]
