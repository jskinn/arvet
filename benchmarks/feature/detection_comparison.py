import numpy as np
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

        # First, we need to find images taken from the same place
        matching_timestamps = [
            (image_id1, image_id2)
            for image_id1 in trial_result.camera_poses.keys()
            for image_id2 in reference_trial_result.camera_poses.keys()
            if is_pose_similar(trial_result.camera_poses[image_id1], reference_trial_result.camera_poses[image_id2])
        ]
        if len(matching_timestamps) <= 0:
            return core.benchmark.FailedBenchmark(self.identifier, reference_trial_result.identifier,
                                                  "Given trials were never in the same place")
        # Then, for each pair of images, find changes in detected features
        results = []
        for trial_image_id, reference_image_id in matching_timestamps:
            points1 = {point.pt for point in trial_result.keypoints[trial_image_id]}
            points2 = {point.pt for point in reference_trial_result.keypoints[reference_image_id]}
            potential_matches = sorted(
                (point_dist(point1, point2), point1, point2)
                for point1 in points1
                for point2 in points2
                if point_dist(point1, point2) < self._acceptable_radius * self._acceptable_radius)
            point_matches = []
            for match in potential_matches:
                coords1 = match[1]
                coords2 = match[2]
                if coords1 in points1 and coords2 in points2:
                    points1.remove(coords1)
                    points2.remove(coords2)
                    point_matches.append((match[1], match[2]))
            results.append({
                'trial_image_id': trial_image_id,
                'reference_image_id': reference_image_id,
                'point_matches': point_matches,
                'new_trial_points': list(points1),
                'missing_reference_points': list(points2)
            })
        return FeatureDetectionComparisonResult(
            benchmark_id=self.identifier,
            trial_result_id=trial_result.identifier,
            reference_id=reference_trial_result.identifier,
            feature_changes=results)

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

    def __init__(self, benchmark_id, trial_result_id, reference_id, feature_changes, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(
            benchmark_id=benchmark_id,
            trial_result_id=trial_result_id,
            reference_id=reference_id,
            id_=id_, **kwargs)
        self._feature_changes = feature_changes

    @property
    def changes(self):
        return self._feature_changes

    def compute_intersection_over_union(self):
        """
        Provide a score for each image, intersection over union of detected feature points
        between the reference and trial images.
        This will be 1 if every point is re-detected with none left over.
        Detecting extra points, or failing to detect points reduce the ratio.
        :return:
        """
        return {changes['trial_image_id']: len(changes['point_matches']) / (len(changes['point_matches']) +
                                                                            len(changes['new_trial_points']) +
                                                                            len(changes['missing_reference_points']))
                for changes in self._feature_changes}

    def serialize(self):
        serialized = super().serialize()
        serialized['feature_changes'] = self._feature_changes
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'feature_changes' in serialized_representation:
            kwargs['feature_changes'] = serialized_representation['feature_changes']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def is_pose_similar(pose1, pose2):
    """
    Are two poses close enough together that we consider them the same
    :param pose1: The first transform
    :param pose2: The second transform
    :return: True if the two transforms are close together
    """
    trans_diff = pose1.location - pose2.location
    rot_diff1 = pose1.rotation_quat(w_first=True) - pose2.rotation_quat(w_first=True)
    rot_diff2 = pose1.rotation_quat(w_first=True) + pose2.rotation_quat(w_first=True)
    return np.dot(trans_diff, trans_diff) < 0.001 and (np.dot(rot_diff1, rot_diff1) < 0.001 or
                                                       np.dot(rot_diff2, rot_diff2) < 0.001)


def point_dist(point1, point2):
    """
    Get the square euclidean distance between feature points
    :param point1:
    :param point2:
    :return:
    """
    diff = (point1[0] - point2[0], point1[1] - point2[1])
    return diff[0] * diff[0] + diff[1] * diff[1]
