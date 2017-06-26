import core.benchmark
import benchmarks.bounding_box_overlap.bounding_box_overlap_result as bbox_result


class BoundingBoxOverlapBenchmark(core.benchmark.Benchmark):
    """
    A simple benchmark for bounding boxes detected over images.
    Produces scores based on the area of overlap between the detected bounding box and the ground truth box.
    Precision is overlap / detected area
    Recall is overlap / ground truth area
    Final score is F1 score times detection confidence
    """

    def __init__(self, id_=None):
        """
        Create a new benchmark instance. No configuration.
        """
        super().__init__(id_=id_)

    def serialize(self):
        output = super().serialize()
        # output['initializing_is_lost'] = self._initializing_is_lost
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        # if 'initializing_is_lost' in serialized_representation:
        #    kwargs['initializing_is_lost'] = serialized_representation['initializing_is_lost']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_trial_requirements(cls):
        return {'success': True, 'tracking_stats': {'$exists': True, '$ne': []}}

    def is_trial_appropriate(self, trial_result):
        return (hasattr(trial_result, 'identifier') and
                hasattr(trial_result, 'get_ground_truth_bounding_boxes') and
                hasattr(trial_result, 'get_bounding_boxes'))

    def benchmark_results(self, trial_result):
        """
        Benchmark bounding boxes against the ground truth object labels
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        ground_truth_bboxes = trial_result.get_ground_truth_bounding_boxes()
        detected_bboxes = trial_result.get_bounding_boxes()

        class_scores = {}
        for image_id in detected_bboxes.keys():
            if image_id in ground_truth_bboxes:
                for bbox in detected_bboxes[image_id]:
                    best_score = 0
                    for gt_bbox in ground_truth_bboxes[image_id]:
                        if bbox.class_name == gt_bbox.class_name:
                            overlap_x = max((bbox.x, gt_bbox.x))
                            overlap_y = max((bbox.y, gt_bbox.y))
                            overlap_upper_x = min((bbox.x + bbox.width, gt_bbox.x + gt_bbox.width))
                            overlap_upper_y = min((bbox.y + bbox.height, gt_bbox.y + gt_bbox.height))
                            if overlap_upper_x > overlap_x and overlap_upper_y > overlap_y:
                                overlap_area = (overlap_upper_x - overlap_x) * (overlap_upper_y - overlap_y)
                                gt_area = gt_bbox.width * gt_bbox.height
                                bbox_area = bbox.width * bbox.height

                                precision = overlap_area / bbox_area
                                recall = overlap_area / gt_area
                                score = 2 * bbox.confidence * (precision * recall / (precision +  recall))
                                if score > best_score:
                                    best_score = score
                    if bbox.class_name not in class_scores:
                        class_scores[bbox.class_name] = []
                    class_scores[bbox.class_name].append(best_score)
        return bbox_result.BoundingBoxOverlapBenchmarkResult(benchmark_id=self.identifier,
                                                             trial_result_id=trial_result.identifier,
                                                             class_bbox_scores=class_scores,
                                                             settings={})
