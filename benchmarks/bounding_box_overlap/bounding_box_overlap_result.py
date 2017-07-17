import bson.objectid
import core.benchmark


class BoundingBoxOverlapBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    Simple results for a list of bounding box scores.
    Produces summaries of the distribution of scores by class,
    as well as an aggregate summary over all the classes.
    """

    def __init__(self, benchmark_id, trial_result_id, overlaps, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)
        self._overlaps = overlaps
        self._settings = settings

    @property
    def overlaps(self):
        """
        Overlaps is a map of image ids to lists of results for each bounding box detected on that image.
        i.e.: {
           bson.ObjectId(): [{
                'overlap': 57,
                'bounding_box_area': 122,
                'ground_truth_area': 100,
                'confidence': bbox.confidence,
                'bounding_box_classes': bbox.class_names,
                'ground_truth_classes': ()
            }, {
                'overlap': 0,
                'bounding_box_area': bbox.height * bbox.width,
                'ground_truth_area': 0,
                'confidence': bbox.confidence,
                'bounding_box_classes': bbox.class_names,
                'ground_truth_classes': ()
            }]
        }
        We can't really match back to the gt bounding boxes, with n ground-truth bounding boxes,
        the first n results are for those bounding boxes, followed by additional results for
        detected bounding boxes that don't match a ground truth box.
        :return:
        """
        return self._overlaps

    def scores_by_class(self):
        """
        Get the F1 scores for each class.
        An example of using map_results.
        :return:
        """
        return self.map_results(lambda x: set(x['bounding_box_classes']) & set(x['ground_truth_classes']),
                                f1_score)

    def map_results(self, key_func, value_func):
        """
        Produce a map of some key value to lists of score results.
        Takes two functions, one used to produce multiple keys per bounding box,
        the other to produce scores for that bounding box.
        Returns a map of returned key values, such as precision, recall, or f1_score

        The key example of its use is for getting scores

        :param key_func: A function to produce lists of keys for a given bounding box. Return an iterable.
        :param value_func: A function to produce values for a given bounding box. Return a score.
        :return:
        """
        results = {}
        for img_result in self.overlaps.values():
            for bbox_result in img_result:
                keys = key_func(bbox_result)
                val = value_func(bbox_result)
                for key in keys:
                    if key not in results:
                        results[key] = []
                    results[key].append(val)
        return results

    def list_results(self, *args):
        """
        Produce lists of values for each bounding box result
        For each input function, this goes through every bounding box result,
        calls the function on it, and returns a list of the results.
        The inner lists are all the same length, equal to the number of bounding boxes
        results in this object.

        For example:
        x, y = list_results(lambda x: x['bounding_box_area'], f1_score)
        produces two lists, one for the bounding box area, and one for the f1 score for each result.

        This is useful for producing sequences for plotting, such as the above,
        which allows us to plot detected box area vs f1 score

        :param args: Any number of callables, producing values that are stored in the results list.
        :return: a list of lists of results
        """
        getters = [func for func in args if callable(func)]
        results = [bbox_result for img_result in self.overlaps.values() for bbox_result in img_result]
        return [list(map(func, results)) for func in getters]

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['overlaps'] = {str(image_id): results for image_id, results in self.overlaps.items()}
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'overlaps' in serialized_representation:
            kwargs['overlaps'] = {bson.objectid.ObjectId(img_id): results
                                  for img_id, results in serialized_representation['overlaps'].items()}
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def precision(bbox_result):
    """
    Compute the precision of a bounding box result,
    based on the area of overlap of the bounding box with a ground truth bounding box.
    Useful both in itself, and as part of more complex measures (i.e. F1 score)
    :param bbox_result: A dict bounding box result, from BoundingBoxOverlapBenchmarkResult
    :return: The precision score, a float
    """
    return bbox_result['overlap'] / bbox_result['bounding_box_area'] if bbox_result['bounding_box_area'] > 0 else 0


def recall(bbox_result):
    """
    Compute the recall of a bounding box result,
    based on the area of overlap with the ground truth result.
    :param bbox_result: A dict bounding box result, from BoundingBoxOverlapBenchmarkResult
    :return: The recall, a float
    """
    return bbox_result['overlap'] / bbox_result['ground_truth_area'] if bbox_result['ground_truth_area'] > 0 else 0


def f1_score(bbox_result):
    """
    Compute the F1 score of a bounding box result.
    :param bbox_result: A dict bounding box result, from BoundingBoxOverlapBenchmarkResult
    :return: The F1 score, a float
    """
    p = precision(bbox_result)
    r = recall(bbox_result)
    return 2 * p * r / (p + r) if p > 0 and r > 0 else 0


def get_f_measure(beta=1):
    """
    Get a function that returns the f-measure of a bounding box
    with a given beta value.
    Returns the function, which is passed to map_results or list_results.
    :param beta: The type of F-measure, as a float.
    :return: A callable function that produces the F_B measure of bounding boxes
    """
    return lambda bbox: ((1 + beta * beta) * precision(bbox) * recall(bbox) /
                         (beta * beta * precision(bbox) + recall(bbox))) if beta > 0 else 0
