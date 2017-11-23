# Copyright (c) 2017, John Skinner
import operator
import database.entity


class ContinuousParameter(database.entity.Entity):

    def __init__(self, name, min_ = None, max_ = None, id_ = None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        self._name = name
        self._min = min_
        self._max = max_

    @property
    def name(self):
        return self._name

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def sample(self, existing_results, num_samples):
        """
        Choose some new sample points for this parameter,
        based on the existing sample points.
        Continuous parameters can be sampled indefinitely, so this should always return some number of values.
        
        The plan is as follows:
        - If we have a min or max, sample there as an upper bound on our range.
        - Otherwise, double the search space in each unbounded direction (new sample is 2 * previous max/min)
        - For sub-sampling, linearly estimate the gradient between subsequent pairs of existing samples.
          candidate sample is midpoint between each pairs. Choose samples with highest variation in gradient
         
        TODO: Maybe a gradient pyramid is a better option, calculate gradients over widening windows around each candidate point
        TODO: numpy.polyfit can estimate gradients over wider windows 
        TODO: Use an information measure as the sample score
        TODO: Rather than/as well as using pairwise gradients, estimate gradient at each sample point
        
        :param existing_results: A map of existing sample values to a numeric score which we're sampling against.
        :param num_samples: The upper limit on the desired samples/ 
        :return: 
        """
        new_samples = set()
        existing_samples = list(existing_results.keys())
        existing_samples.sort()

        if self.max is not None and self.max not in existing_results and len(new_samples) < num_samples:
            new_samples.add(self.max)
        elif self.max is None and len(new_samples) < num_samples:
            new_samples.add(max(existing_samples) * 2)

        if self.min is not None and self.min not in existing_results and len(new_samples) < num_samples:
            new_samples.add(self.min)
        elif self.min is None and len(new_samples) < num_samples:
            new_samples.add(min(existing_samples) * 2)

        if (self.max is not None and self.min is not None and (self.max - self.min) / 2 not in existing_results and
                    len(new_samples) < num_samples):
            new_samples.add(0.5 * (self.max - self.min))

        if len(existing_results) > 2 and len(new_samples) < num_samples:
            gradients = [(existing_results[existing_samples[i]] - existing_results[existing_samples[i-1]]) /
                         (existing_samples[i] - existing_samples[i-1]) for i in range(1, len(existing_samples))]

            candidate_samples = []
            for i in range(1, len(existing_samples)):
                candidate_sample = 0.5 * (existing_samples[i] - existing_samples[i-1])
                gradient = gradients[i-1]
                if i > 2:
                    score +=

            # Sort the candidate samples by score
            candidate_samples.sort(key=operator.itemgetter(1), reverse=True)
            for i in range(0, min(len(candidate_samples), ))

        return new_samples


class IntegerParameter(database.entity.Entity):

    def __init__(self, name, min_ = None, max_ = None, id_ = None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        self._name = name
        self._min = min_
        self._max = max_


class DiscreteEnumParameter(database.entity.Entity):
    """
    Parameters of this type are discretized examples of unmeasurable quantities
    like light level or time of day.
    """
    pass


class IndependentEnum(database.entity.Entity):
    pass

