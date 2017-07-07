import abc
import operator

import cv2

import core.system
import core.sequence_type
import trials.feature_detection.feature_detector_result as detector_result


class FeatureDetector(core.system.VisionSystem, metaclass=abc.ABCMeta):
    """
    A Feature Detector as a vision system, as a generalization of the OpenCV standard detector interface.
    Has subclasses that hold configuration and create individual detectors.
    """

    def __init__(self, id_=None):
        super().__init__(id_=id_)

        # variables for trial state
        self._detector = None
        self._key_points = None
        self._timestamps = None

    def is_trial_running(self):
        """
        Has the trial started.
        Don't change the config when the trial is running
        :return:
        """
        return self._key_points is not None

    @abc.abstractmethod
    def make_detector(self):
        """
        Make the cv2 detector for this system.
        Call the constructor
        :return:
        """
        pass

    @abc.abstractmethod
    def get_system_settings(self):
        """
        Get the settings values used by the detector
        :return:
        """
        pass

    @property
    def is_deterministic(self):
        """
        Is the visual system deterministic.
        :return: True, features are deterministic
        :rtype: bool
        """
        return True

    def is_image_source_appropriate(self, image_source):
        """
        Is the dataset appropriate for testing this vision system.
        True for sources where the image is stored in the database,
         we need to be able to associate feature points with particular images
        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        return image_source.is_stored_in_database()

    def start_trial(self, sequence_type):
        """
        Start a trial for a feature detector
        :param sequence_type: The type of image sequence that will be provided for this trial
        :return: void
        """
        self._sequence_type = core.sequence_type.ImageSequenceType(sequence_type)
        self._key_points = {}
        self._timestamps = {}
        self._detector = self.make_detector()

    def process_image(self, image, timestamp):
        """
        Process an image as part of the current run.
        :param image: An Image object
        :param timestamp: The timestamp or index associated with this image in the image source. Irrelevant for this.
        :return: void
        """
        if not self.is_trial_running():
            self.start_trial(core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
        if hasattr(image, 'identifier'):
            grey_image = cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY)
            key_points = self._detector.detect(grey_image, None)    # No mask, detect over the whole image
            key_points.sort(key=operator.attrgetter('response'))
            self._key_points[image.identifier] = key_points
            self._timestamps[timestamp] = image.identifier

    def finish_trial(self):
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        result = None
        if self._key_points is not None and len(self._key_points) > 0:
            result = detector_result.FeatureDetectorResult(system_id=self.identifier,
                                                           keypoints=self._key_points,
                                                           timestamps=self._timestamps,
                                                           sequence_type=self._sequence_type,
                                                           system_settings=self.get_system_settings())
        self._key_points = None
        self._timestamps = None
        self._detector = None
        return result
