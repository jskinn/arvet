#Copyright (c) 2017, John Skinner
import os
import time
import numpy as np
import re
import signal
import queue
import multiprocessing
import enum
import core.system
import core.sequence_type
import core.trial_result
import trials.slam.visual_slam
import util.transform as tf
import util.dict_utils as du


# Try and use LibYAML where available, fall back to the python implementation
from yaml import dump as yaml_dump
try:
    from yaml import CDumper as YamlDumper
except ImportError:
    from yaml import Dumper as YamlDumper


class SensorMode(enum.Enum):
    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


class ORBSLAM2(core.system.VisionSystem):
    """
    Python wrapper for ORB_SLAM2
    """

    def __init__(self, vocabulary_file, settings, mode=SensorMode.RGBD, resolution=None, temp_folder='temp', id_=None):
        super().__init__(id_=id_)
        self._vocabulary_file = vocabulary_file

        self._mode = mode if isinstance(mode, SensorMode) else SensorMode.RGBD
        self._resolution = resolution if resolution is not None and len(resolution) == 2 else (640, 480)
        # Default settings based on UE4 calibration results
        self._orbslam_settings = du.defaults({}, settings, {
            'Camera': {
                # Camera calibration and distortion parameters (OpenCV)
                # Most of these get overridden with the camera intrinsics at the start of the run.
                'fx': 900,
                'fy': 900,
                'cx': 480,
                'cy': 272,

                'k1': -0.1488979,
                'k2': 1.34554205,
                'p1': 0.00487404,
                'p2': 0.0040448,
                'k3': -2.98873439,

                # Camera frames per second
                'fps': 30.0,

                # stereo baseline times fx
                'bf': 387.5744,

                # Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
                # All the images in this system will be RGB order
                'RGB': 1,
            },
            'ORBextractor': {
                # ORB Extractor: Number of features per image
                'nFeatures': 2000,

                # ORB Extractor: Scale factor between levels in the scale pyramid
                'scaleFactor': 1.2,

                # ORB Extractor: Number of levels in the scale pyramid
                'nLevels': 8,

                # ORB Extractor: Fast threshold
                # Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
                # Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
                # You can lower these values if your images have low contrast
                'iniThFAST': 20,
                'minThFAST': 7
            },
            # Viewer configuration expected by ORB_SLAM2
            'Viewer': {
                'KeyFrameSize': 0.05,
                'KeyFrameLineWidth': 1,
                'GraphLineWidth': 0.9,
                'PointSize': 2,
                'CameraSize': 0.08,
                'CameraLineWidth': 3,
                'ViewpointX': 0,
                'ViewpointY': -0.7,
                'ViewpointZ': -1.8,
                'ViewpointF': 500
            }
        })
        self._temp_folder = temp_folder

        self._expected_completion_timeout = 300     # This is how long we wait after the dataset is finished
        self._settings_file = None
        self._child_process = None
        self._input_queue = None
        self._output_queue = None
        self._gt_trajectory = None

    @property
    def mode(self):
        return self._mode

    @property
    def is_deterministic(self):
        """
        Is the visual system deterministic.

        If this is false, it will have to be tested multiple times, because the performance will be inconsistent
        between runs.

        :return: True iff the algorithm will produce the same results each time.
        :rtype: bool
        """
        return False

    def is_image_source_appropriate(self, image_source):
        """
        Is the dataset appropriate for testing this vision system.
        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        return (image_source.sequence_type == core.sequence_type.ImageSequenceType.SEQUENTIAL and (
            self._mode == SensorMode.MONOCULAR or
            (self._mode == SensorMode.STEREO and image_source.is_stereo_available) or
            (self._mode == SensorMode.RGBD and image_source.is_depth_available)))

    def set_camera_intrinsics(self, camera_intrinsics, resolution):
        """
        Set the intrinsics of the camera using
        :param camera_intrinsics: A metadata.camera_intrinsics.CameraIntriniscs object
        :return:
        """
        if self._child_process is None:
            new_fx = camera_intrinsics.fx * self._resolution[0]
            self._orbslam_settings['Camera']['bf'] = (new_fx * self._orbslam_settings['Camera']['bf']
                                                      / self._orbslam_settings['Camera']['fx'])
            self._orbslam_settings['Camera']['fx'] = new_fx
            self._orbslam_settings['Camera']['fy'] = camera_intrinsics.fy * self._resolution[1]
            self._orbslam_settings['Camera']['cx'] = camera_intrinsics.cx * self._resolution[0]
            self._orbslam_settings['Camera']['cy'] = camera_intrinsics.cy * self._resolution[1]
            self._orbslam_settings['Camera']['k1'] = camera_intrinsics.k1
            self._orbslam_settings['Camera']['k2'] = camera_intrinsics.k2
            self._orbslam_settings['Camera']['k3'] = camera_intrinsics.k3
            self._orbslam_settings['Camera']['p1'] = camera_intrinsics.p1
            self._orbslam_settings['Camera']['p2'] = camera_intrinsics.p2

    def set_stereo_baseline(self, baseline):
        """
        Set the stereo baseline configuration.
        :param baseline:
        :return:
        """
        self._orbslam_settings['Camera']['bf'] = float(baseline) * self._orbslam_settings['Camera']['fx']

    def start_trial(self, sequence_type):
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :return: void
        """
        if sequence_type is not core.sequence_type.ImageSequenceType.SEQUENTIAL:
            return

        self.save_settings()  # we have to save the settings, so that orb-slam can load them
        self._gt_trajectory = {}
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._child_process = multiprocessing.Process(target=run_orbslam,
                                                      args=(self._output_queue,
                                                            self._input_queue,
                                                            self._vocabulary_file,
                                                            self._settings_file,
                                                            self._mode,
                                                            self._resolution))
        self._child_process.start()
        try:
            started = self._output_queue.get(block=True, timeout=self._expected_completion_timeout)
        except queue.Empty:
            started = False
        if not started:
            self._child_process.terminate()
            self._child_process.join(timeout=5)
            if self._child_process.is_alive():
                os.kill(self._child_process.pid, signal.SIGKILL)  # Definitely kill the process.
            if os.path.isfile(self._settings_file):
                os.remove(self._settings_file)  # Delete the settings file
            self._settings_file = None
            self._child_process = None
            self._input_queue = None
            self._output_queue = None
            self._gt_trajectory = None
        return started

    def process_image(self, image, timestamp):
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image: The image object for this frame
        :param timestamp: A timestamp or index associated with this image. Sometimes None.
        :return: void
        """
        if self._input_queue is not None:
            # Wait here, to throttle the input rate to the queue, and prevent it from growing too large
            delay_time = 0
            while self._input_queue.qsize() > 10 and delay_time < 10:
                time.sleep(1)
                delay_time += 1

            # Add the camera pose to the ground-truth trajectory
            self._gt_trajectory[timestamp] = image.camera_pose

            # Send different input based on the running mode
            if self._mode == SensorMode.MONOCULAR:
                self._input_queue.put((np.copy(image.data), None, timestamp))
            elif self._mode == SensorMode.STEREO:
                self._input_queue.put((np.copy(image.left_data), np.copy(image.right_data), timestamp))
            elif self._mode == SensorMode.RGBD:
                self._input_queue.put((np.copy(image.data), np.copy(image.depth_data), timestamp))

    def finish_trial(self):
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        self._input_queue.put(None)     # This will end the main loop, see run_orbslam, below
        try:
            trajectory_list, tracking_stats = self._output_queue.get(block=True,
                                                                     timeout=self._expected_completion_timeout)
        except queue.Empty:
            # process has failed to complete within expected time, kill it and move on.
            trajectory_list = None
            tracking_stats = {}

        if isinstance(trajectory_list, list):
            # completed successfully, return the trajectory
            self._child_process.join()    # explicitly join

            trajectory = {}
            for timestamp, x, y, z, qx, qy, qz, qw in trajectory_list:
                # ORB_SLAM by default uses ROS coordinate frame, so we shouldn't need to convert
                trajectory[timestamp] = tf.Transform(location=(x, y, z),
                                                     rotation=(qw, qx, qy, qz), w_first=True)

            result = trials.slam.visual_slam.SLAMTrialResult(
                system_id=self.identifier,
                trajectory=trajectory,
                ground_truth_trajectory=self._gt_trajectory,
                tracking_stats=tracking_stats,
                sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL,
                system_settings=self.get_settings()
            )
        else:
            # something went wrong, kill it with fire
            self._child_process.terminate()
            self._child_process.join(timeout=5)
            if self._child_process.is_alive():
                os.kill(self._child_process.pid, signal.SIGKILL)    # Definitely kill the process.
            result = core.trial_result.FailedTrial(
                system_id=self.identifier,
                reason="Child process timed out after {0} seconds.".format(self._expected_completion_timeout),
                sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL,
                system_settings=self.get_settings()
            )

        if os.path.isfile(self._settings_file):
            os.remove(self._settings_file)  # Delete the settings file
        self._settings_file = None
        self._child_process = None
        self._input_queue = None
        self._output_queue = None
        self._gt_trajectory = None
        return result

    def get_settings(self):
        return self._orbslam_settings

    def save_settings(self):
        if self._settings_file is None:
            # Choose a new settings file
            self._settings_file = os.path.join(self._temp_folder, 'orb-slam2-settings-{0}'.format(
                self.identifier if self.identifier is not None else 'unregistered'))
            if os.path.isfile(self._settings_file):
                for idx in range(10000):
                    if not os.path.isfile(self._settings_file + '-' + str(idx)):
                        self._settings_file += '-' + str(idx)
                        break
            dump_config(self._settings_file, self._orbslam_settings)

    def serialize(self):
        serialized = super().serialize()
        serialized['vocabulary_file'] = self._vocabulary_file
        serialized['mode'] = self._mode.value
        serialized['resolution'] = self._resolution
        serialized['settings'] = self.get_settings()
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'vocabulary_file' in serialized_representation:
            kwargs['vocabulary_file'] = serialized_representation['vocabulary_file']
        if 'mode' in serialized_representation:
            kwargs['mode'] = SensorMode(serialized_representation['mode'])
        if 'resolution' in serialized_representation:
            kwargs['resolution'] = serialized_representation['resolution']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        kwargs['temp_folder'] = db_client.temp_folder
        return super().deserialize(serialized_representation, db_client, **kwargs)


def load_config(filename):
    """
    Load an opencv yaml FileStorage file, accounting for a couple of inconsistencies in syntax.
    :param filename: The file to load from
    :return: A python object constructed from the config, or an empty dict if not found
    """
    config = {}
    with open(filename, 'r') as config_file:
        re_comment_split = re.compile('[%#]')
        for line in config_file:
            line = re_comment_split.split(line, 1)[0]
            if len(line) <= 0:
                continue
            else:
                key, value = line.split(':', 1)
                key = key.strip('"\' \t')
                value = value.strip()
                value_lower = value.lower()
                if value_lower == 'true':
                    actual_value = True
                elif value_lower == 'false':
                    actual_value = False
                else:
                    try:
                        actual_value = float(value)
                    except ValueError:
                        actual_value = value
                config[key] = actual_value
    return config


def dump_config(filename, data, dumper=YamlDumper, default_flow_style=False, **kwargs):
    """
    Dump the ORB_SLAM config to file,
    There's some fiddling with the format here so that OpenCV will read it on the other end.
    :param filename:
    :param data:
    :param dumper:
    :param default_flow_style:
    :param kwargs:
    :return:
    """
    with open(filename, 'w') as config_file:
        config_file.write("%YAML:1.0\n")
        return yaml_dump(nested_to_dotted(data), config_file, Dumper=dumper,
                         default_flow_style=default_flow_style, **kwargs)


def nested_to_dotted(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for inner_key, value in nested_to_dotted(value).items():
                result[key + '.' + inner_key] = value
        else:
            result[key] = value
    return result


def run_orbslam(output_queue, input_queue, vocab_file, settings_file, mode, resolution):
    import orbslam2
    import logging
    import trials.slam.tracking_state

    logging.getLogger(__name__).info("Starting ORBSLAM2...")
    sensor_mode = orbslam2.Sensor.RGBD
    if mode == SensorMode.MONOCULAR:
        sensor_mode = orbslam2.Sensor.MONOCULAR
    elif mode == SensorMode.STEREO:
        sensor_mode = orbslam2.Sensor.STEREO

    tracking_stats = {}
    orbslam_system = orbslam2.System(vocab_file, settings_file, resolution[0], resolution[1], sensor_mode)
    orbslam_system.set_use_viewer(False)
    orbslam_system.initialize()
    output_queue.put(True)  # Tell the parent process we've set-up correctly and are ready to go.
    logging.getLogger(__name__).info("ORBSLAM2 Ready.")

    running = True
    while running:
        in_data = input_queue.get(block=True)
        if isinstance(in_data, tuple) and len(in_data) == 3:
            img1, img2, timestamp = in_data
            if mode == SensorMode.MONOCULAR:
                orbslam_system.process_image_mono(img1, timestamp)
            elif mode == SensorMode.STEREO:
                orbslam_system.process_image_stereo(img1, img2, timestamp)
            elif mode == SensorMode.RGBD:
                orbslam_system.process_image_rgbd(img1, img2, timestamp)

            tracking_state = orbslam_system.get_tracking_state()
            if (tracking_state == orbslam2.TrackingState.SYSTEM_NOT_READY or
                    tracking_state == orbslam2.TrackingState.NO_IMAGES_YET or
                    tracking_state == orbslam2.TrackingState.NOT_INITIALIZED):
                tracking_stats[timestamp] = trials.slam.tracking_state.TrackingState.NOT_INITIALIZED
            elif tracking_state == orbslam2.TrackingState.OK:
                tracking_stats[timestamp] = trials.slam.tracking_state.TrackingState.OK
            else:
                tracking_stats[timestamp] = trials.slam.tracking_state.TrackingState.LOST
        else:
            # Non-matching input indicates the end of processing, stop the main loop
            logging.getLogger(__name__).info("Got terminate input, finishing up and sending results.")
            running = False

    # send the final trajectory to the parent
    output_queue.put((orbslam_system.get_trajectory_points(), tracking_stats))

    # shut down the system. This is going to crash it, but that's ok, because it's a subprocess
    orbslam_system.shutdown()
    logging.getLogger(__name__).info("Finished running ORBSLAM2")
