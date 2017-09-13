#Copyright (c) 2017, John Skinner
import numpy as np
import cv2  # TODO: Replace with a lighter library, maybe pillow
import viso2 as libviso2

import core.sequence_type
import core.system
import core.trial_result
import trials.visual_odometry.visual_odometry_result as vo_result
import util.transform as tf


class LibVisOSystem(core.system.VisionSystem):
    """
    Class to run LibVisO2 as a vision system.
    """

    def __init__(self, focal_distance=1, cu=640, cv=360, base=0.3, id_=None):
        super().__init__(id_=id_)
        self._camera_settings = None
        self._focal_distance = float(focal_distance)
        self._cu = float(cu)
        self._cv = float(cv)
        self._base = float(base)
        self._viso = None
        self._frame_deltas = None
        self._gt_poses = None

    def is_image_source_appropriate(self, image_source):
        return (image_source.sequence_type == core.sequence_type.ImageSequenceType.SEQUENTIAL and
                image_source.is_stereo_available)

    def is_deterministic(self):
        return True

    def set_camera_intrinsics(self, camera_intrinsics, resolution):
        """
        Set the camera intrinisics for libviso2
        :param camera_intrinsics: The camera intrinsics, relative to the image resolution
        :param resolution: The image resolution
        :return:
        """
        self._focal_distance = float(camera_intrinsics.fx * resolution[0])
        self._cu = float(camera_intrinsics.cx * resolution[0])
        self._cv = float(camera_intrinsics.cy * resolution[1])

    def set_stereo_baseline(self, baseline):
        """
        Set the stereo baseline
        :param baseline:
        :return:
        """
        self._base = float(baseline)

    def start_trial(self, sequence_type):
        if not sequence_type == core.sequence_type.ImageSequenceType.SEQUENTIAL:
            return False
        params = libviso2.Stereo_parameters()
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        params.base = self._base

        self._viso = libviso2.VisualOdometryStereo(params)
        self._frame_deltas = {}
        self._gt_poses = {}

    def process_image(self, image, timestamp):
        left_grey = prepare_image(image.left_data)
        right_grey = prepare_image(image.right_data)
        self._viso.process_frame(left_grey, right_grey)
        motion = self._viso.getMotion()  # Motion is a 4x4 pose matrix
        np_motion = np.zeros((4, 4))
        motion.toNumpy(np_motion)
        self._frame_deltas[timestamp] = make_relative_pose(np_motion)
        self._gt_poses[timestamp] = image.camera_pose
        # TODO: Aggregate the image metadata

    def finish_trial(self):
        result = vo_result.VisualOdometryResult(
            system_id=self.identifier,
            sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL,
            system_settings={
                'f': self._focal_distance,
                'cu': self._cu,
                'cv': self._cv,
                'base': self._base
            },
            frame_deltas=self._frame_deltas,
            ground_truth_trajectory=self._gt_poses)
        self._frame_deltas = None
        self._gt_poses = None
        self._viso = None
        return result


def make_relative_pose(frame_delta):
    """
    LibVisO2 uses a different coordinate frame to the one I'm using,
    this function is to convert computed frame deltas as libviso estimates them to usable poses.
    Thankfully, its still a right-handed coordinate frame, which makes this easier.
    Frame is: z forward, x right, y down
    Documentation at: http://www.cvlibs.net/software/libviso/

    :param frame_delta: A 4x4 matrix (possibly in list form)
    :return: A Transform object representing the pose of the current frame with respect to the previous frame
    """
    frame_delta = np.asmatrix(frame_delta)
    coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
    pose = np.dot(np.dot(coordinate_exchange, frame_delta), coordinate_exchange.T)
    return tf.Transform(pose)


def prepare_image(image_data):
    """
    Process image data ready to input it to libviso2.
    We expect input images to be greyscale uint8 images, other image types are massaged into that shape
    :param image_data: A numpy array of image data
    :return: An uint8 numpy array, intensity range 0-255
    """
    output_image = image_data
    if len(image_data.shape) > 2:
        output_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    if output_image.dtype != np.uint8:
        if 0.99 < np.max(output_image) <= 1.001:
            output_image = 255 * output_image
        output_image = np.asarray(output_image, dtype=np.uint8)
    return output_image
