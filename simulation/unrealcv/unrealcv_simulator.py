import util.dict_utils as du
import util.transform as tf
import simulation.simulator
import core.image as im
import unrealcv


class UnrealCVSimulator(simulation.simulator.Simulator):

    def __init__(self, controller=None, config=None):
        super().__init__(controller)

        if config is None:
            config = {}
        config = du.defaults(config, {
            'framerate': 30,    # fps
            'stereo_offset': 0, # unreal units
            'provide_rgb': True,
            'provide_depth': False,
            'provide_labels': False,
            'provide_world_normals': False,

            # Run settings
            'max_frames': 1000000,  # maximum number of frames, set to <= 0 for infinite

            # Simulation server config
            'host': 'localhost',
            'port': 9000
        })

        self._framerate = float(config['framerate'])

        self._stereo_offset = float(config['stereo_offset'])
        self._provide_depth = bool(config['provide_depth'])
        self._provide_labels = bool(config['provide_labels'])
        self._provide_world_normals = bool(config['provide_world_normals'])

        self._max_frames = int(config['max_frames'])

        self._host = str(config['host'])
        self._port = int(config['port'])

        self._client = None
        self._current_pose = None
        self._timestamp = 0
        self._frame_count = 0

    @property
    def is_depth_available(self):
        """
        Can this simulator produce depth images.
        :return: True if the simulator is configured to take depth images
        """
        return self._provide_depth

    @property
    def is_labels_available(self):
        """
        Can this simulator produce object labels.
        :return: True if the simulator is configured to produce labels for each frame
        """
        return self._provide_labels

    @property
    def is_normals_available(self):
        """
        Can this simulator produce object labels.
        :return: True if the simulator is configured to produce labels for each frame
        """
        return self._provide_world_normals

    @property
    def is_stereo_available(self):
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images
        :return:
        """
        return self._stereo_offset > 0

    @property
    def sequence_type(self):
        """
        Get the type of image sequence produced by this image source.
        For instance, the source may produce sequential images, or disjoint, random images.
        This may change with the configuration of the image source.
        It is useful for determining which sources can run with which algorithms.
        :return: The image sequence type enum
        :rtype core.image_sequence.ImageSequenceType:
        """
        return self.controller.motion_type

    @property
    def current_pose(self):
        """
        Get the current location of the camera
        :return: The current camera pose
        """
        return self._current_pose

    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff the simulator has started correctly.
        """
        # TODO: Launch external process running the simulator



        self.controller.reset()
        self._frame_count = 0
        self._timestamp = 0

        if self._client is None:
            self._client = unrealcv.Client((self._host, self._port))
        if self._client.isconnected():
            return True
        return self._client.connect()

    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images

        :return: An Image object (see core.image) or None
        """
        if self._client is not None:
            # Send the camera pose to the simulator
            pose = self.controller.get_next_pose()
            self.set_camera_pose(pose)

            # move forward in time
            self._timestamp += 1/self._framerate
            self._frame_count += 1

            # Get and return the image from the simulator
            return self._get_image()
        return None

    def is_complete(self):
        """
        Have we got all the images from this source?
        Some sources are infinite, some are not,
        and this method lets those that are not end the iteration.
        :return: True
        """
        return self._max_frames > 0 and self._frame_count >= self._max_frames

    def set_camera_pose(self, pose):
        """
        Set the camera pose in the simulator
        :param pose:
        :return:
        """
        if self._client is not None:
            self._current_pose = pose
            unreal_pose = self.transform_to_unreal(pose)
            euler_rotation = unreal_pose.euler
            self._client.request("vset /camera/0/location {0} {1} {2}".format(unreal_pose.location[0],
                                                                              unreal_pose.location[1],
                                                                              unreal_pose.location[2]))
            self._client.request("vset /camera/0/rotation {0} {1} {2}".format(euler_rotation[0],
                                                                              euler_rotation[1],
                                                                              euler_rotation[2]))

    def _get_additional_metadata(self):
        # TODO: Add properties like realism settings and camera FOV as we can control them
        return {

        }

    def _get_image(self):
        if self._client is not None:
            image_filename = self._client.request('vget /camera/0/lit')
            depth_filename = None
            labels_filename = None
            world_normals_filename = None

            if self.is_depth_available:
                depth_filename = self._client.request('vget /camera/0/depth')
            if self.is_labels_available:
                labels_filename = self._client.request('vget /camera/0/object_mask')
            if self.is_normals_available:
                world_normals_filename = self._client.request('vget /camera/0/normal')

            if self.is_stereo_available:
                cached_pose = self.current_pose
                right_pose = self.current_pose.find_independent((0, 0, self._stereo_offset))
                self.set_camera_pose(right_pose)

                right_image_filename = self._client.request('vget /camera/0/lit')
                right_depth_filename = None
                right_labels_filename = None
                right_world_normals_filename = None

                if self.is_depth_available:
                    right_depth_filename = self._client.request('vget /camera/0/depth')
                if self.is_labels_available:
                    right_labels_filename = self._client.request('vget /camera/0/object_mask')
                if self.is_normals_available:
                    right_world_normals_filename = self._client.request('vget /camera/0/normal')

                return im.StereoImage(timestamp=self._timestamp,
                                      left_filename=image_filename,
                                      right_filename=right_image_filename,
                                      left_camera_pose=cached_pose,
                                      right_camera_pose=right_pose,
                                      additional_metadata=self._get_additional_metadata(),
                                      left_depth_filename=depth_filename,
                                      left_labels_filename=labels_filename,
                                      left_world_normals_filename=world_normals_filename,
                                      right_depth_filename=right_depth_filename,
                                      right_labels_filename=right_labels_filename,
                                      right_world_normals_filename=right_world_normals_filename)
            return im.Image(timestamp=self._timestamp,
                            filename=image_filename,
                            camera_pose=self.current_pose,
                            additional_metadata=self._get_additional_metadata(),
                            depth_filename=depth_filename,
                            labels_filename=labels_filename,
                            world_normals_filename=world_normals_filename)
        return None

    @staticmethod
    def transform_to_unreal(pose):
        """
        Swap the coordinate frames from my standard coordinate frame
        to the one used by unreal
        :param pose: A point, as any 3-length indexable, or a Transform object
        :return: A tuple or Transform representing the same point in unreal coordinates
        """
        if isinstance(pose, tf.Transform):
            location = pose.location
            location = (location[0], location[2], location[1])
            rotation = pose.rotation_quat(w_first=True)
            rotation = (-rotation[0], rotation[0], rotation[2], rotation[1])
            return tf.Transform(location=location, rotation=rotation)
        return pose[0], pose[2], pose[1]

    @staticmethod
    def transform_from_unreal(pose):
        """
        Swap the coordinate frames from unreal coordinates
        to my standard convention
        :param pose: A point, as any 3-indexable, or a Transform object
        :return: A point or Transform object, depending on the parameter
        """
        if isinstance(pose, tf.Transform):
            location = pose.location
            location = (location[0], location[2], location[1])
            rotation = pose.rotation_quat(w_first=True)
            rotation = (-rotation[0], rotation[0], rotation[2], rotation[1])
            return tf.Transform(location=location, rotation=rotation)
        return pose[0], pose[2], pose[1]
