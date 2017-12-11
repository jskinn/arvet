# Copyright (c) 2017, John Skinner
import os
import time
import subprocess
import xxhash
import numpy as np
import unrealcv

import arvet.core.image as im
import arvet.database.entity
import arvet.config.path_manager
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.simulation.simulator
import arvet.simulation.depth_noise
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.util.unreal_transform as uetf
import arvet.util.image_utils as image_utils


TEMPLATE_UNREALCV_SETTINGS = """
[UnrealCV.Core]
Port={port}
Width={width}
Height={height}

"""

# Make sure we only EVER have 1 simulator process, because otherwise we'll confuse our connections.
_simulator_process = None


def start_simulator(path):
    """
    Internal use only. Start up an UnrealCV simulator at the given location.
    Stops any existing simulator.
    :param path:
    :return:
    """
    global _simulator_process
    if _simulator_process is not None:
        stop_simulator()
    _simulator_process = subprocess.Popen(path, stdout=subprocess.DEVNULL)


def stop_simulator():
    """
    Internal use only. Stop the UnrealCV simulator.
    :return:
    """
    global _simulator_process
    if _simulator_process is not None:
        _simulator_process.terminate()
        try:
            _simulator_process.wait(100)
        except subprocess.TimeoutExpired:
            try:
                _simulator_process.kill()
            except OSError:
                pass
        _simulator_process = None


class UnrealCVSimulator(arvet.simulation.simulator.Simulator, arvet.database.entity.Entity):

    def __init__(self, executable_path, world_name, environment_type=imeta.EnvironmentType.INDOOR,
                 light_level=imeta.LightingLevel.EVENLY_LIT, time_of_day=imeta.TimeOfDay.DAY, config=None, id_=None):
        """
        Create an unrealcv simulator with a bunch of configuration.
        :param executable_path: The path to the simulator executable, which will be started with this image source
        :param config: Simulator configuration, this can change dynamically, see generate_dataset_task
        """
        super().__init__(id_=id_)
        if config is None:
            config = {}
        du.defaults(config, {
            # Simulation execution config
            'stereo_offset': 0,  # meters
            'provide_rgb': True,
            'provide_depth': False,
            'provide_labels': False,
            'provide_world_normals': False,

            # Simulator settings
            'origin': {},
            'resolution': {'width': 1280, 'height': 720},
            'fov': 90,
            'depth_of_field_enabled': True,
            'focus_distance': None,     # None indicates autofocus
            'aperture': 2.2,
            #'exposure': None,           # None indicates histogram auto-exposure. Not implemented yet.

            # Quality settings
            'lit_mode': True,
            'texture_mipmap_bias': 0,
            'normal_maps_enabled': True,
            'roughness_enabled': True,
            'geometry_decimation': 0,

            # Simulation server config
            'host': 'localhost',
            'port': 9000,

            # Additional metadata that will be added to each image
            'metadata': {}
        })
        # Constant settings, these are saved to the database
        self._executable = str(executable_path)
        self._world_name = str(world_name)
        self._environment_type = imeta.EnvironmentType(environment_type)
        self._light_level = imeta.LightingLevel(light_level)
        self._time_of_day = imeta.TimeOfDay(time_of_day)

        # Configuration loaded at run time, and specified in GenerateDatasetTask
        self._stereo_offset = float(config['stereo_offset'])
        self._provide_depth = bool(config['provide_depth'])
        self._provide_labels = bool(config['provide_labels'])
        self._provide_world_normals = bool(config['provide_world_normals'])

        self._origin = uetf.deserialize(config['origin'])
        self._resolution = (config['resolution']['width'], config['resolution']['height'])
        self._fov = float(config['fov'])
        self._use_dof = bool(config['depth_of_field_enabled'])
        self._focus_distance = float(config['focus_distance']) if config['focus_distance'] is not None else None
        self._aperture = float(config['aperture'])
        #self._exposure = float(config['exposure']) if config['exposure'] is not None else None

        self._lit_mode = bool(config['lit_mode'])
        self._texture_mipmap_bias = int(config['texture_mipmap_bias'])
        self._normal_maps_enabled = bool(config['normal_maps_enabled'])
        self._roughness_enabled = bool(config['roughness_enabled'])
        self._geometry_decimation = int(config['geometry_decimation'])

        self._host = str(config['host'])
        self._port = int(config['port'])

        # Additional metatada included in the config
        self._additional_metadata = dict(config['metadata'])

        self._actual_executable = None
        self._client = None
        self._current_pose = None

    @property
    def is_depth_available(self):
        """
        Can this simulator produce depth images.
        :return: True if the simulator is configured to take depth images
        """
        return self._provide_depth

    @property
    def is_per_pixel_labels_available(self):
        """
        Can this simulator produce object labels.
        :return: True if the simulator is configured to produce labels for each frame
        """
        return self._provide_labels

    @property
    def is_labels_available(self):
        """
        Can this simulator produce object bounding boxes.
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
    def is_stored_in_database(self):
        """
        Simulators do not produce image stored in the database,
        which is why this method exists.
        :return:
        """
        return False

    def resolve_paths(self, path_manager: arvet.config.path_manager.PathManager):
        """
        Find the actual executable to launch this simulator.
        :param path_manager:
        :return:
        """
        self._actual_executable = path_manager.find_file(self._executable)

    def get_camera_intrinsics(self):
        """
        Get the current camera intrinsics from the simulator, based on its fov and aspect ratio
        :return:
        """
        rad_fov = np.pi * self.field_of_view / 180
        focal_length = 1 / (2 * np.tan(rad_fov / 2))

        # In unreal 4, field of view is whichever is the larger dimension
        # See: https://answers.unrealengine.com/questions/36550/perspective-camera-and-field-of-view.html
        if self._resolution[0] > self._resolution[1]:  # Wider than tall, fov is horizontal FOV
            focal_length = focal_length * self._resolution[0]
        else:  # Taller than wide, fov is vertical fov
            focal_length = focal_length * self._resolution[1]
        return cam_intr.CameraIntrinsics(
            width=self._resolution[0],
            height=self._resolution[1],
            fx=focal_length, fy=focal_length,
            cx=0.5 * self._resolution[0], cy=0.5 * self._resolution[1]
        )

    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff the simulator has started correctly.
        """
        # Start the simulator process
        if self._host == 'localhost':
            if self._client is not None:
                # Stop the client to free up the port
                self._client.disconnect()
                self._client = None
            self._store_config()
            start_simulator(self._actual_executable)
            # Wait for the UnrealCV server to start, pulling lines from stdout to check
            time.sleep(2)   # Wait, we can't capture some of the output right now
#            while self._simulator_process.poll() is not None:
#                line = self._simulator_process.stdout.readline()
#                if 'listening on port {0}'.format(self._port) in line.lower():
#                    # Server has started, break
#                    break

        # Create and connect the client
        if self._client is None:
            self._client = unrealcv.Client((self._host, self._port))
        if not self._client.isconnected():
            # Try and connect to the server 3 times, waiting 2s between tries
            for _ in range(10):
                self._client.connect()
                if self._client.isconnected():
                    break
                else:
                    time.sleep(2)
            if not self._client.isconnected():
                # Cannot connect to the server, shutdown completely.
                self.shutdown()

        # Set camera state from configuration
        if self._client is not None and self._client.isconnected():
            # Set camera properties
            self.set_field_of_view(self._fov)
            self.set_fstop(self._aperture)
            self.set_enable_dof(self._use_dof)
            if self._use_dof:   # Don't bother setting dof properties if dof is disabled
                if self._focus_distance is None:
                    self.set_autofocus(True)
                else:
                    self.set_focus_distance(self._focus_distance)

            # Set quality settings
            self._client.request("vset /quality/texture-mipmap-bias {0}".format(self._texture_mipmap_bias))
            self._client.request("vset /quality/normal-maps-enabled {0}".format(int(self._normal_maps_enabled)))
            self._client.request("vset /quality/roughness-enabled {0}".format(int(self._roughness_enabled)))
            self._client.request("vset /quality/geometry-decimation {0}".format(self._geometry_decimation))

            # Get the initial camera pose, which will be set by playerstart in the level
            location = self._client.request("vget /camera/0/location")
            location = location.split(' ')
            if len(location) == 3:
                rotation = self._client.request("vget /camera/0/rotation")
                rotation = rotation.split(' ')
                if len(rotation) == 3:
                    ue_trans = uetf.UnrealTransform(
                        location=(float(location[0]), float(location[1]), float(location[2])),
                        # Reorder to roll, pitch, yaw, Unrealcv order is pitch, yaw, roll
                        rotation=(float(rotation[2]), float(rotation[0]), float(location[1])))
                    self._current_pose = uetf.transform_from_unreal(ue_trans)
        return self._client is not None and self._client.isconnected()

    def get_next_image(self):
        """
        Get the next image from this source.
        It assumes that the simulation state has been changed externally
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images

        :return: An Image object (see arvet.core.image) or None, and None for the timestamp
        """
        if self._client is not None:
            # Get and return the image from the simulator
            return self._get_image(), None
        return None, None

    def is_complete(self):
        """
        Have we got all the images from this source?
        Simulators never run out of images.
        :return: False
        """
        return False

    def shutdown(self):
        """
        Shut down the simulator.
        At the moment, this is less relevant for other image source types.
        If it becomes more common, move it into image_source
        :return:
        """
        # Disconnect the client first
        if self._client is not None:
            self._client.disconnect()
            self._client = None
        # Stop the subprocess if it is running
        stop_simulator()
        self._current_pose = None

    @property
    def current_pose(self):
        """
        Get the current location of the camera
        :return: The current camera pose
        """
        return self._current_pose

    def set_camera_pose(self, pose):
        """
        Set the camera pose in the simulator
        :param pose:
        :return:
        """
        if self._client is not None:
            if isinstance(pose, uetf.UnrealTransform):
                self._current_pose = uetf.transform_from_unreal(pose)
            else:
                self._current_pose = pose
                pose = uetf.transform_to_unreal(pose)
            pose = self._origin.find_independent(pose)  # Find world coordinates from relative to simulator origin
            self._client.request("vset /camera/0/location {0} {1} {2}".format(pose.location[0],
                                                                              pose.location[1],
                                                                              pose.location[2]))
            self._client.request("vset /camera/0/rotation {0} {1} {2}".format(pose.pitch,
                                                                              pose.yaw,
                                                                              pose.roll))

    def move_camera_to(self, pose):
        """
        Move the camera to a given pose, colliding with objects and stopping if blocked.
        :param pose:
        :return:
        """
        if self._client is not None:
            if isinstance(pose, uetf.UnrealTransform):
                self._current_pose = uetf.transform_from_unreal(pose)
            else:
                self._current_pose = pose
                pose = uetf.transform_to_unreal(pose)
            pose = self._origin.find_independent(pose)
            self._client.request("vset /camera/0/moveto {0} {1} {2}".format(pose.location[0],
                                                                            pose.location[1],
                                                                            pose.location[2]))
            self._client.request("vset /camera/0/rotation {0} {1} {2}".format(pose.pitch,
                                                                              pose.yaw,
                                                                              pose.roll))
            # Re-extract the location from the sim, because we might not have made it to the desired location
            location = self._client.request("vget /camera/0/location")
            location = location.split(' ')
            if len(location) == 3:
                pose = uetf.UnrealTransform(
                    location=(float(location[0]), float(location[1]), float(location[2])),
                    rotation=pose.euler)
            self._current_pose = uetf.transform_from_unreal(pose)

    def get_obstacle_avoidance_force(self, radius=1, velocity=(1, 0, 0)):
        """
        Get a force for obstacle avoidance.
        The simulator should get all objects within the given radius,
        and provide a net force away from all the objects, scaled by the distance to the objects.

        :param radius: Distance to detect objects, in meters
        :param velocity: The current velocity of the camera, for predicting collisions.
        :return: A repulsive force vector, as a numpy array
        """
        if self._client is not None:
            velocity = uetf.transform_to_unreal(velocity)
            force = self._client.request("vget /camera/0/avoid {0} {1} {2} {3}".format(
                radius * 100, velocity[0], velocity[1], velocity[2]))
            force = force.split(' ')
            if len(force) == 3:
                force = np.array([float(force[0]), float(force[1]), float(force[2])])
                return uetf.transform_from_unreal(force)
        return np.array([0, 0, 0])

    @property
    def field_of_view(self):
        return self._fov

    def set_field_of_view(self, fov):
        """
        Set the field of view of the simulator
        :param fov:
        :return:
        """
        self._fov = float(fov)
        if self._client is not None:
            self._client.request("vset /camera/0/fov {0}".format(self._fov))

    def set_enable_dof(self, enable=True):
        """
        Set whether this simulator uses depth of field
        If true, we will mimic lens behaviour by blurring at different depths.
        Otherwise, this camera is a perfect pinhole camera, and is in-focus at all depths.
        :param enable: Whether depth of field is enabled. Default True.
        :return: void
        """
        self._use_dof = bool(enable)
        if self._client is not None:
            if self._use_dof:
                self._client.request("vset /camera/0/enable-dof 1")
            else:
                self._client.request("vset /camera/0/enable-dof 0")

    @property
    def focus_distance(self):
        """
        Get the focus distance, that is, the distance from the camera at which the world is in focus, in cm.
        :return:
        """
        if self._focus_distance is not None and self._focus_distance >= 0:
            return self._focus_distance
        elif self._client is not None:
            self._focus_distance = float(self._client.request("vget /camera/0/focus-distance")) / 100
            return self._focus_distance
        return None

    def set_focus_distance(self, focus_distance):
        """
        Set the focus distance of the simulator.
        This overrides the autofocus, you need to call set_autofocus again to focus automatically
        :param focus_distance:
        :return:
        """
        self._focus_distance = float(focus_distance)
        if self._client is not None:
            self._client.request("vset /camera/0/autofocus 0")
            self._client.request("vset /camera/0/focus-distance {0}".format(float(focus_distance) * 100))

    def set_autofocus(self, autofocus):
        """
        Set to the simulator to automatically choose the focus distance
        :param autofocus:
        :return:
        """
        if self._client is not None:
            self._client.request("vset /camera/0/autofocus {0}".format(int(bool(autofocus))))
            if bool(autofocus):
                self._focus_distance = -1

    @property
    def fstop(self):
        """
        Get the aperture or FStop settings from the simulator.
        :return:
        """
        if self._aperture is not None and self._aperture > 0:
            return self._aperture
        elif self._aperture is not None:
            return self._client.request("vget /camera/0/fstop")
        return None

    def set_fstop(self, fstop):
        """
        Set the simulated fstop of the camera aperture
        :param fstop:
        :return:
        """
        if self._client is not None:
            self._aperture = float(fstop)
            self._client.request("vset /camera/0/fstop {0}".format(float(fstop)))

    def get_object_pose(self, object_name):
        """
        Get the world pose of an object within the simulation.
        :param object_name: The name of the object
        :return:
        """
        if self._client is not None:
            location = self._client.request("vget /object/{0}/location".format(object_name))
            location = location.split(' ')
            if len(location) == 3:
                rotation = self._client.request("vget /object/{0}/rotation".format(object_name))
                rotation = rotation.split(' ')
                if len(rotation) == 3:
                    ue_trans = uetf.UnrealTransform(
                        location=(float(location[0]), float(location[1]), float(location[2])),
                        # Reorder to roll, pitch, yaw, Unrealcv order is pitch, yaw, roll
                        rotation=(float(rotation[2]), float(rotation[0]), float(location[1])))
                    ue_trans = self._origin.find_relative(ue_trans)  # Express relative to the simulator origin
                    return uetf.transform_from_unreal(ue_trans)
        return None

    def num_visible_objects(self):
        """
        Get the number of visible labelled objects.
        This is useful when generating object detection datasets, so we can skip frames where we don't see anything
        important.
        :return:
        """
        if self._client is not None:
            result = self._client.request('vget /object/num_visible')
            return int(result)
        return 0

    def validate(self):
        valid = super().validate()
        return valid

    def serialize(self):
        serialized = super().serialize()
        serialized['executable'] = self._executable
        serialized['world_name'] = self._world_name
        serialized['environment_type'] = self._environment_type.value
        serialized['light_level'] = self._light_level.value
        serialized['time_of_day'] = self._time_of_day.value
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'executable' in serialized_representation:
            kwargs['executable_path'] = serialized_representation['executable']
        if 'world_name' in serialized_representation:
            kwargs['world_name'] = serialized_representation['world_name']
        if 'environment_type' in serialized_representation:
            kwargs['environment_type'] = serialized_representation['environment_type']
        if 'light_level' in serialized_representation:
            kwargs['light_level'] = serialized_representation['light_level']
        if 'time_of_day' in serialized_representation:
            kwargs['time_of_day'] = serialized_representation['time_of_day']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    def _store_config(self):
        """
        Save the configuration information for the simulator into a config file where the simulator will read it.
        This makes sure we're connecting on the right port, and generating the desired resolution.
        :return: void
        """
        if self._host == 'localhost' and self._actual_executable is not None:
            with open(os.path.join(os.path.dirname(self._actual_executable), 'unrealcv.ini'), 'w') as file:
                file.write(TEMPLATE_UNREALCV_SETTINGS.format(port=self._port,
                                                             width=self._resolution[0],
                                                             height=self._resolution[1]))

    def _request_image(self, viewmode):
        filename = self._client.request('vget /camera/0/{0}'.format(viewmode))
        data = image_utils.read_colour(filename)
        os.remove(filename)     # Clean up after ourselves, now that we have the image data
        if len(data.shape) >= 3 and data.shape[2] >= 4:
            # Slice off the alpha channel if there is one, we're going to ignore it.
            data = data[:, :, 0:3]
        return data

    def _get_image(self):
        if self._client is not None:
            if self._lit_mode:
                image_data = self._request_image('lit')
            else:
                image_data = self._request_image('unlit')
            image_data = np.ascontiguousarray(image_data)
            depth_data = None
            ground_truth_depth_data = None
            labels_data = None
            world_normals_data = None

            # Generate depth
            if self.is_depth_available:
                ground_truth_depth_data = self._request_image('depth')
                # I've encoded the depth into all three channels, Red is depth / 65536, green depth on 256,
                # and blue raw depth. Green and blue channels loop within their ranges, red clamps.
                ground_truth_depth_data = np.asarray(ground_truth_depth_data, dtype=np.float32)   # Back to floats
                ground_truth_depth_data = np.sum(ground_truth_depth_data * (255, 1, 1/255), axis=2)   # Rescale the channels and combine.
                # We now have depth in unreal world units, ie, centimenters. Convert to meters.
                ground_truth_depth_data /= 100

            if self.is_per_pixel_labels_available:
                labels_data = self._request_image('object_mask')

            if self.is_normals_available:
                world_normals_data = self._request_image('normal')

            if self.is_stereo_available:
                cached_pose = self.current_pose
                right_relative_pose = tf.Transform((0, -1 * self._stereo_offset, 0))
                right_pose = self.current_pose.find_independent(right_relative_pose)
                self.set_camera_pose(right_pose)

                if self._lit_mode:
                    right_image_data = self._request_image('lit')
                else:
                    right_image_data = self._request_image('unlit')
                right_ground_truth_depth_data = None
                right_labels = None
                right_world_normals = None

                if self.is_depth_available:
                    right_ground_truth_depth_data = self._request_image('depth')
                    # This is the same as for the base depth, above.
                    right_ground_truth_depth_data = np.asarray(right_ground_truth_depth_data, dtype=np.float32)
                    right_ground_truth_depth_data = np.sum(right_ground_truth_depth_data * (255, 1, 1/255), axis=2)
                    # Convert to meters.
                    right_ground_truth_depth_data /= 100
                    depth_data = arvet.simulation.depth_noise.kinect_depth_model(
                        ground_truth_depth_data,
                        right_ground_truth_depth_data,
                        self.get_camera_intrinsics(),
                        right_relative_pose)
                if self.is_per_pixel_labels_available:
                    right_labels = self._request_image('object_mask')
                if self.is_normals_available:
                    right_world_normals = self._request_image('normal')

                # Reset the camera pose, so that we don't drift right
                self.set_camera_pose(cached_pose)

                return im.StereoImage(left_data=image_data,
                                      right_data=right_image_data,
                                      metadata=self._make_metadata(image_data, ground_truth_depth_data, labels_data,
                                                                   cached_pose, right_pose),
                                      additional_metadata=self._additional_metadata,
                                      left_depth_data=depth_data,
                                      left_ground_truth_depth_data=ground_truth_depth_data,
                                      left_labels_data=labels_data,
                                      left_world_normals_data=world_normals_data,
                                      right_ground_truth_depth_data=right_ground_truth_depth_data,
                                      right_labels_data=right_labels,
                                      right_world_normals_data=right_world_normals)
            else:
                # No stereo, but we still need noisy depth, this is repeated from above
                if self.is_depth_available and ground_truth_depth_data is not None and depth_data is None:
                    # We haven't generated noisy depth yet, we need stereo depth to do that
                    cached_pose = self.current_pose
                    right_relative_pose = tf.Transform((0, -0.15, 0))   # A default separation of 15 cm
                    right_pose = self.current_pose.find_independent(right_relative_pose)
                    self.set_camera_pose(right_pose)

                    right_ground_truth_depth_data = self._request_image('depth')
                    # This is the same as for the base depth, above.
                    right_ground_truth_depth_data = np.asarray(right_ground_truth_depth_data, dtype=np.float32)
                    right_ground_truth_depth_data = np.sum(right_ground_truth_depth_data * (255, 1, 1 / 255), axis=2)
                    # Convert to meters.
                    right_ground_truth_depth_data /= 100

                    depth_data = arvet.simulation.depth_noise.kinect_depth_model(
                        ground_truth_depth_data,
                        right_ground_truth_depth_data,
                        self.get_camera_intrinsics(),
                        right_relative_pose)

                    # Reset the camera pose, so that we don't drift right
                    self.set_camera_pose(cached_pose)

                return im.Image(data=image_data,
                                metadata=self._make_metadata(image_data, ground_truth_depth_data, labels_data, self.current_pose),
                                additional_metadata=self._additional_metadata,
                                depth_data=depth_data,
                                ground_truth_depth_data=ground_truth_depth_data,
                                labels_data=labels_data,
                                world_normals_data=world_normals_data)
        return None

    def _make_metadata(self, im_data, depth_data, label_data, camera_pose, right_camera_pose=None):
        focus_length = self._focus_distance
        aperture = self._aperture
        # if self._client is not None:
        #     fov = self._client.request('vget /camera/0/fov')
        #     focus_length = self._client.request('vget /camera/0/focus-distance')
        #     aperture = self._client.request('vget /camera/0/fstop')
        camera_intrinsics = self.get_camera_intrinsics()

        labelled_objects = []
        if label_data is not None:
            label_colors = set(tuple(color) for m2d in label_data for color in m2d)
            for color in label_colors:
                if color != (0, 0, 0):
                    name = self._client.request("vget /object/name {0} {1} {2}".format(color[0], color[1], color[2]))
                    class_names = self._client.request("vget /object/{0}/labels".format(name))
                    class_names = set(class_names.lower().split(','))

                    # TODO: Other ground-truth bounding boxes could be useful, and are trivial to calculate here
                    # E.g.: Oriented bounding boxes, or fit ellipses. see:
                    # http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
                    labelled_objects.append(imeta.LabelledObject(
                        class_names=class_names,
                        bounding_box=image_utils.get_bounding_box(np.all(label_data == color, axis=2)),
                        label_color=color,
                        relative_pose=self.get_object_pose(name),
                        object_id=name
                    ))

        return imeta.ImageMetadata(
            hash_=xxhash.xxh64(im_data).digest(),
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=camera_pose,
            right_camera_pose=right_camera_pose,
            intrinsics=camera_intrinsics, right_intrinsics=camera_intrinsics,
            environment_type=self._environment_type,
            light_level=self._light_level, time_of_day=self._time_of_day,
            lens_focal_distance=focus_length, aperture=aperture,
            simulator=self.identifier,
            simulation_world=self._world_name,
            lighting_model=imeta.LightingModel.LIT if self._lit_mode else imeta.LightingModel.UNLIT,
            texture_mipmap_bias=None, normal_maps_enabled=None, roughness_enabled=None,
            geometry_decimation=None, procedural_generation_seed=None,
            labelled_objects=labelled_objects,
            average_scene_depth=np.mean(depth_data) if depth_data is not None else None)
