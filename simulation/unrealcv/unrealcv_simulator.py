import os
import xxhash
import numpy as np
import cv2
import unrealcv

import core.image as im
import metadata.image_metadata as imeta
import simulation.simulator
import util.dict_utils as du
import util.unreal_transform as uetf


class UnrealCVSimulator(simulation.simulator.Simulator):

    def __init__(self, config=None):
        """
        Create an unrealcv simulator with a bunch of configuration.
        :param config: Simulator configuration
        """
        if config is None:
            config = {}
        du.defaults(config, {
            'stereo_offset': 0,  # unreal units
            'provide_rgb': True,
            'provide_depth': False,
            'provide_labels': False,
            'provide_world_normals': False,

            # Simulation server config
            'host': 'localhost',
            'port': 9000,

            'metadata': {}
        })
        # Simulation metadata, provided as kwargs to ImageMetadata
        du.defaults(config['metadata'], {
            'environment_type': imeta.EnvironmentType.INDOOR,
            'light_level': imeta.LightingLevel.EVENLY_LIT,
            'time_of_day': imeta.TimeOfDay.DAY,
            'simulation_world': 'UnrealWorld'
        })

        self._stereo_offset = float(config['stereo_offset'])
        self._provide_depth = bool(config['provide_depth'])
        self._provide_labels = bool(config['provide_labels'])
        self._provide_world_normals = bool(config['provide_world_normals'])

        self._host = str(config['host'])
        self._port = int(config['port'])

        self._metadata = {
            'environment_type': imeta.EnvironmentType(config['metadata']['environment_type']),
            'light_level': imeta.LightingLevel(config['metadata']['light_level']),
            'time_of_day': imeta.TimeOfDay(config['metadata']['time_of_day']),
            'simulation_world': str(config['metadata']['simulation_world'])
        }
        self._additional_metadata = {k: v for k, v in config['metadata'].items() if k not in self._metadata}

        self._client = None
        self._current_pose = None
        self._fov = None
        self._focus_distance = None
        self._aperture = None
        self._lit_mode = (bool(config['metadata']['lit']) if 'metadata' in config and
                                                             'lit' in config['metadata'] else True)

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

    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff the simulator has started correctly.
        """
        # TODO: Launch external process running the simulator
        if self._client is None:
            self._client = unrealcv.Client((self._host, self._port))
        if self._client.isconnected():
            return True
        conn = self._client.connect()
        if conn:
            # TODO: Read the
            self._fov = 90
            self._focus_distance = -1
            self._aperture = -1

    def get_next_image(self):
        """
        Get the next image from this source.
        It assumes that the simulation state has been changed externally
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images

        :return: An Image object (see core.image) or None, and None for the timestamp
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
        # TODO: Stop the external process.
        self._client.disconnect()
        self._current_pose = None
        self._fov = None
        self._focus_distance = None
        self._aperture = None

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
            self._client.request("vset /camera/0/location {0} {1} {2}".format(pose.location[0],
                                                                              pose.location[1],
                                                                              pose.location[2]))
            self._client.request("vset /camera/0/rotation {0} {1} {2}".format(pose.pitch,
                                                                              pose.yaw,
                                                                              pose.roll))

    @property
    def field_of_view(self):
        return self._fov

    def set_field_of_view(self, fov):
        """
        Set the field of view of the simulator
        :param fov:
        :return:
        """
        if self._client is not None:
            self._client.request("vset /camera/0/fov {0}".format(float(fov)))

    @property
    def focus_distance(self):
        if self._focus_distance is not None and self._focus_distance >= 0:
            return self._focus_distance
        elif self._client is not None:
            return float(self._client.request("vget /camera/0/fov"))
        return None

    def set_focus_distance(self, focus_distance):
        """
        Set the focus distance of the simulator.
        This overrides the autofocus, you need to call set_autofocus again to focus automatically
        :param focus_distance:
        :return:
        """
        if self._client is not None:
            self._focus_distance = focus_distance
            self._client.request("vset /camera/0/autofocus 0")
            self._client.request("vset /camera/0/focus-distance {0}".format(float(focus_distance)))

    def set_autofocus(self, autofocus):
        """
        Set to the simulator to automatically choose the focus distance
        :param autofocus:
        :return:
        """
        if self._client is not None:
            self._focus_distance = -1
            self._client.request("vset /camera/0/autofocus {0}".format(int(bool(autofocus))))

    @property
    def fstop(self):
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

    def get_camera_matrix(self):
        """
        Get the camera matrix for this simulator.
        TODO: get it correctly from unrealcv
        :return: The camera matrix as a numpy array
        """
        return np.identity(3)

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
                    return uetf.transform_from_unreal(ue_trans)
        return None

    def num_visible_objects(self):
        if self._client is not None:
            result = self._client.request('vget /object/num_visible')
            return int(result)
        return 0

    def _request_image(self, viewmode):
        filename = self._client.request('vget /camera/0/{0}'.format(viewmode))
        data = cv2.imread(filename)
        os.remove(filename)     # Clean up after ourselves, now that we have the image data
        return np.ascontiguousarray(data[:, :, ::-1], dtype='uint8')

    def _get_image(self):
        if self._client is not None:
            if self._lit_mode:
                image_data = self._request_image('lit')
            else:
                image_data = self._request_image('unlit')
            depth_data = None
            labels_data = None
            world_normals_data = None

            if self.is_depth_available:
                depth_data = self._request_image('depth')
            if self.is_per_pixel_labels_available:
                labels_data = self._request_image('object_mask')
            if self.is_normals_available:
                world_normals_data = self._request_image('normal')

            if self.is_stereo_available:
                cached_pose = self.current_pose
                right_pose = self.current_pose.find_independent((0, 0, self._stereo_offset))
                self.set_camera_pose(right_pose)

                if self._lit_mode:
                    right_image_data = self._request_image('lit')
                else:
                    right_image_data = self._request_image('unlit')
                right_depth = None
                right_labels = None
                right_world_normals = None

                if self.is_depth_available:
                    right_depth = self._request_image('depth')
                if self.is_per_pixel_labels_available:
                    right_labels = self._request_image('object_mask')
                if self.is_normals_available:
                    right_world_normals = self._request_image('normal')

                return im.StereoImage(left_data=image_data,
                                      right_data=right_image_data,
                                      left_camera_pose=cached_pose,
                                      right_camera_pose=right_pose,
                                      metadata=self._make_metadata(image_data, depth_data, labels_data,
                                                                   cached_pose, right_pose),
                                      additional_metadata=self._additional_metadata,
                                      left_depth_data=depth_data,
                                      left_labels_data=labels_data,
                                      left_world_normals_data=world_normals_data,
                                      right_depth_data=right_depth,
                                      right_labels_data=right_labels,
                                      right_world_normals_data=right_world_normals)
            return im.Image(data=image_data,
                            metadata=self._make_metadata(image_data, depth_data, labels_data, self.current_pose),
                            additional_metadata=self._additional_metadata,
                            depth_data=depth_data,
                            labels_data=labels_data,
                            world_normals_data=world_normals_data)
        return None

    def _make_metadata(self, im_data, depth_data, label_data, camera_pose, right_camera_pose=None):
        fov = None
        focus_length = None
        aperture = None
        if self._client is not None:
            fov = self._client.request('vget /camera/0/fov')
            focus_length = self._client.request('vget /camera/0/focus-distance')
            aperture = self._client.request('vget /camera/0/fstop')

        labelled_objects = []
        if label_data is not None:
            label_colors = set(tuple(color) for m2d in label_data for color in m2d)
            for color in label_colors:
                if color != (0, 0, 0):
                    name = self._client.request("vget /object/name {0} {1} {2}".format(color[0], color[1], color[2]))
                    class_names = self._client.request("vget /object/{0}/labels".format(name))
                    class_names = set(class_names.lower().split(','))

                    label_points = cv2.findNonZero(np.asarray(np.all(label_data == color, axis=2), dtype='uint8'))
                    # TODO: Other ground-truth bounding boxes could be useful, and are trivial to calculate here
                    # E.g.: Oriented bounding boxes, or fit ellipses. see:
                    # http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
                    labelled_objects.append(imeta.LabelledObject(
                        class_names=class_names,
                        bounding_box=cv2.boundingRect(label_points),
                        label_color=color,
                        relative_pose=self.get_object_pose(name),
                        object_id=name
                    ))

        return imeta.ImageMetadata(
            hash_=xxhash.xxh64(im_data).digest(),
            source_type=imeta.ImageSourceType.SYNTHETIC, height=im_data.shape[0], width=im_data.shape[1],
            camera_pose=camera_pose,
            right_camera_pose=right_camera_pose,
            environment_type=self._metadata['environment_type'],
            light_level=self._metadata['light_level'], time_of_day=self._metadata['time_of_day'],
            fov=fov, focal_length=focus_length, aperture=aperture,
            simulation_world=self._metadata['simulation_world'],
            lighting_model=imeta.LightingModel.LIT if self._lit_mode else imeta.LightingModel.UNLIT,
            texture_mipmap_bias=None, normal_maps_enabled=None, roughness_enabled=None,
            geometry_decimation=None, procedural_generation_seed=None,
            labelled_objects=labelled_objects,
            average_scene_depth=np.mean(depth_data) if depth_data is not None else None)
