# Copyright (c) 2017, John Skinner
import pickle
import bson
import numpy as np
import core.sequence_type
import database.entity
import simulation.simulator
import simulation.controller
import util.transform as tf
import util.associate


class TrajectoryFollowController(simulation.controller.Controller, database.entity.Entity):
    """
    A controller that follows a fixed sequence of poses.
    This is useful for copying the path of one sequence in a new generated sequence.
    This produces either sequential image data.
    """

    def __init__(self, trajectory, sequence_type, trajectory_source=None, id_=None):
        """
        Create The controller, with some some settings
        :param trajectory: The trajectory the camera will follow
        :param trajectory_source: The ID of the root image source that this trajectory is following. For uniqueness.
        :param sequence_type: The sequence type produced by following this trajectory.
        """
        super().__init__(id_=id_)
        self._trajectory = trajectory
        self._trajectory_source = trajectory_source
        self._sequence_type = core.sequence_type.ImageSequenceType(sequence_type)
        self._timestamps = sorted(trajectory.keys())
        self._current_index = 0
        self._simulator = None

    def __len__(self):
        return len(self._trajectory)

    def __getitem__(self, item):
        return self.get(item)

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def trajectory_source(self):
        return self._trajectory_source

    @property
    def sequence_type(self):
        """
        Get the kind of image sequence produced by this controller.
        Depends on the trajectory really.
        :return: ImageSequenceType
        """
        return self._sequence_type

    def supports_random_access(self):
        """
        True iff we can randomly access the images in this source by index.
        Because we know all the poses before time, this does support random access.
        :return:
        """
        return True

    @property
    def is_depth_available(self):
        """
        Can this image source produce depth images.
        Determined by the underlying simulator
        :return:
        """
        return self._simulator is not None and self._simulator.is_depth_available

    @property
    def is_per_pixel_labels_available(self):
        """
        Do images from this image source include object labels.
        Determined by the underlying simulator.
        :return: True if this image source can produce object labels for each image
        """
        return self._simulator is not None and self._simulator.is_per_pixel_labels_available

    @property
    def is_labels_available(self):
        """
        Do images from this source include object bounding boxes and simple labels in their metadata.
        Determined by the underlying simulator.
        :return: True iff the image metadata includes bounding boxes
        """
        return self._simulator is not None and self._simulator.is_labels_available

    @property
    def is_normals_available(self):
        """
        Do images from this image source include world normals.
        Determined by the underlying simulator.
        :return: True if images have world normals associated with them
        """
        return self._simulator is not None and self._simulator.is_normals_available

    @property
    def is_stereo_available(self):
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images.
        Determined by the underlying simulator.
        :return:
        """
        return self._simulator is not None and self._simulator.is_stereo_available

    @property
    def is_stored_in_database(self):
        """
        Do this images from this source come from the database.
        Since they come from a simulator, it doesn't seem likely.
        :return:
        """
        return False

    def get_camera_intrinsics(self):
        """
        Get the camera intrinsics from the simulator
        :return:
        """
        return self._simulator.get_camera_intrinsics() if self._simulator is not None else None

    def get_stereo_baseline(self):
        """
        Get the stereo baseline from the simulator if it is in stereo mode
        :return:
        """
        return self._simulator.get_stereo_baseline() if self._simulator is not None else None

    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff we have successfully started iteration
        """
        if self._simulator is None:
            return False
        self._simulator.begin()
        self._current_index = 0

    def get(self, index):
        """
        Get an image by timestamp
        :param index: The timestamp to get the image for
        :return:
        """
        if self._simulator is not None and index in self._trajectory:
            self._simulator.set_camera_pose(self._trajectory[index])
            image, _ = self._simulator.get_next_image()
            return image, index
        return None, None

    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images.
        The second return value must always be the time

        :return: An Image object (see core.image) or None, and an index (or None)
        """
        if self._simulator is not None:
            timestamp = self._timestamps[self._current_index]
            image, _ = self.get(timestamp)
            self._current_index += 1
            return image, timestamp
        return None, None

    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        This controller is complete when it has produced the configured number of images.
        If it must return to start, it is not complete until it subsequently returns to the start point.
        :return:
        """
        return self._current_index >= len(self)

    def shutdown(self):
        """
        Shut down the controller, also shutting down the simulator.
        :return:
        """
        if self._simulator is not None:
            self._simulator.shutdown()

    def can_control_simulator(self, simulator):
        """
        Can this controller control the given simulator.
        This one is pretty general, it needs to actually be a simulator though
        :param simulator: The simulator we may potentially control
        :return:
        """
        return simulator is not None and isinstance(simulator, simulation.simulator.Simulator)

    def set_simulator(self, simulator):
        """
        Set the simulator used by this controller
        :param simulator:
        :return:
        """
        if self.can_control_simulator(simulator):
            self._simulator = simulator

    def validate(self):
        valid = super().validate()
        return valid

    def serialize(self):
        serialized = super().serialize()
        serialized['trajectory'] = bson.Binary(pickle.dumps(self._trajectory, protocol=pickle.HIGHEST_PROTOCOL))
        serialized['trajectory_source'] = self._trajectory_source
        serialized['sequence_type'] = self._sequence_type.value
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trajectory' in serialized_representation:
            kwargs['trajectory'] = pickle.loads(serialized_representation['trajectory'])
        if 'trajectory_source' in serialized_representation:
            kwargs['trajectory_source'] = serialized_representation['trajectory_source']
        if 'sequence_type' in serialized_representation:
            kwargs['sequence_type'] = serialized_representation['sequence_type']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def create_follow_controller(db_client, image_collection_id, sequence_type):
    """
    Create and save an new trajectory follow controller following a given image collection.
    checking for uniqueness. Due to the trajectories, this is harder than it seems.
    :param db_client: The database client, to
    :param image_collection_id:
    :param sequence_type: The sequence type to create. This is a hack, should use the value from the collection.
    :return:
    """
    # First, look for an existing controller indexed against this image collection
    existing = db_client.image_source_collection.find_one({
        '_type': TrajectoryFollowController.__module__ + '.' + TrajectoryFollowController.__name__,
        'trajectory_source': image_collection_id
    }, {'_id': True})
    if existing is not None:
        return existing['_id']
    # No direct controller, look for one which has the same trajectory.
    trajectory = get_trajectory_for_image_source(db_client, image_collection_id)
    if len(trajectory) <= 0:
        return None
    existing = db_client.image_source_collection.find({
        '_type': TrajectoryFollowController.__module__ + '.' + TrajectoryFollowController.__name__})
    for s_follow_controller in existing:
        controller = TrajectoryFollowController.deserialize(s_follow_controller, db_client)
        matches = util.associate.associate(trajectory, controller.trajectory, max_difference=0.1, offset=0)
        if len(matches) < len(trajectory):
            continue
        ok = True
        for stamp1, stamp2 in matches:
            pose1 = trajectory[stamp1]
            pose2 = controller.trajectory[stamp2]
            trans_diff = pose1.location - pose2.location
            rot_diff = pose1.rotation_quat(w_first=True) - pose2.rotation_quat(w_first=True)
            if np.dot(trans_diff, trans_diff) > 0.01 or np.dot(rot_diff, rot_diff) > 0.01:
                ok = False
                break
        if ok:
            # Got a controller
            if controller.trajectory_source is None:
                db_client.image_source_collection.update({'_id': controller.identifier},
                                                         {'$set': {'trajectory_source': image_collection_id}})
            return controller.identifier
    # Couldn't find an existing one, make a new one.
    controller = TrajectoryFollowController(trajectory=trajectory,
                                            trajectory_source=image_collection_id,
                                            sequence_type=sequence_type)
    return db_client.image_source_collection.insert(controller.serialize())


def get_trajectory_for_image_source(db_client, image_collection_id):
    """
    Image collections are too large for us to load into memory here,
    but we need to be able to do logic on their trajectores.
    This utility uses the database to get just the trajectory for a given image collection.
    Only works for image collections, due to their database structure.
    :param db_client: The database client
    :param image_collection_id: The id of the image collection to load
    :return: A trajectory, a map of timestamp to camera pose. Ignores right-camera for stereo
    """
    images = db_client.image_source_collection.find_one({'_id': image_collection_id, 'images': {'$exists': True}},
                                                        {'images': True})
    trajectory = {}
    if images is not None:
        for timestamp, image_id in images['images']:
            position_result = db_client.image_collection.find_one({'_id': image_id}, {'metadata.camera_pose': True})
            if position_result is not None:
                trajectory[timestamp] = tf.Transform.deserialize(position_result['metadata']['camera_pose'])
    return trajectory
