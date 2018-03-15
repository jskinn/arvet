# Copyright (c) 2017, John Skinner
import typing
import bson
import numpy as np
import arvet.database.client
import arvet.util.transform as tf


def get_trajectory_for_image_source(db_client: arvet.database.client.DatabaseClient,
                                    image_collection_id: bson.ObjectId) -> typing.Mapping[float, tf.Transform]:
    """
    Image collections are too large for us to load into memory here,
    but we need to be able to do logic on their trajectores.
    This utility uses the database to get just the trajectory for a given image collection.
    Only works for image collections, due to their database structure.
    The returned trajectory starts at (0, 0, 0) and performs the same relative motion as the original trajectory,
    but does not have the same world coordinates.
    :param db_client: The database client
    :param image_collection_id: The id of the image collection to load
    :return: A trajectory, a map of timestamp to camera pose. Ignores right-camera for stereo
    """
    images = db_client.image_source_collection.find_one({'_id': image_collection_id, 'images': {'$exists': True}},
                                                        {'images': True})
    trajectory = {}
    if images is not None:
        first_pose = None
        for timestamp, image_id in images['images']:
            position_result = db_client.image_collection.find_one({'_id': image_id}, {'metadata.camera_pose': True})
            if position_result is not None:
                current_pose = tf.Transform.deserialize(position_result['metadata']['camera_pose'])
                if first_pose is None:
                    trajectory[timestamp] = tf.Transform()
                    first_pose = current_pose
                else:
                    trajectory[timestamp] = first_pose.find_relative(current_pose)
    return trajectory


def zero_trajectory(trajectory: typing.Mapping[float, tf.Transform]) -> typing.Mapping[float, tf.Transform]:
    """
    Reset a trajectory to start at (0, 0, 0) facing along the x-axis.
    Useful when comparing trajectories that could start at any arbitrary location
    :param trajectory: The mapping from
    :return:
    """
    first_pose = trajectory[min(trajectory.keys())]
    return {
        stamp: first_pose.find_relative(pose)
        for stamp, pose in trajectory.items()
    }


def find_trajectory_scale(trajectory: typing.Mapping[float, tf.Transform]) -> float:
    """
    Find the average speed of the trajectory as the scale.
    That is, the average distance moved between frames. We use this to rescale trajectories for comparison
    :param trajectory:
    :return:
    """
    timestamps = sorted(trajectory.keys())
    speeds = []
    for idx in range(1, len(timestamps)):
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        dist = np.linalg.norm(trajectory[t1].location - trajectory[t0].location)
        speeds.append(dist / (t1 - t0))
    return float(np.mean(speeds))


def rescale_trajectory(trajectory: typing.Mapping[float, tf.Transform], scale: float) \
        -> typing.Mapping[float, tf.Transform]:
    """
    Rescale a trajectory to have a given average distance moved between frames.
    Multiplies each frame motion by desired scale / current scale.
    Does not affect rotations.
    Use find_trajectory_scale on a reference trajectory to get the scale parameter
    :param trajectory: The trajectory to rescale
    :param scale: The desired average motion between frames
    :return: The same motions, rescaled to have a certain average speed.
    """
    current_scale = find_trajectory_scale(trajectory)
    timestamps = sorted(trajectory.keys())
    scaled_trajectory = {timestamps[0]: trajectory[timestamps[0]]}
    for idx in range(1, len(timestamps)):
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        motion = trajectory[t0].find_relative(trajectory[t1])
        scaled_trajectory[t1] = tf.Transform(
            location=scaled_trajectory[t0].find_independent(
                (scale / current_scale) * motion.location
            ),
            rotation=trajectory[t1].rotation_quat(w_first=True),
            w_first=True
        )
    return scaled_trajectory


def trajectory_to_motion_sequence(trajectory: typing.Mapping[float, tf.Transform]) -> \
        typing.Mapping[float, tf.Transform]:
    """
    Convert a trajectory into a sequence of relative motions.
    Useful for comparing motio
    :param trajectory:
    :return:
    """
    times = sorted(trajectory.keys())
    prev_time = times[0]
    motions = {}
    for time in times[1:]:
        motion = trajectory[prev_time].find_relative(trajectory[time])
        prev_time = time
        motions[time] = motion
    return motions
