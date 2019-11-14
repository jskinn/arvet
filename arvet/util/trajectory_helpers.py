# Copyright (c) 2017, John Skinner
import typing
import numpy as np
import arvet.util.transform as tf
import arvet.util.associate


def get_trajectory_for_image_source(image_collection) -> typing.Mapping[float, tf.Transform]:
    """
    Image collections are too large for us to load into memory here,
    but we need to be able to do logic on their trajectores.
    This utility uses the database to get just the trajectory for a given image collection.
    Only works for image collections, due to their database structure.
    The returned trajectory starts at (0, 0, 0) and performs the same relative motion as the original trajectory,
    but does not have the same world coordinates.
    :param image_collection: The id of the image collection to load
    :return: A trajectory, a map of timestamp to camera pose. Ignores right-camera for stereo
    """
    trajectory = {}
    first_pose = None
    for timestamp, image in image_collection:
        current_pose = image.camera_pose
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


def find_trajectory_scale(trajectory: typing.Mapping[float, typing.Union[tf.Transform, None]]) -> float:
    """
    Find the average speed of the trajectory as the scale.
    That is, the average distance moved between frames. We use this to rescale trajectories for comparison.
    Timestamps matched to 'None' indicate that the motion is unknown within that time, and the scale can't be found
    :param trajectory:
    :return:
    """
    if len(trajectory) <= 1:
        return 0
    timestamps = sorted(time for time, position in trajectory.items() if position is not None)
    speeds = [
        np.linalg.norm(trajectory[t1].location - trajectory[t0].location) / (t1 - t0)
        for t1, t0 in zip(timestamps[1:], timestamps[:-1])
    ]
    if len(speeds) <= 0:
        return 0
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
    if len(trajectory) <= 1:
        return trajectory
    current_scale = find_trajectory_scale(trajectory)
    timestamps = sorted(trajectory.keys())
    scaled_trajectory = {timestamps[0]: trajectory[timestamps[0]]}
    prev_time = None
    for idx, time in enumerate(timestamps):
        if trajectory[time] is None:
            scaled_trajectory[time] = None
        elif prev_time is None:
            prev_time = time
            scaled_trajectory[time] = trajectory[time]
        else:
            t0 = prev_time
            t1 = time
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
    if len(trajectory) <= 1:
        return {}
    times = sorted(trajectory.keys())
    prev_time = times[0]
    motions = {}
    for time in times[1:]:
        motion = trajectory[prev_time].find_relative(trajectory[time])
        prev_time = time
        motions[time] = motion
    return motions


def compute_average_trajectory(trajectories: typing.Iterable[typing.Mapping[float, tf.Transform]]) \
        -> typing.Mapping[float, tf.Transform]:
    """
    Find the average trajectory from a number of estimated trajectories.
    Can handle small variations in timestamps, but privileges timestamps from earlier trajectories for association
    :param trajectories:
    :return:
    """
    associated_times = {}
    associated_poses = {}
    for traj in trajectories:
        traj_times = set(traj.keys())
        # First, add all the times that can be associated to an existing time
        matches = arvet.util.associate.associate(associated_times, traj, offset=0, max_difference=0.1)
        for match in matches:
            associated_times[match[0]].append(match[1])
            if traj[match[1]] is not None:
                associated_poses[match[0]].append(traj[match[1]])
            traj_times.remove(match[1])
        # Add all the times in this trajectory that don't have associations yet
        for time in traj_times:
            associated_times[time] = [time]
            associated_poses[time] = [traj[time]] if traj[time] is not None else []
    # Take the median associated time and pose together
    return {
        np.median(associated_times[time]): (
            tf.compute_average_pose(associated_poses[time])
            if time in associated_poses and len(associated_poses[time]) > 0
            else None
        )
        for time in associated_times.keys()
    }
