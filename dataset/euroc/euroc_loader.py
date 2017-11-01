# Copyright (c) 2017, John Skinner
import os.path
import numpy as np
import xxhash
import cv2
import util.associate
import metadata.camera_intrinsics as cam_intr
import metadata.image_metadata as imeta
import util.transform as tf
import core.image_entity
import dataset.image_collection_builder
import yaml
try:
    from yaml import CDumper as YamlDumper, CLoader as YamlLoader
except ImportError:
    from yaml import Dumper as YamlDumper, Loader as YamlLoader


def make_camera_pose(tx, ty, tz, qx, qy, qz, qw):
    """
    As far as I can tell, EuRoC uses ROS coordinates, which are the same as my coordinates.

    :param tx: The x coordinate of the location
    :param ty: The y coordinate of the location
    :param tz: The z coordinate of the location
    :param qx: The x part of the quaternion orientation
    :param qy: The y part of the quaternion orientation
    :param qz: The z part of the quaternion orientation
    :param qw: The scalar part of the quaternion orientation
    :return: A Transform object representing the world pose of the current frame
    """
    return tf.Transform(
        location=(tx, ty, tz),
        rotation=(qw, qx, qy, qz),
        w_first=True
    )


def read_image_filenames(images_file_path):
    filename_map = {}
    with open(images_file_path, 'r') as images_file:
        for line in images_file:
            if line.startswith('#'):
                # This line is a comment
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                timestamp, relative_path = parts[0:2]
                filename_map[float(timestamp)] = relative_path.rstrip()  # To remove trailing newlines
    return filename_map


def read_trajectory(trajectory_filepath):
    """
    Read the ground-truth camera trajectory from file
    :param trajectory_filepath:
    :return: A map of timestamp to camera pose.
    """
    trajectory = {}
    with open(trajectory_filepath, 'r') as trajectory_file:
        for line in trajectory_file:
            if line.startswith('#'):
                # This line is a comment, skip and continue
                continue
            parts = line.split(',')
            if len(parts) >= 8:
                timestamp, tx, ty, tz, qw, qx, qy, qz = parts[0:8]
                trajectory[float(timestamp)] = make_camera_pose(float(tx), float(ty), float(tz),
                                                                float(qx), float(qy), float(qz), float(qw))
    return trajectory


def associate_data(root_map, *args):
    """
    Convert a number of maps key->value to a list of lists
    [[key, map1[key], map2[key] map3[key] ...] ...]

    The list will be sorted in key order
    Returned inner lists will be in the same order as they are passed as arguments.

    The first map passed is considered the reference point for the list of keys,
    :param root_map: The first map to associate
    :param args: Additional maps to associate to the first one
    :return:
    """
    if len(args) <= 0:
        # Nothing to associate, flatten the root map and return
        return sorted([k, v] for k, v in root_map.items())
    root_keys = set(root_map.keys())
    all_same = True
    # First, check if all the maps have the same list of keys
    for other_map in args:
        if set(other_map.keys()) != root_keys:
            all_same = False
            break
    if all_same:
        # All the maps have the same set of keys, just flatten them
        return sorted([key, root_map[key]] + [other_map[key] for other_map in args]
                      for key in root_keys)
    else:
        # We need to associate the maps, the timestamps are a little out
        rekeyed_maps = []
        for other_map in args:
            matches = util.associate.associate(root_map, other_map, offset=0, max_difference=3)
            rekeyed_map = {root_key: other_map[other_key] for root_key, other_key in matches}
            root_keys &= set(rekeyed_map.keys())
            rekeyed_maps.append(rekeyed_map)
        return sorted([key, root_map[key]] + [rekeyed_map[key] for rekeyed_map in rekeyed_maps]
                      for key in root_keys)


def get_camera_calibration(sensor_yaml_path):
    with open(sensor_yaml_path, 'r') as sensor_file:
        sensor_data = yaml.load(sensor_file, YamlLoader)

    d = sensor_data['T_BS']['data']
    extrinsics = tf.Transform(np.array([
        [d[0], d[1], d[2], d[3]],
        [d[4], d[5], d[6], d[7]],
        [d[8], d[9], d[10], d[11]],
        [d[12], d[13], d[14], d[15]],
    ]))
    resolution = sensor_data['resolution']
    intrinsics = cam_intr.CameraIntrinsics(
        width=resolution[0],
        height=resolution[1],
        fx=sensor_data['intrinsics'][0],
        fy=sensor_data['intrinsics'][1],
        cx=sensor_data['intrinsics'][2],
        cy=sensor_data['intrinsics'][3],
        k1=sensor_data['distortion_coefficients'][0],
        k2=sensor_data['distortion_coefficients'][1],
        p1=sensor_data['distortion_coefficients'][2],
        p2=sensor_data['distortion_coefficients'][3]
    )
    return extrinsics, intrinsics


def import_dataset(root_folder, db_client):
    """
    Load a TUM image sequences into the database.
    :return:
    """
    if not os.path.isdir(root_folder):
        return None

    # Step 1: Read the meta-information from the files
    left_rgb_path = os.path.join(root_folder, 'cam0', 'data.csv')
    left_camera_intrinsics_path = os.path.join(root_folder, 'cam0', 'sensor.yaml')
    right_rgb_path = os.path.join(root_folder, 'cam1', 'data.csv')
    right_camera_intrinsics_path = os.path.join(root_folder, 'cam1', 'sensor.yaml')
    trajectory_path = os.path.join(root_folder, 'state_groundtruth_estimate0', 'data.csv')

    if (not os.path.isfile(left_rgb_path) or not os.path.isfile(left_camera_intrinsics_path) or
            not os.path.isfile(right_rgb_path) or not os.path.isfile(right_camera_intrinsics_path) or
            not os.path.isfile(trajectory_path)):
        # Stop if we can't find the metadata files within the directory
        return None

    left_image_files = read_image_filenames(left_rgb_path)
    left_extrinsics, left_intrinsics = get_camera_calibration(left_camera_intrinsics_path)
    right_image_files = read_image_filenames(left_rgb_path)
    right_extrinsics, right_intrinsics = get_camera_calibration(right_camera_intrinsics_path)
    trajectory = read_trajectory(trajectory_path)

    # Step 2: Associate the different data types by timestamp. Trajectory last because it's bigger than the stereo.
    all_metadata = associate_data(left_image_files, right_image_files, trajectory)

    # Step 3: Load the images from the metadata
    builder = dataset.image_collection_builder.ImageCollectionBuilder(db_client)
    for timestamp, left_image_file, right_image_file, robot_pose in all_metadata:
        left_data = cv2.imread(os.path.join(root_folder, 'cam0', 'data', left_image_file), cv2.IMREAD_COLOR)
        right_data = cv2.imread(os.path.join(root_folder, 'cam1', 'data', right_image_file), cv2.IMREAD_COLOR)
        left_data = np.ascontiguousarray(left_data[:, :, ::-1])
        right_data = np.ascontiguousarray(right_data[:, :, ::-1])

        left_pose = robot_pose.find_independent(left_extrinsics)
        right_pose = robot_pose.find_independent(right_extrinsics)

        builder.add_image(image=core.image_entity.StereoImageEntity(
            left_data=left_data,
            right_data=right_data,
            metadata=imeta.ImageMetadata(
                hash_=xxhash.xxh64(left_data).digest(),
                camera_pose=left_pose,
                right_camera_pose=right_pose,
                intrinsics=left_intrinsics,
                right_intrinsics=right_intrinsics,
                source_type=imeta.ImageSourceType.REAL_WORLD,
                environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                light_level=imeta.LightingLevel.WELL_LIT,
                time_of_day=imeta.TimeOfDay.DAY,
            )
        ), timestamp=timestamp)
    return builder.save()
