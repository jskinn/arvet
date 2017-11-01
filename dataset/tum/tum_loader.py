# Copyright (c) 2017, John Skinner
import os.path
import xxhash
import cv2
import util.associate
import metadata.camera_intrinsics as cam_intr
import metadata.image_metadata as imeta
import util.transform as tf
import core.image_entity
import dataset.image_collection_builder


def make_camera_pose(tx, ty, tz, qx, qy, qz, qw):
    # TODO: Check the TUM dataset coordinate frame
    """
    TUM dataset use a different coordinate frame to the one I'm using, which is the same as the Libviso2 frame.
    This function is to convert dataset ground-truth poses to transform objects.
    Thankfully, its still a right-handed coordinate frame, which makes this easier.
    Frame is: z forward, y right, x down

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
            parts = line.split(' ')
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
            parts = line.split(' ')
            if len(parts) >= 8:
                timestamp, tx, ty, tz, qx, qy, qz, qw = parts[0:8]
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
            matches = util.associate.associate(root_map, other_map, offset=0, max_difference=1)
            rekeyed_map = {root_key: other_map[other_key] for root_key, other_key in matches}
            root_keys &= set(rekeyed_map.keys())
            rekeyed_maps.append(rekeyed_map)
        return sorted([key, root_map[key]] + [rekeyed_map[key] for rekeyed_map in rekeyed_maps]
                      for key in root_keys)


def get_camera_intrinsics(folder_path):
    folder_path = folder_path.lower()
    if 'freiburg1' in folder_path:
        return cam_intr.CameraIntrinsics(
            width=640,
            height=480,
            fx=517.3,
            fy=516.5,
            cx=318.6,
            cy=255.3,
            k1=0.2624,
            k2=-0.9531,
            k3=1.1633,
            p1=-0.0054,
            p2=0.0026
        )
    elif 'freiburg2' in folder_path:
        return cam_intr.CameraIntrinsics(
            width=640,
            height=480,
            fx=580.8,
            fy=581.8,
            cx=308.8,
            cy=253.0,
            k1=-0.2297,
            k2=1.4766,
            k3=-3.4194,
            p1=0.0005,
            p2=-0.0075
        )
    elif 'frieburg3' in folder_path:
        return cam_intr.CameraIntrinsics(
            width=640,
            height=480,
            fx=535.4,
            fy=539.2,
            cx=320.1,
            cy=247.6
        )
    else:
        # Default to ROS parameters
        return cam_intr.CameraIntrinsics(
            width=640,
            height=480,
            fx=525.0,
            fy=525.0,
            cx=319.5,
            cy=239.5
        )


def import_dataset(root_folder, db_client):
    """
    Load a TUM image sequences into the database.
    :return:
    """
    if not os.path.isdir(root_folder):
        return None

    # Step 1: Read the meta-information from the files
    rgb_path = os.path.join(root_folder, 'rgb.txt')
    trajectory_path = os.path.join(root_folder, 'groundtruth.txt')
    depth_path = os.path.join(root_folder, 'depth.txt')

    if not os.path.isfile(rgb_path) or not os.path.isfile(trajectory_path) or not os.path.isfile(depth_path):
        # Stop if we can't find the metadata files within the directory
        return None

    image_files = read_image_filenames(rgb_path)
    trajectory = read_trajectory(trajectory_path)
    depth_files = read_image_filenames(depth_path)

    # Step 2: Associate the different data types by timestamp
    all_metadata = associate_data(image_files, trajectory, depth_files)

    # Step 3: Load the images from the metadata
    builder = dataset.image_collection_builder.ImageCollectionBuilder(db_client)
    for timestamp, image_file, camera_pose, depth_file in all_metadata:
        rgb_data = cv2.imread(os.path.join(root_folder, image_file), cv2.IMREAD_COLOR)
        depth_data = cv2.imread(os.path.join(root_folder, depth_file), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data / 5000  # Re-scale depth to meters
        camera_intrinsics = get_camera_intrinsics(root_folder)
        builder.add_image(image=core.image_entity.ImageEntity(
            data=rgb_data[:, :, ::-1],
            depth_data=depth_data,
            metadata=imeta.ImageMetadata(
                hash_=xxhash.xxh64(rgb_data).digest(),
                camera_pose=camera_pose,
                intrinsics=camera_intrinsics,
                source_type=imeta.ImageSourceType.REAL_WORLD,
                environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                light_level=imeta.LightingLevel.WELL_LIT,
                time_of_day=imeta.TimeOfDay.DAY,
            )
        ), timestamp=timestamp)
    return builder.save()
