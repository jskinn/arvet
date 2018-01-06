# Copyright (c) 2017, John Skinner
import typing
import arvet.database.client
import bson
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
