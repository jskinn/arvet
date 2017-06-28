import os.path
import re
import cv2
import copy

import util.transform as tf
import util.database_helpers as db_help
import util.unreal_transform as uetf
import metadata.image_metadata as imeta
import core.image_entity
import core.image_collection
import core.sequence_type


def import_rw_dataset(labels_path, db_client):
    image_ids = []
    with open(labels_path, 'r') as labels_file:
        base_dir = os.path.dirname(labels_path)
        for line in labels_file:
            split = re.split('[, ]', line)
            if len(split) != 6:
                continue
            imfile, x1, y1, x2, y2, label = split
            label = label.rstrip()
            im = cv2.imread(os.path.join(base_dir, imfile))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            bounding_box = imeta.BoundingBox(
                class_name=label,
                confidence=1,
                x=int(x1),
                y=int(y2),
                height=int(x2) - int(x1),
                width=int(y2) - int(y1)
            )
            image_entity = core.image_entity.ImageEntity(
                data=im,
                camera_pose=tf.Transform(),
                metadata=imeta.ImageMetadata(
                    source_type=imeta.ImageSourceType.REAL_WORLD,
                    environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                    light_level=imeta.LightingLevel.EVENLY_LIT,
                    time_of_day=imeta.TimeOfDay.DAY,
                    height=im.shape[0],
                    width=im.shape[1],
                    fov=40,
                    focal_length=-1,
                    aperture=-1,
                    label_classes=[label],
                    label_bounding_boxes={bounding_box}
                ),
                additional_metadata=None
            )
            s_image = image_entity.serialize()
            query = db_help.query_to_dot_notation(copy.deepcopy(s_image))
            existing = db_client.image_collection.find_one(query, {'_id': True})
            if existing is None:
                image_entity.save_image_data(db_client)
                im_id = db_client.image_collection.insert(s_image)
            else:
                im_id = existing['_id']
            image_ids.append(im_id)
    s_collection = core.image_collection.ImageCollection.create_serialized(image_ids=image_ids, sequence_type=core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
    query = db_help.query_to_dot_notation(copy.deepcopy(s_collection))
    if db_client.image_source_collection.find_one(query, {'_id': True}) is None:
        return db_client.image_source_collection.insert(s_collection)


def import_synth_dataset(labels_path, db_client, height_and_width=False):
    image_ids = []
    pose_x_re = re.compile("'x': (\d+)")
    pose_y_re = re.compile("'y': (\d+)")
    pose_z_re = re.compile("'z': (\d+)")
    pose_roll_re = re.compile("'roll': (\d+)")
    pose_pitch_re = re.compile("'pitch': (\d+)")
    pose_yaw_re = re.compile("'yaw': (\d+)")
    with open(labels_path, 'r') as labels_file:
        base_dir = os.path.dirname(labels_path)
        pose_path = os.path.join(base_dir, 'poses.txt')
        with open(pose_path, 'r') as pose_file:
            object_pose_desc = pose_file.readline()

            #TODO: Parse the object pose for more ground truth

            for line in labels_file:
                split = re.split('[, ]', line)
                if len(split) != 6:
                    continue
                imfile, x1, y1, x2, y2, label = split
                label = label.rstrip()
                im = cv2.imread(os.path.join(base_dir, imfile))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                pose_desc = pose_file.readline()
                ue_trans = uetf.UnrealTransform(
                    location=(
                        float(pose_x_re.search(pose_desc).group(1)),
                        float(pose_y_re.search(pose_desc).group(1)),
                        float(pose_z_re.search(pose_desc).group(1))
                    ),
                    rotation=(
                        float(pose_roll_re.search(pose_desc).group(1)),
                        float(pose_pitch_re.search(pose_desc).group(1)),
                        float(pose_yaw_re.search(pose_desc).group(1))
                    )
                )
                bounding_box = imeta.BoundingBox(
                    class_name=label,
                    confidence=1,
                    x=int(x1),
                    y=int(y2),
                    height=int(x2) if height_and_width else int(x2) - int(x1),
                    width=int(y2) if height_and_width else int(y2) - int(y1)
                )
                image_entity = core.image_entity.ImageEntity(
                    data=im,
                    camera_pose=uetf.transform_from_unreal(ue_trans),
                    metadata=imeta.ImageMetadata(
                        source_type=imeta.ImageSourceType.SYNTHETIC,
                        environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                        light_level=imeta.LightingLevel.EVENLY_LIT,
                        time_of_day=imeta.TimeOfDay.DAY,
                        height=im.shape[1],
                        width=im.shape[0],
                        fov=40,
                        focal_length=-1,
                        aperture=-1,
                        label_classes=label,
                        label_bounding_boxes={bounding_box},
                        distances_to_labelled_objects=None,
                        average_scene_depth=None
                    ),
                    additional_metadata=None
                )
                s_image = image_entity.serialize()
                query = db_help.query_to_dot_notation(copy.deepcopy(s_image))
                existing = db_client.image_collection.find_one(query, {'_id': True})
                if existing is None:
                    image_entity.save_image_data(db_client)
                    im_id = db_client.image_collection.insert(s_image)
                else:
                    im_id = existing['_id']
                image_ids.append(im_id)
    s_collection = core.image_collection.ImageCollection.create_serialized(image_ids=image_ids,
                                                                           sequence_type=core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
    query = db_help.query_to_dot_notation(copy.deepcopy(s_collection))
    if db_client.image_source_collection.find_one(query, {'_id': True}) is not None:
        # return db_client.image_source_collection.insert(s_object)
        pass
