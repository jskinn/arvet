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

            labelled_object = imeta.LabelledObject(
                class_names=(label.lower(),),
                bounding_box=(int(x1), int(y2), int(x2) - int(x1), int(y2) - int(y1)),
                object_id='starbucks-cup-001'           # This is so I can refer to it later.
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
                    labelled_objects={labelled_object}
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
