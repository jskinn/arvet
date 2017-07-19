import os.path
import re
import cv2

import util.transform as tf
import metadata.image_metadata as imeta
import core.image_entity
import core.image_collection
import core.sequence_type
import dataset.image_collection_builder


def import_rw_dataset(labels_path, db_client):
    builder = dataset.image_collection_builder.ImageCollectionBuilder(db_client)
    builder.set_non_sequential()
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
                bounding_box=(int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)),
                object_id='StarbucksCup_170'           # This is so I can refer to it later, matches Unreal name
            )
            image_entity = core.image_entity.ImageEntity(
                data=im,
                camera_pose=tf.Transform(),
                metadata=imeta.ImageMetadata(
                    source_type=imeta.ImageSourceType.REAL_WORLD,
                    height=im.shape[0],
                    width=im.shape[1],
                    environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                    light_level=imeta.LightingLevel.EVENLY_LIT,
                    labelled_objects=(labelled_object,)),
                additional_metadata=None
            )
            builder.add_image(image_entity)
    return builder.save()
