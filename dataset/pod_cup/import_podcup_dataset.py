import os.path
import glob
import re
import xxhash
import cv2

import util.transform as tf
import metadata.image_metadata as imeta
import core.image_entity
import core.image_collection
import core.sequence_type
import dataset.image_collection_builder


def import_dataset(labels_path, db_client, **kwargs):
    """
    Import a real-world dataset with labelled images.
    :param labels_path:
    :param db_client:
    :param kwargs: Additional arguments passed to the image metadata
    :return:
    """
    if os.path.isdir(labels_path):
        # Look in the given folder for possible labels files
        candidates = glob.glob(os.path.join(labels_path, '*.txt'))
        if len(candidates) >= 1:
            labels_path = candidates[0]
        else:
            # Cannot find the labels file, return None
            return None
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
                metadata=imeta.ImageMetadata(
                    hash_=xxhash.xxh64(im).digest(),
                    source_type=imeta.ImageSourceType.REAL_WORLD,
                    height=im.shape[0],
                    width=im.shape[1],
                    camera_pose=tf.Transform(),
                    labelled_objects=(labelled_object,),
                    **kwargs),
                additional_metadata=None
            )
            builder.add_image(image_entity)
    return builder.save()
