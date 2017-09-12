#Copyright (c) 2017, John Skinner
import sys
import bson.objectid
import numpy as np
import cv2
import config.global_configuration as global_conf
import database.client
import core.image_collection
import core.image_entity


def main(*args):
    """
    Patch the database
    :return:
    """
    for image_source_id in args:
        config = global_conf.load_global_config('config.yml')
        db_client = database.client.DatabaseClient(config=config)

        s_image_source = db_client.image_source_collection.find_one({'_id': bson.ObjectId(image_source_id)})
        for image_id in s_image_source['images']:
            s_image = db_client.image_collection.find_one({'_id': image_id})
            image = db_client.deserialize_entity(s_image)

            if image.labels_data is not None:
                for idx, labelled_object in enumerate(image.metadata.labelled_objects):
                    color = labelled_object.label_color
                    label_points = cv2.findNonZero(np.asarray(np.all(image.labels_data == color, axis=2), dtype='uint8'))
                    db_client.image_collection.update({'_id': image_id}, {
                        '$set': {
                            'metadata.labelled_objects.{}.bounding_box'.format(idx): cv2.boundingRect(label_points)
                        }
                    })


if __name__ == '__main__':
    main(*sys.argv[1:])
