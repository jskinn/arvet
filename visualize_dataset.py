import numpy as np
import bson
import config.global_configuration as global_conf
import database.client
import cv2


def main():
    """
    Visualize a random generated dataset, to make sure we're generating them right
    :return:
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    generate_tasks = db_client.tasks_collection.find({
        '_type': 'batch_analysis.tasks.generate_dataset_task.GenerateDatasetTask', 'state': 2}).limit(2)
    for s_generate_task in generate_tasks:
        task = db_client.deserialize_entity(s_generate_task)
        result_ids = task.result
        if result_ids is None:
            continue
        elif isinstance(result_ids, bson.ObjectId):
            result_ids = [result_ids]
        for result_id in result_ids:
            s_image_source = db_client.image_source_collection.find_one({'_id': result_id})
            if s_image_source is not None:
                image_source = db_client.deserialize_entity(s_image_source)
                with image_source:
                    while not image_source.is_complete():
                        image, _ = image_source.get_next_image()
                        cv2.imshow('rgb', image.data[:, :, ::-1])
                        cv2.imshow('depth', image.depth_data)
                        cv2.waitKey(1000)


if __name__ == '__main__':
    main()
