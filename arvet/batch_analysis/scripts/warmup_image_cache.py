import logging
import logging.config
import argparse
import bson

import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.core.image_collection
import arvet.batch_analysis.task_manager
import arvet.batch_analysis.job_systems.job_system_factory as job_system_factory


def warmup_cache(image_collection_ids, task_ids):
    image_collection_ids = [bson.ObjectId(oid) for oid in image_collection_ids]
    task_ids = [bson.ObjectId(oid) for oid in task_ids]

    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])
    db_client = arvet.database.client.DatabaseClient(config=config)

    # First, warm up the cache for all the requested image sources
    for s_image_collection in db_client.image_source_collection.find({'_id': {'$in': image_collection_ids}}):
        image_collection = db_client.deserialize_entity(s_image_collection)
        if isinstance(image_collection, arvet.core.image_collection.ImageCollection):
            image_collection.warmup_cache()

    # Then, schedule all the tasks that were waiting for this
    task_manager = arvet.batch_analysis.task_manager.TaskManager(db_client.tasks_collection, db_client, config)
    job_system = job_system_factory.create_job_system(config=config)
    task_manager.schedule_dependent_tasks(task_ids, job_system)

    # Actually run the queued jobs.
    job_system.run_queued_jobs()


def main():
    """
    Run the script to warm up
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Update and schedule tasks from experiments.'
                    'By default, this will update and schedule tasks for all experiments.')
    parser.add_argument('--image_collection', action='append',
                        help='A image collection id to preload')
    parser.add_argument('task_ids', nargs='*', default=[],
                        help='Limit the update to only the specified experiment by id. '
                             'You may specify any number of ids.')

    args = parser.parse_args()
    warmup_cache(args.image_collection, args.task_ids)


if __name__ == '__main__':
    main()
