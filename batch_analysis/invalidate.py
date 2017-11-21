import logging
import bson
import database.client


def invalidate_image_source(db_client: database.client.DatabaseClient, image_source_id: bson.ObjectId):
    """
    Invalidate the data associated with a particular image source.
    This cascades to derived trial results, and from there to benchmark results.
    Also cleans out tasks.
    :param db_client:
    :param image_source_id:
    :return:
    """
    # Step 1: Find all the tasks that involve this image source, and remove them
    result = db_client.tasks_collection.remove({'$or': [{'image_source_id': image_source_id},
                                                        {'result_id': image_source_id}]})
    logging.getLogger(__name__).info("removed {0} tasks".format(result['n'] if 'n' in result else 0))

    # Step 2: Find all the trial results that use this image source, and invalidate them
    # Except we don't have a good way of matching trials to particular image source, so for now we're going
    # to get them all
    trials = db_client.trials_collection.find({}, {'_id': True})
    for s_image_source in trials:
        invalidate_trial_result(db_client, s_image_source['_id'])

    # Step 3: Remove the image result
    result = db_client.image_source_collection.remove({'_id': image_source_id})
    logging.getLogger(__name__).info("removed {0} image sources".format(result['n'] if 'n' in result else 0))


def invalidate_system(db_client: database.client.DatabaseClient, system_id: bson.ObjectId):
    # Step 1: Find all the tasks that involve this system, and remove them
    result = db_client.tasks_collection.remove({'system_id': system_id})
    logging.getLogger(__name__).info("removed {0} tasks".format(result['n'] if 'n' in result else 0))

    # Step 2: Find all the trial results that use this image source, and invalidate them
    trials = db_client.trials_collection.find({'system': system_id}, {'_id': True})
    for s_trial_result in trials:
        invalidate_trial_result(db_client, s_trial_result['_id'])

    # Step 3: Actually remove the system
    result = db_client.system_collection.remove({'_id': system_id})
    logging.getLogger(__name__).info("removed {0} systems".format(result['n'] if 'n' in result else 0))


def invalidate_trial_result(db_client: database.client.DatabaseClient, trial_result_id: bson.ObjectId):
    # Step 1: Find all the tasks that involve this trial result, and remove them
    result = db_client.tasks_collection.remove({'$or': [{'result': trial_result_id},
                                                        {'trial_result_id': trial_result_id},
                                                        {'trial_result1_id': trial_result_id},
                                                        {'trial_result2_id': trial_result_id}]})
    logging.getLogger(__name__).info("removed {0} tasks".format(result['n'] if 'n' in result else 0))

    # Step 2: Find all the benchmark results that use this trial result, and invalidate them
    results = db_client.results_collection.find({'trial_result_id': trial_result_id}, {'_id': True})
    for s_result in results:
        invalidate_benchmark_result(db_client, s_result['_id'])

    # Step 3: actually remove the trial result
    result = db_client.trials_collection.remove({'_id': trial_result_id})
    logging.getLogger(__name__).info("removed {0} trials".format(result['n'] if 'n' in result else 0))


def invalidate_benchmark(db_client: database.client.DatabaseClient, benchmark_id: bson.ObjectId):
    # Step 1: Find all the tasks that involve this benchmark, and remove them
    result = db_client.tasks_collection.remove({'$or': [{'benchmark_id': benchmark_id},
                                                        {'comparison_id': benchmark_id},
                                                        {'benchmark_result1_id': benchmark_id},
                                                        {'benchmark_result2_id': benchmark_id}]})
    logging.getLogger(__name__).info("removed {0} tasks".format(result['n'] if 'n' in result else 0))

    # Step 2: Find all the benchmark results that use this benchmark, and invalidate them
    results = db_client.results_collection.find({'benchmark': benchmark_id}, {'_id': True})
    for s_result in results:
        invalidate_benchmark_result(db_client, s_result['_id'])

    # Step 3: actually remove the benchmark
    result = db_client.benchmarks_collection.remove({'_id': benchmark_id})
    logging.getLogger(__name__).info("removed {0} benchmarks".format(result['n'] if 'n' in result else 0))


def invalidate_benchmark_result(db_client: database.client.DatabaseClient, benchmark_result_id: bson.ObjectId):
    result = db_client.tasks_collection.remove({'result_id': benchmark_result_id})
    logging.getLogger(__name__).info("removed {0} tasks".format(result['n'] if 'n' in result else 0))
    result = db_client.results_collection.remove({'_id': benchmark_result_id})
    logging.getLogger(__name__).info("removed {0} results".format(result['n'] if 'n' in result else 0))
