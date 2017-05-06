import util.dict_utils as du



def schedule_training(db_client, job_system):
    pass


def schedule_system_trials(db_client, job_system):
    # find all systems
    system_ids = db_client.system_collection.find({}, {'id_': True})
    for system_id in system_ids:
        # Find all image sources we've already run this system with
        done_image_source_ids = db_client.trials_collection.find({'system_id': system_id}, {'id_': True}).distinct('id_')
        done_image_source_ids = [result['id_'] for result in done_image_source_ids]

        # find all appropriate image sources we haven't done trials with yet
        image_source_ids = db_client.image_source_collection.find({'id_' : {'$nin': done_image_source_ids}}, {'id_':True})
        for image_source_id in image_source_ids:
            job_system.run_script('task_run_system.py', str(system_id), str(image_source_id))


def schedule_benchmarks(db_client, job_system):
    pass


def schedule_trial_comparisons(db_client, job_system):
    pass


def schedule_benchmark_comparisons(db_client, job_system):
    pass
