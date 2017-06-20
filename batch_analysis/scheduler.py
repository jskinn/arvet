import util.dict_utils as du
import database.entity_registry


TRAIN_SYSTEM_SCRIPT = 'task_train_system.py'
RUN_SYSTEM_SCRIPT = 'task_run_system.py'
BENCHMARK_RESULT_SCRIPT = 'task'


def schedule_training(db_client, job_system):
    # find all system trainers
    system_trainer_ids = db_client.system_trainers_collection.find({}, {'_id': True, '_type': True})
    for system_trainer_info in system_trainer_ids:
        system_trainer_type = database.entity_registry.get_entity_type(system_trainer_info['_type'])
        # Find all the image sources already used to train systems
        done_image_source_ids = db_client.systems_collection.find({
            'vision_system_trainer': system_trainer_info['_id']
        }, {'id_': False, 'training_image_sources': True})

        done_image_source_ids = [result['id_'] for result in done_image_source_ids]

        # find all appropriate image sources we haven't done trials with yet
        image_source_ids = db_client.image_source_collection.find({'id_': {'$nin': done_image_source_ids}},
                                                                  {'id_': True})
        for image_source_id in image_source_ids:
            job_system.run_script(RUN_SYSTEM_SCRIPT, str(system_id), str(image_source_id))


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
            job_system.run_script(RUN_SYSTEM_SCRIPT, str(system_id), str(image_source_id))


def schedule_benchmarks(db_client, job_system):
    pass


def schedule_trial_comparisons(db_client, job_system):
    pass


def schedule_benchmark_comparisons(db_client, job_system):
    pass
