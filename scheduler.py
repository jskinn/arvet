import config.global_configuration as global_conf
import database.client
import util.database_helpers as dh


def main():
    """
    Schedule tasks for all experiments.
    We need to find a way of running this repeatedly as a daemon
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)
    job_system = None   # TODO: Make a working job system

    experiment_ids = db_client.experiments_collection.find({}, {'_id': True})
    for experiment_id in experiment_ids:
        experiment = dh.load_object(db_client, db_client.experiments_collection, experiment_id['_id'])
        if experiment is not None:
            # TODO: Separate scripts for scheduling and importing, they run at different rates
            experiment.do_imports(db_client, save_changes=False)
            experiment.schedule_tasks(job_system, db_client)

if __name__ == '__main__':
    main()
