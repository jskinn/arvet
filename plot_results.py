import database.client
import config.global_configuration as global_conf
import util.database_helpers as dh


def main(*args):
    config = global_conf.load_global_config("config.yml")
    db_client = database.client.DatabaseClient(config)
    experiment_ids = db_client.experiments_collection.find({}, {'_id': True})
    for ex_id in experiment_ids:
        experiment = dh.load_object(db_client, db_client.experiments_collection, ex_id['_id'])
        if experiment is not None:
            experiment.plot_results(db_client)


if __name__ == "__main__":
    main()
