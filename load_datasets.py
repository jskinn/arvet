import config.global_configuration as global_conf
import database.client

import dataset.pod_cup.import_podcup_dataset


def main():
    """
    Import existing datasets into the database
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    dataset.pod_cup.import_podcup_dataset.import_rw_dataset('/home/john/datasets/Custom_Cup/clicks-1497585183.txt', db_client)
    #dataset.pod_cup.import_podcup_dataset.import_synth_dataset('/home/john/Renders/Object_rec_test_1/labels.txt', db_client)


if __name__ == '__main__':
    main()
