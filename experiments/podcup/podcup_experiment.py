import os.path
import pickle
import glob
import batch_analysis.experiment
import util.database_helpers as dh
import dataset.pod_cup.import_podcup_dataset as pod_dataset
import training.epoch_trainer
import systems.deep_learning.keras_frcnn as sys_frcnn
import systems.deep_learning.keras_frcnn_trainee as train_frcnn
import benchmarks.bounding_box_overlap.bounding_box_overlap as bench_bbox_overlap


class PodCupExperiment(batch_analysis.experiment.Experiment):
    """
    My experiment for testing the detection of cups in pods.
    """
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)

    def import_trainers(self, db_client):
        """
        Import trainers to train new systems for the experiment.
        This experiment trains FRCNN on many different datasets

        :param db_client: The database client
        :return: A set of database ids for system trainers
        """
        training_datasets = []
        root_dir = os.path.expanduser('~/datasets/pod_cup')
        subdirs = {'cup_in_pod', 'cup_outside_pod', 'other_cups_indoor', 'other_cups_outdoor', 'other_cups_pod'}
        for subdir in subdirs:
            for clicks_file in glob.iglob(os.path.join(root_dir, subdir, 'clicks-*.txt')):  # there should only be one
                training_datasets.append(pod_dataset.import_rw_dataset(clicks_file, db_client))

        trainers = set()
        for dataset in training_datasets:
            s_trainer = training.epoch_trainer.EpochTrainer.create_serialized(
                num_epochs=1,
                use_source_length=False,
                epoch_length=50,
                image_sources=(dataset,),
                horizontal_flips=True,
                vertical_flips=True,
                rot_90=True,
                validation_fraction=0.2
            )
            trainers.add(dh.add_unique(db_client.trainer_collection, s_trainer))
        return trainers

    def import_trainees(self, db_client):
        """
        Make the FRCNN trainee that will produce the trained systems
        :param db_client: The database client
        :return: The set of ids of new trainees. May include existing ids.
        """
        models_folder = os.path.expanduser('~/keras-models/podcup')
        base_weights_path = os.path.expanduser('~/Documents/TensorflowModels/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        frcnn_trainee = train_frcnn.KerasFRCNNTrainee(
            weights_file=train_frcnn.generate_filename(models_folder),
            classes={'cup'},
            num_rois=32,
            input_weight_path=base_weights_path,
            use_training_loss=False
        )
        trainee_id = dh.add_unique(db_client.trainee_collection, frcnn_trainee)
        return {trainee_id}

    def import_image_sources(self, db_client):
        """
        Import the cup in pod dataset used for testing in this experiment
        :param db_client: The database client, used to do the importing
        :return: A collection of the imported image source ids. May include existing ids.
        """
        image_sources = set()
        image_sources.add(pod_dataset.import_rw_dataset('/home/john/datasets/pod_cup/cup_in_pod/clicks-1497585183.txt',
                                                        db_client))
        return image_sources

    def import_systems(self, db_client):
        """
        Add keras frcnns from a folder of pre-trained models.
        :param db_client: The database client used to do the importing
        :return: A collection of the database ids of the imported image sources
        """
        systems = set()
        #model_dir = os.path.expanduser('~/keras-models')
        #for config_pickle_path in glob.iglob(os.path.join(model_dir, '*.pickle')):
        #    model_hdf5_path = os.path.splitext(config_pickle_path)[0] + '.hdf5'
        #    if os.path.isfile(model_hdf5_path):
        #        with open(config_pickle_path, 'rb') as config_file:
        #            frcnn_config = pickle.load(config_file)
        #        frcnn_config.model_path = model_hdf5_path  # Update the path to the model file
        #        systems.add(dh.add_unique(db_client.system_collection, sys_frcnn.KerasFRCNN(frcnn_config)))
        return systems

    def import_benchmarks(self, db_client):
        """
        Create and store the benchmarks for bounding boxes.
        Just using the default settings for now
        :param db_client:
        :return: The
        """
        c = db_client.benchmarks_collection
        benchmarks = set()
        benchmarks.add(dh.add_unique(c, bench_bbox_overlap.BoundingBoxOverlapBenchmark()))
        return benchmarks

    def plot_results(self, db_client):
        #TODO: Sort out pandas, and then move the contents of plot_results here
        pass
