import os.path
import pickle
import glob
import batch_analysis.experiment
import util.database_helpers as dh
import dataset.pod_cup.import_podcup_dataset as pod_dataset
import systems.deep_learning.keras_frcnn as sys_frcnn
import benchmarks.bounding_box_overlap.bounding_box_overlap as bench_bbox_overlap


class PodCupExperiment(batch_analysis.experiment.Experiment):
    """
    My experiment for testing the detection of cups in pods.
    """

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
        model_dir = os.path.expanduser('~/keras-models')
        for config_pickle_path in glob.iglob(os.path.join(model_dir, '*.pickle')):
            model_hdf5_path = os.path.splitext(config_pickle_path)[0] + '.hdf5'
            if os.path.isfile(model_hdf5_path):
                with open(config_pickle_path, 'rb') as config_file:
                    frcnn_config = pickle.load(config_file)
                frcnn_config.model_path = model_hdf5_path  # Update the path to the model file
                systems.add(dh.add_unique(db_client.system_collection, sys_frcnn.KerasFRCNN(frcnn_config)))
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
