import os.path
import glob
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import bson.objectid
import batch_analysis.experiment
import util.database_helpers as dh
import dataset.pod_cup.import_podcup_dataset as pod_dataset
import training.epoch_trainer
import systems.deep_learning.keras_frcnn_trainee as train_frcnn
import benchmarks.bounding_box_overlap.bounding_box_overlap as bench_bbox_overlap
import benchmarks.bounding_box_overlap.bounding_box_overlap_result as overlap_result


class PodCupExperiment(batch_analysis.experiment.Experiment):
    """
    My experiment for testing the detection of cups in pods.
    """
    def __init__(self, training_data_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_data_names = training_data_names if training_data_names is not None else {}

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
                dataset_id = pod_dataset.import_rw_dataset(clicks_file, db_client)
                self._training_data_names[dataset_id] = subdir
                training_datasets.append(dataset_id)

        trainers = set()
        for dataset in training_datasets:
            s_trainer = training.epoch_trainer.EpochTrainer.create_serialized(
                num_epochs=50,
                use_source_length=False,
                epoch_length=1000,
                image_sources=(dataset,),
                horizontal_flips=True,
                vertical_flips=True,
                rot_90=True,
                validation_fraction=0.2
            )
            trainers.add(dh.add_unique(db_client.trainer_collection, s_trainer))
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        self._updates['$set']['training_data_names'] = {str(oid): name for oid, name
                                                        in self._training_data_names.items()}
        return trainers

    def import_trainees(self, db_client):
        """
        Make the FRCNN trainee that will produce the trained systems
        :param db_client: The database client
        :return: The set of ids of new trainees. May include existing ids.
        """
        base_weights_path = os.path.expanduser('~/Documents/TensorflowModels/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        frcnn_trainee = train_frcnn.KerasFRCNNTrainee(
            weights_folder=os.path.expanduser('~/keras-models/podcup'),
            classes={'cup'},
            num_rois=64,
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




        for result_id in self.results_ids:
            result = dh.load_object(db_client, db_client.results_collection, result_id)
            if result is not None:
                precision, recall, f1score, log_gt_area, iou = result.list_results(
                    overlap_result.precision,
                    overlap_result.recall,
                    overlap_result.f1_score,
                    lambda x: np.log(x['ground_truth_area']) if x['ground_truth_area'] > 0 else 0,
                    lambda x: x['overlap'] / (x['bounding_box_area'] + x['ground_truth_area'] - x['overlap'])
                )

        figure = pyplot.figure(figsize=(14, 10), dpi=80)
        ax_pr = figure.add_subplot(111)
        ax_pr.set_xlabel('precision')
        ax_pr.set_ylabel('recall')
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)

        figure = pyplot.figure(figsize=(14, 10), dpi=80)
        ax_p_area = figure.add_subplot(121)
        ax_p_area.set_xlabel('bounding box area')
        ax_p_area.set_ylabel('precision')

        ax_r_area = figure.add_subplot(122)
        ax_r_area.set_xlabel('bounding box area')
        ax_r_area.set_ylabel('recall')
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)

        fp_values = []
        fn_values = []
        false_labels = []

        boxplot_stuff = []
        boxplot_labels = []
        for result_id in self.results_ids:
            s_result = db_client.results_collection.find_one({'_id': result_id})
            result = db_client.deserialize_entity(s_result)
            name = self.get_name(result, db_client)
            precision, recall, f1score, gt_area, iou = result.list_results(
                overlap_result.precision, overlap_result.recall, overlap_result.f1_score,
                lambda x: np.log(x['ground_truth_area']) if x['ground_truth_area'] > 0 else 0,
                lambda x: x['overlap'] / (x['bounding_box_area'] + x['ground_truth_area'] - x['overlap']))

            boxplot_stuff.append(iou)
            boxplot_labels.append(name)

            ax_pr.scatter(precision, recall, label=name)
            x, y = prune_zero_points(gt_area, precision)
            ax_p_area.scatter(x, y, label=name)
            x, y = prune_zero_points(gt_area, recall)
            ax_r_area.scatter(x, y, label=name)

            # Count false positives
            false_labels.append(name)
            false_positives = 0
            false_negatives = 0
            for bbox_results in result.overlaps.values():
                for bbox_result in bbox_results:
                    if bbox_result['ground_truth_area'] == 0:
                        false_positives += 1
                    if bbox_result['bounding_box_area'] == 0:
                        false_negatives += 1
            fp_values.append(false_positives)
            fn_values.append(false_negatives)

        ax_pr.legend()
        ax_p_area.legend()

        # False positives
        x = np.arange(len(fp_values))
        figure = pyplot.figure(figsize=(14, 10), dpi=80)
        ax_fp = figure.add_subplot(121)
        ax_fp.set_ylabel('false positives')
        ax_fp.set_xticklabels(false_labels)
        ax_fp.bar(x, fp_values, align='center')
        ax_fn = figure.add_subplot(122)
        ax_fn.set_ylabel('false negatives')
        ax_fn.set_xticklabels(false_labels)
        ax_fn.bar(x, fn_values, align='center')
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)

        figure = pyplot.figure(figsize=(14, 10), dpi=80)
        ax = figure.add_subplot(111)
        ax.set_xlabel('precision')
        ax.set_ylabel('Intersection over Union')
        ax.boxplot(boxplot_stuff, positions=list(range(len(self.results_ids))))
        ax.set_xticks(list(range(len(self.results_ids))))
        ax.set_xticklabels(boxplot_labels)

        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()

    def serialize(self):
        serialized = super().serialize()
        serialized['training_data_names'] = {str(oid): name for oid, name in self._training_data_names.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'training_data_names' in serialized_representation:
            kwargs['training_data_names'] = {bson.objectid.ObjectId(oid): name
                                             for oid, name in serialized_representation['training_data_names'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)


    def get_name(self, bbox_result, db_client):
        s_trial_result = db_client.trials_collection.find_one({'_id': bbox_result.trial_result}, {'system': True})
        if s_trial_result is not None:
            s_system = db_client.system_collection.find_one({'_id': s_trial_result['system']},
                                                             {'training_image_sources': True})
            if s_system is not None:
                for image_source_id in s_system['training_image_sources']:
                    if image_source_id in self._training_data_names:
                        return self._training_data_names[image_source_id]
        return 'Unknown system'


def fix_line_order(x, y):
    points = list(zip(list(x), list(y)))
    points.sort()
    return [point[0] for point in points], [point[1] for point in points]


def prune_zero_points(x, y, prune_x=True, prune_y=True, sort=False):
    points = list(zip(x, y))
    points = [(x_val, y_val) for x_val, y_val in points
              if (not prune_x or x_val != 0) and (not prune_y or y_val != 0)]
    if sort:
        points.sort()
    return [point[0] for point in points], [point[1] for point in points]
