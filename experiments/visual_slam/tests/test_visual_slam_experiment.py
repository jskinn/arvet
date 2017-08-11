import unittest
import unittest.mock as mock
import bson.objectid as oid
import pymongo.collection
import util.dict_utils as du
import database.client
import database.tests.test_entity as entity_test
import batch_analysis.experiment as ex
import experiments.visual_slam.visual_slam_experiment as vse


class TestVisualSlamExperiment(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return vse.VisualSlamExperiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'trainers': {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()},
            'trainees': {oid.ObjectId()},
            'image_sources': {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()},
            'systems': {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()},
            'benchmarks': {oid.ObjectId(), oid.ObjectId()},
            'trial_results': {oid.ObjectId()},
            'benchmark_results': {oid.ObjectId()},
            'dataset_map': {
                '/home/user/Renders/Visual Realism/Experiment 1/Sim/Trajectory 1/metadata-json': oid.ObjectId()
            },
            'uncategorized_datasets': {oid.ObjectId(), oid.ObjectId()}
        })
        if 'training_map' not in kwargs:
            states = [0, 0, 1, 0, 2, 0, 1, 0, 1]
            kwargs['training_map'] = {
                trainer_id: {trainee_id: ex.ProgressState(states[(7 * idx) % len(states)])
                             for idx, trainee_id in enumerate(kwargs['image_sources'])}
                for trainer_id in kwargs['systems']
            }
        if 'trial_map' not in kwargs:
            states = [0, 0, 2, 1, 0, 1, 1, 0, 0]
            kwargs['trial_map'] = {
                sys_id: {source_id: ex.ProgressState(states[(11 * idx) % len(states)])
                         for idx, source_id in enumerate(kwargs['image_sources'])}
                for sys_id in kwargs['systems']
            }
        if 'benchmark_map' not in kwargs:
            states = [2, 0, 1, 0, 0, 0, 2, 1, 1]
            kwargs['benchmark_map'] = {
                trial_id: {bench_id: ex.ProgressState(states[(13 * idx) % len(states)])
                           for idx, bench_id in enumerate(kwargs['benchmarks'])}
                for trial_id in kwargs['trial_results']
            }
        return vse.VisualSlamExperiment(*args, **kwargs)

    def assert_models_equal(self, experiment1, experiment2):
        """
        Helper to assert that two visual slam experiments are equal
        :param experiment1: First experiment to compare
        :param experiment2: Second experiment to compare
        :return:
        """
        if (not isinstance(experiment1, vse.VisualSlamExperiment) or
                not isinstance(experiment2, vse.VisualSlamExperiment)):
            self.fail('object was not a VisualSlamExperiment')
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._trainers, experiment2._trainers)
        self.assertEqual(experiment1._trainees, experiment2._trainees)
        self.assertEqual(experiment1._image_sources, experiment2._image_sources)
        self.assertEqual(experiment1._systems, experiment2._systems)
        self.assertEqual(experiment1._benchmarks, experiment2._benchmarks)
        self.assertEqual(experiment1._trial_results, experiment2._trial_results)
        self.assertEqual(experiment1._trial_map, experiment2._trial_map)
        self.assertEqual(experiment1._benchmark_map, experiment2._benchmark_map)
        self.assertEqual(experiment1._dataset_map, experiment2._dataset_map)
        self.assertEqual(experiment1._uncategorized_datasets, experiment2._uncategorized_datasets)

    def assert_serialized_equal(self, s_model1, s_model2):
        # Check that either both or neither have an id
        if '_id' in s_model1 and '_id' in s_model2:
            self.assertEqual(s_model1['_id'], s_model2['_id'])
        else:
            self.assertNotIn('_id', s_model1)
            self.assertNotIn('_id', s_model2)
        for key in {'training_map', 'trial_map', 'benchmark_map', 'dataset_map'}:
            self.assertEqual(s_model1[key], s_model2[key], "Values for {0} were not equal".format(key))
        # Special handling for set keys, since the order is allowed to change.
        for key in {'trainers', 'trainees', 'image_sources', 'systems', 'trial_results',
                    'benchmarks', 'benchmark_results', 'uncategorized_datasets'}:
            self.assertEqual(set(s_model1[key]), set(s_model2[key]), "Values for {0} were not equal".format(key))

    def test_constructor_works_with_minimal_arguments(self):
        vse.VisualSlamExperiment()

    def test_add_image_source_doesnt_immediately_create_trials(self):
        image_source_id = oid.ObjectId()
        system_id = oid.ObjectId()
        subject = vse.VisualSlamExperiment(systems={system_id}, trial_map={system_id: {}})
        subject.add_image_source(image_source_id, '/tmp/dataset1.json')
        self.assertEqual({system_id: {}}, subject._trial_map)

    def test_add_image_source_adds_to_uncategorized_datasets_and_dataset_map(self):
        image_source_id = oid.ObjectId()
        subject = vse.VisualSlamExperiment()
        subject.add_image_source(image_source_id, '/tmp/dataset1.json')
        self.assertEqual({image_source_id}, subject._uncategorized_datasets)
        self.assertEqual({'/tmp/dataset1-json': image_source_id}, subject._dataset_map)

    def test_add_image_source_doesnt_reset_existing_trials(self):
        image_source_id = oid.ObjectId()
        system_id = oid.ObjectId()
        subject = vse.VisualSlamExperiment(systems={system_id}, image_sources={image_source_id}, trial_map={
            system_id: {image_source_id: ex.ProgressState.FINISHED}
        })
        subject.add_image_source(image_source_id, '/tmp/dataset1.json')
        self.assertEqual(ex.ProgressState.FINISHED, subject._trial_map[system_id][image_source_id])

    def test_add_image_source_saves_changes_if_given_db_client(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        image_source_id = oid.ObjectId()
        system_id = oid.ObjectId()
        subject = vse.VisualSlamExperiment(systems={system_id}, trial_map={system_id: {}}, id_=oid.ObjectId())

        subject.add_image_source(image_source_id, '/tmp/dataset1.json', mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
            '_id': subject.identifier
        }, {
            '$set': {
                "dataset_map./tmp/dataset1-json": image_source_id
            },
            '$addToSet': {'uncategorized_datasets': {'$each': [image_source_id]}}
        }), mock_db_client.experiments_collection.update.call_args)
