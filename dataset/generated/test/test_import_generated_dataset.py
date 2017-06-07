import unittest
import numpy as np
import cv2
import os.path
import json
import pickle
import pymongo.collection
import gridfs
import bson.objectid
import transforms3d as tf3d
import unittest.mock as mock
import util.transform
import util.dict_utils as du
import database.client
import dataset.generated.import_generated_dataset as import_gen


class TestImportGeneratedDataset(unittest.TestCase):

    def test_generate_filename(self):
        filename = import_gen.generate_image_filename(base_path='/home/user',
                                                      filename_format="Test_{name}.{frame}.{stereopass}",
                                                      mappings={'name': 'Juan'},
                                                      index_padding=6,
                                                      index=13,
                                                      extension='.png',
                                                      stereo_pass=15,
                                                      render_pass='WorldNormals')
        self.assertEqual(filename, '/home/user/Test_Juan.000013.15_WorldNormals.png')

    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_read_image_file_returns_none_if_file_doesnt_exist(self, mock_cv2):
        result = import_gen.read_image_file('notafile.error')
        self.assertIsNone(result)
        self.assertFalse(mock_cv2.imread.called)

    @mock.patch('dataset.generated.import_generated_dataset.os.path')
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_read_image_file_reorders_channels(self, mock_cv2, mock_path):
        mock_path.isfile.return_value = True
        mock_cv2.imread = mock.create_autospec(cv2.imread)
        mock_image = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        mock_cv2.imread.return_value = mock_image

        result = import_gen.read_image_file('check-bypassed.png')
        self.assertIsNotNone(result)
        self.assertNPEqual(mock_image[:, :, 2], result[:, :, 0])
        self.assertNPEqual(mock_image[:, :, 1], result[:, :, 1])
        self.assertNPEqual(mock_image[:, :, 0], result[:, :, 2])

    def test_parse_transform_builds_transform(self):
        transform = import_gen.parse_transform({'X': 113, 'Y': -127, 'Z': 79},
                                               {'W': 1, 'X': 0, 'Y': 0, 'Z': 0})
        self.assertIsInstance(transform, util.transform.Transform)

    def test_parse_transform_corrects_coordinates(self):
        rot = tf3d.quaternions.axangle2quat((1, -2, 4), -np.pi / 6)
        transform = import_gen.parse_transform({'X': 113, 'Y': -127, 'Z': 79},
                                               {'W': rot[0], 'X': rot[1], 'Y': rot[2], 'Z': rot[3]})

        self.assertNPEqual((113, 127, 79), transform.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((1, 2, 4), np.pi / 6),
                           transform.rotation_quat(w_first=True),
                           approx=0.0000000000000001)

    def test_make_additional_metadata_combines_dicts(self):
        metadata = import_gen.make_additional_metadata({
            'Aye': 10,
            'Bee': '/test/util/geom',
            'Cee': 'python importlib.py --do-import --prepare --remodel',
            'Dee': 12.4
        }, {
            'Aye': 'Test',
            'Nay': util.transform.Transform(location=(100, 10, -30), rotation=(-4, 152, 15, -2)),
            'Abstain': 'Whoops'
        }, {
            'Abstain': 'NEVER!',
            'Cee': 18.02,
            'Foo': 842
        })
        self.assertEqual(metadata, {
            'Aye': 'Test',
            'Bee': '/test/util/geom',
            'Cee': 18.02,
            'Dee': 12.4,
            'Foo': 842,
            'Abstain': 'Whoops',
            'Nay': util.transform.Transform(location=(100, 10, -30), rotation=(-4, 152, 15, -2))
        })

    def test_make_additional_metadata_ignores_keys(self):
        metadata = import_gen.make_additional_metadata({
            'Version': '1.2.3.45.6',
            'A': 10
        }, {
            'Camera Location': util.transform.Transform(location=(100, 10, -30), rotation=(-4, 152, 15, -2)),
            'B': 11
        }, {
            'Camera Orientation': {'W': 12, 'X': 15, 'Y': -22, 'Z': 999},
            'C': 12
        })
        self.assertEqual(metadata, {
            'A': 10,
            'B': 11,
            'C': 12
        })

    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_image_object_returns_none_if_image_doesnt_exist(self, mock_isfile, *_):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        expected_filename = '/home/user/Test.000013.0.png'
        mock_isfile.side_effect = lambda fn: (fn != expected_filename)
        result = import_gen.import_image_object(db_client=mock_db_client,
                                                base_path='/home/user',
                                                filename_format="Test.{frame}.{stereopass}",
                                                mappings={},
                                                index_padding=6,
                                                index=13,
                                                extension='.png',
                                                timestamp=10,
                                                dataset_metadata={})
        self.assertIn(mock.call(expected_filename), mock_isfile.call_args_list)
        self.assertIsNone(result)

    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_image_object_returns_none_if_metadata_doesnt_exist(self, mock_isfile, *_):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        expected_filename = '/home/user/Test.000013.0.png.metadata.json'
        mock_isfile.side_effect = lambda fn: (fn != expected_filename)
        result = import_gen.import_image_object(db_client=mock_db_client,
                                                base_path='/home/user',
                                                filename_format="Test.{frame}.{stereopass}",
                                                mappings={},
                                                index_padding=6,
                                                index=13,
                                                extension='.png',
                                                timestamp=10,
                                                dataset_metadata={})
        self.assertIn(mock.call(expected_filename), mock_isfile.call_args_list)
        self.assertIsNone(result)

    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_import_image_object_returns_id_of_existing_image(self, mock_cv2, mock_isfile, mock_json_load):
        mock_data = self.prepare_mocks(mock_cv2=mock_cv2, mock_isfile=mock_isfile, mock_json_load=mock_json_load,
                                       image_metadata={
                                           'Camera Location': {'X': -22, 'Y': 13, 'Z': 4},
                                           'Camera Orientation': {'W': -0.5, 'X': 0.5, 'Y': -0.5, 'Z': -0.5}
                                       })
        mock_db_client = mock_data[0]

        existing_id = bson.objectid.ObjectId()
        mock_db_client.image_collection.find_one.return_value = {'_id': existing_id}
        result = import_gen.import_image_object(db_client=mock_db_client,
                                                base_path='/home/user',
                                                filename_format="Test.{frame}.{stereopass}",
                                                mappings={},
                                                index_padding=6,
                                                index=13,
                                                extension='.png',
                                                timestamp=10,
                                                dataset_metadata={'world': 'mock', 'quality': 'maximum'})
        self.assertEqual(existing_id, result)
        self.assertFalse(mock_db_client.grid_fs.put.called)
        self.assertFalse(mock_db_client.image_collection.insert.called)
        self.assertTrue(mock_db_client.image_collection.find_one.called)
        existing_query = mock_db_client.image_collection.find_one.call_args[0][0]
        self.assertIn('camera_pose.location', existing_query)
        self.assertNPEqual((-22, -13, 4), existing_query['camera_pose.location'])
        self.assertIn('camera_pose.rotation', existing_query)
        self.assertNPEqual((0.5, 0.5, 0.5, -0.5), existing_query['camera_pose.rotation'], approx=0.000000000000001)
        self.assertIn('additional_metadata.world', existing_query)
        self.assertEqual('mock', existing_query['additional_metadata.world'])
        self.assertIn('additional_metadata.quality', existing_query)
        self.assertEqual('maximum', existing_query['additional_metadata.quality'])

    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('core.image_entity.pickle', autospec=pickle)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_import_image_object_loads_image(self, mock_cv2, mock_isfile, mock_json_load, mock_pickle):
        mock_db_client, images, ids = self.prepare_mocks(mock_isfile=mock_isfile, mock_json_load=mock_json_load,
                                                         mock_cv2=mock_cv2, mock_pickle=mock_pickle)
        im_id, depth_id, labels_id, normals_id = ids
        im_data, depth_data, labels_data, normals_data = images

        import_gen.import_image_object(db_client=mock_db_client,
                                       base_path='/home/user',
                                       filename_format="Test.{frame}.{stereopass}",
                                       mappings={},
                                       index_padding=6,
                                       index=13,
                                       extension='.png',
                                       timestamp=10,
                                       dataset_metadata={'world': 'nope', 'quality': 'overpowered'})

        self.assertCalled(mock_db_client.grid_fs.put)
        found_im = False
        found_depth = False
        found_labels = False
        found_normals = False
        for call_args in mock_db_client.grid_fs.put.call_args_list:
            if np.array_equal(im_data, call_args[0][0]):
                found_im = True
            elif np.array_equal(depth_data, call_args[0][0]):
                found_depth = True
            elif np.array_equal(labels_data, call_args[0][0]):
                found_labels = True
            elif np.array_equal(normals_data, call_args[0][0]):
                found_normals = True
        self.assertTrue(found_im, "{0} was not called with the image data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_depth, "{0} was not called with the depth data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_labels, "{0} was not called with the labels data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_normals, "{0} was not called with the normals".format(str(mock_db_client.grid_fs.put)))

        self.assertCalled(mock_db_client.image_collection.insert)
        s_result_image = mock_db_client.image_collection.insert.call_args[0][0]  # First argument of first call
        self.assertEqual('ImageEntity', s_result_image['_type'])
        self.assertEqual(10, s_result_image['timestamp'])
        self.assertNPEqual((1, -1, 1), s_result_image['camera_pose']['location'])
        self.assertNPEqual((0.2, -0.4, 0.8, -0.4), s_result_image['camera_pose']['rotation'], approx=0.000000000000001)
        self.assertEqual({'world': 'nope', 'quality': 'overpowered'}, s_result_image['additional_metadata'])
        self.assertEqual(im_id, s_result_image['data'])
        self.assertEqual(depth_id, s_result_image['depth_data'])
        self.assertEqual(labels_id, s_result_image['labels_data'])
        self.assertEqual(normals_id, s_result_image['world_normals_data'])

    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('core.image_entity.pickle', autospec=pickle)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_import_image_object_loads_stereo_image(self, mock_cv2, mock_isfile, mock_json_load, mock_pickle):
        mock_db_client, images, ids = self.prepare_mocks(mock_isfile=mock_isfile, mock_json_load=mock_json_load,
                                                         mock_cv2=mock_cv2, mock_pickle=mock_pickle, make_stereo=True)
        im_id, depth_id, labels_id, normals_id, right_id, right_depth_id, right_labels_id, right_normals_id = ids
        (im_data, depth_data, labels_data, normals_data, right_im_data,
         right_depth_data, right_labels_data, right_normals_data) = images

        import_gen.import_image_object(db_client=mock_db_client,
                                       base_path='/home/user',
                                       filename_format="Test.{frame}.{stereopass}",
                                       mappings={},
                                       index_padding=6,
                                       index=13,
                                       extension='.png',
                                       timestamp=10,
                                       dataset_metadata={'world': 'nope', 'quality': 'overpowered'})

        self.assertCalled(mock_db_client.grid_fs.put)
        found_im = False
        found_depth = False
        found_labels = False
        found_normals = False
        found_right_im = False
        found_right_depth = False
        found_right_labels = False
        found_right_normals = False
        for call_args in mock_db_client.grid_fs.put.call_args_list:
            if np.array_equal(im_data, call_args[0][0]):
                found_im = True
            elif np.array_equal(depth_data, call_args[0][0]):
                found_depth = True
            elif np.array_equal(labels_data, call_args[0][0]):
                found_labels = True
            elif np.array_equal(normals_data, call_args[0][0]):
                found_normals = True
            elif np.array_equal(right_im_data, call_args[0][0]):
                found_right_im = True
            elif np.array_equal(right_depth_data, call_args[0][0]):
                found_right_depth = True
            elif np.array_equal(right_labels_data, call_args[0][0]):
                found_right_labels = True
            elif np.array_equal(right_normals_data, call_args[0][0]):
                found_right_normals = True
        self.assertTrue(found_im, "{0} was not called with the image data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_depth, "{0} was not called with the depth data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_labels, "{0} was not called with the labels data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_normals, "{0} was not called with the normals".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_right_im,
                        "{0} was not called with the right image data".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_right_depth,
                        "{0} was not called with the right depth".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_right_labels,
                        "{0} was not called with the right labels".format(str(mock_db_client.grid_fs.put)))
        self.assertTrue(found_right_normals,
                        "{0} was not called with the right normals".format(str(mock_db_client.grid_fs.put)))

        self.assertCalled(mock_db_client.image_collection.insert)
        s_result_image = mock_db_client.image_collection.insert.call_args[0][0]  # First argument of first call
        self.assertEqual('StereoImageEntity', s_result_image['_type'])
        self.assertEqual(10, s_result_image['timestamp'])
        self.assertNPEqual((1, -1, 1), s_result_image['left_camera_pose']['location'])
        self.assertNPEqual((0.2, -0.4, 0.8, -0.4), s_result_image['left_camera_pose']['rotation'],
                           approx=0.000000000000001)
        self.assertNPEqual((1, -1, 1), s_result_image['right_camera_pose']['location'])
        self.assertNPEqual((0.2, -0.4, 0.8, -0.4), s_result_image['right_camera_pose']['rotation'],
                           approx=0.000000000000001)
        self.assertEqual({'world': 'nope', 'quality': 'overpowered'}, s_result_image['additional_metadata'])
        self.assertEqual(im_id, s_result_image['left_data'])
        self.assertEqual(depth_id, s_result_image['left_depth_data'])
        self.assertEqual(labels_id, s_result_image['left_labels_data'])
        self.assertEqual(normals_id, s_result_image['left_world_normals_data'])
        self.assertEqual(right_id, s_result_image['right_data'])
        self.assertEqual(right_depth_id, s_result_image['right_depth_data'])
        self.assertEqual(right_labels_id, s_result_image['right_labels_data'])
        self.assertEqual(right_normals_id, s_result_image['right_world_normals_data'])

    @staticmethod
    def prepare_mocks(mock_cv2=None, mock_isfile=None, mock_json_load=None, mock_pickle=None,
                      image_metadata=None, make_stereo=False):
        """
        Set up mocks for tests.
        The mocks passed in will be modified, additional mocks will be returned
        :param mock_cv2:
        :param mock_isfile:
        :param mock_json_load:
        :param mock_pickle:
        :return:
        """
        # Create example data, and map it between expected filenames and the data
        im_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        depth_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        labels_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        normals_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        files = {
            '/home/user/Test.000013.0.png.metadata.json': True,
            '/home/user/Test.000013.0.png': im_data,
            '/home/user/Test.000013.0_SceneDepthWorldUnits.png': depth_data,
            '/home/user/Test.000013.0_ObjectMask.png': labels_data,
            '/home/user/Test.000013.0_WorldNormals.png': normals_data,
        }
        right_im_data = right_depth_data = right_labels_data = right_normals_data = None
        if make_stereo:
            right_im_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
            right_depth_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
            right_labels_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
            right_normals_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
            files['/home/user/Test.000013.1.png.metadata.json'] = True
            files['/home/user/Test.000013.1.png'] = right_im_data
            files['/home/user/Test.000013.1_SceneDepthWorldUnits.png'] = right_depth_data
            files['/home/user/Test.000013.1_ObjectMask.png'] = right_labels_data
            files['/home/user/Test.000013.1_WorldNormals.png'] = right_normals_data

        if mock_isfile is not None:
            mock_isfile.side_effect = lambda fn: fn in files
        if mock_cv2 is not None:
            mock_cv2.imread.side_effect = lambda fn: files[fn][:, :, ::-1] if fn in files else None
        if mock_json_load is not None:
            if image_metadata is None:
                image_metadata = {}
            mock_json_load.return_value = du.defaults(image_metadata, {
                'Version': '0.1.0',
                'Camera Location': {'X': 1, 'Y': 1, 'Z': 1},
                'Camera Orientation': {'W': 1, 'X': 1, 'Y': 1, 'Z': 1}
            })
        if mock_pickle is not None:
            mock_pickle.dumps.side_effect = lambda data, protocol=None: data

        # Mock the database client for insert actions
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_collection.find_one.return_value = None
        mock_db_client.grid_fs = mock.create_autospec(gridfs.GridFS)

        im_id = bson.objectid.ObjectId()
        depth_id = bson.objectid.ObjectId()
        labels_id = bson.objectid.ObjectId()
        normals_id = bson.objectid.ObjectId()

        right_im_id = right_depth_id = right_labels_id = right_normals_id = None
        if make_stereo:
            right_im_id = bson.objectid.ObjectId()
            right_depth_id = bson.objectid.ObjectId()
            right_labels_id = bson.objectid.ObjectId()
            right_normals_id = bson.objectid.ObjectId()

        def mock_put(data):
            if np.array_equal(data, im_data):
                return im_id
            elif np.array_equal(data, depth_data):
                return depth_id
            elif np.array_equal(data, labels_data):
                return labels_id
            elif np.array_equal(data, normals_data):
                return normals_id
            elif np.array_equal(data, right_im_data):
                return right_im_id
            elif np.array_equal(data, right_depth_data):
                return right_depth_id
            elif np.array_equal(data, right_labels_data):
                return right_labels_id
            elif np.array_equal(data, right_normals_data):
                return right_normals_id
            return None
        mock_db_client.grid_fs.put.side_effect = mock_put

        if make_stereo:
            return (mock_db_client, (im_data, depth_data, labels_data, normals_data,
                                     right_im_data, right_depth_data, right_labels_data, right_normals_data),
                    (im_id, depth_id, labels_id, normals_id,
                     right_im_id, right_depth_id, right_labels_id, right_normals_id))
        return (mock_db_client, (im_data, depth_data, labels_data, normals_data),
                (im_id, depth_id, labels_id, normals_id))

    def assertCalled(self, mock_callable):
        self.assertTrue(mock_callable.called, "{0} was not called".format(str(mock_callable)))

    def assertNPEqual(self, arr1, arr2, approx=-1.0):
        if approx >= 0:
            self.assertTrue(np.all(np.isclose(arr1, arr2, rtol=0, atol=approx)),
                            "{0} != {1}".format(str(arr1), str(arr2)))
        else:
            self.assertTrue(np.array_equal(arr1, arr2), "{0} != {1}".format(str(arr1), str(arr2)))
