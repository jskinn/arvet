import unittest
import numpy as np
import cv2
import os.path
import json
import glob
import pymongo.collection
import gridfs
import bson.objectid
import transforms3d as tf3d
import unittest.mock as mock
import util.transform
import util.dict_utils as du
import metadata.image_metadata as imeta
import core.image_entity
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

    def test_sanitize_additional_metadata_removes_keys(self):
        metadata = import_gen.sanitize_additional_metadata({
            'Version': '1.2.3.45.6',
            'A': 10,
            'Camera Location': util.transform.Transform(location=(100, 10, -30), rotation=(-4, 152, 15, -2)),
            'B': 11,
            'Camera Orientation': {'W': 12, 'X': 15, 'Y': -22, 'Z': 999},
            'C': 12
        })
        self.assertEqual(metadata, {
            'A': 10,
            'B': 11,
            'C': 12
        })

    def test_build_image_metadata_constructs_image_metadata_from_metadata(self):
        metadata = import_gen.build_image_metadata(np.zeros((600, 800, 3), dtype='uint8'), None, None, {
            "Version": "0.1.2",
            "Material Properties": {
                "RoughnessQuality": 0,
                "BaseMipMapBias": 0,
                "NormalQuality": 1
            },
            "Geometry Detail": {
                "Forced LOD level": 0
            },
            "World Name": "AIUE_V01_001",
            "Framerate": 30,
            "Index Padding": 4,
            "Capture Resolution": {
                "Width": 1280,
                "Height": 720
            },
            "Image Filename Format": "{world}{material}.{frame}.{stereopass}",
            "File Extension": ".png",
            "Image Filename Format Mappings": {
                "stereopass": "1",
                "material": "base-colour",
                "world": "AIUE_V01_001",
                "height": "720",
                "width": "1280",
                "fps": "30"
            },
            "Available Frame Metadata": [
                "Camera Location",
                "Camera Orientation"
            ],
            "World Information": {
                "Camera Path": {
                    "Path Length": 4720.88818359375,
                    "Path Generation": {
                        "Generation Type": "Automatic",
                        "Generator": "ACameraPathGenerator",
                        "Min Path Length": 2000,
                        "Path Height": 50,
                        "Circuits": 0,
                        "Smoothing Iterations": 10,
                        "Smoothing Learning Rate": 0.05000000074505806,
                        "Turning Circle": 50,
                        "Random Seed": 0,
                        "Bounds Min": {
                            "X": 1000,
                            "Y": 1000,
                            "Z": 150
                        },
                        "Negligible Distance": 10,
                        "Agent Properties": {
                            "Agent Height": -1,
                            "Agent Radius": -1,
                            "Agent Step Height": -1,
                            "Nav Walinkg Search Height Scale": 0.5
                        }
                    }
                }
            }
        })
        self.assertEqual(600, metadata.height)
        self.assertEqual(800, metadata.width)
        self.assertEqual(imeta.ImageSourceType.SYNTHETIC, metadata.source_type)
        self.assertEqual("AIUE_V01_001", metadata.simulation_world)
        self.assertFalse(metadata.roughness_enabled)
        self.assertTrue(metadata.normal_maps_enabled)
        self.assertEqual(0, metadata.texture_mipmap_bias)
        self.assertEqual(0, metadata.geometry_decimation)
        self.assertEqual(90, metadata.fov)
        self.assertEqual(0, metadata.procedural_generation_seed)

    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_image_object_returns_none_if_image_doesnt_exist(self, mock_isfile, *_):
        expected_filename = '/home/user/Test.000013.0.png'
        mock_isfile.side_effect = lambda fn: (fn != expected_filename)
        result = import_gen.import_image_object(base_path='/home/user',
                                                filename_format="Test.{frame}.{stereopass}",
                                                mappings={},
                                                index_padding=6,
                                                index=13,
                                                extension='.png',
                                                dataset_metadata={})
        self.assertIn(mock.call(expected_filename), mock_isfile.call_args_list)
        self.assertIsNone(result)

    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_image_object_returns_none_if_metadata_doesnt_exist(self, mock_isfile, *_):
        expected_filename = '/home/user/Test.000013.0.png.metadata.json'
        mock_isfile.side_effect = lambda fn: (fn != expected_filename)
        result = import_gen.import_image_object(base_path='/home/user',
                                                filename_format="Test.{frame}.{stereopass}",
                                                mappings={},
                                                index_padding=6,
                                                index=13,
                                                extension='.png',
                                                dataset_metadata={
                                                    "Material Properties": {
                                                        "RoughnessQuality": 0,
                                                        "BaseMipMapBias": 0,
                                                        "NormalQuality": 1
                                                    },
                                                    "Geometry Detail": {
                                                        "Forced LOD level": 0
                                                    },
                                                    "World Name": "AIUE_V01_001",
                                                    "World Information": {
                                                        "Camera Path": {
                                                            "Path Generation": {
                                                                "Random Seed": 0,
                                                            }
                                                        }
                                                    }
                                                })
        self.assertIn(mock.call(expected_filename), mock_isfile.call_args_list)
        self.assertIsNone(result)

    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_import_image_object_returns_constructed_image_object(self, mock_cv2, mock_isfile, mock_json_load):
        _, images, _ = self.prepare_mocks(mock_cv2=mock_cv2, mock_isfile=mock_isfile, mock_json_load=mock_json_load,
                                          image_metadata={
                                              'Camera Location': {'X': -22, 'Y': 13, 'Z': 4},
                                              'Camera Orientation': {'W': -0.5, 'X': 0.5, 'Y': -0.5, 'Z': -0.5}
                                          })
        im_data, depth_data, labels_data, normals_data = images
        image = import_gen.import_image_object(base_path='/home/user',
                                               filename_format="Test.{frame}.{stereopass}",
                                               mappings={},
                                               index_padding=6,
                                               index=13,
                                               extension='.png',
                                               dataset_metadata={
                                                   'world': 'mock',
                                                   'quality': 'maximum',
                                                   "Material Properties": {
                                                       "RoughnessQuality": 0,
                                                       "BaseMipMapBias": 2,
                                                       "NormalQuality": 1
                                                   },
                                                   "Geometry Detail": {
                                                       "Forced LOD level": 4
                                                   },
                                                   "World Name": "AIUE_V01_001",
                                                   "World Information": {
                                                       "Camera Path": {
                                                           "Path Generation": {
                                                               "Random Seed": 1236,
                                                           }
                                                       }
                                                   }
                                               })
        self.assertIsInstance(image, core.image_entity.ImageEntity)
        self.assertNPEqual((-22, -13, 4), image.metadata.camera_pose.location)
        self.assertNPEqual((0.5, 0.5, 0.5, -0.5), image.metadata.camera_pose.rotation_quat(True),
                           approx=0.000000000000001)
        self.assertNPEqual(im_data, image.data)
        self.assertNPEqual(depth_data, image.depth_data)
        self.assertNPEqual(labels_data, image.labels_data)
        self.assertNPEqual(normals_data, image.world_normals_data)
        self.assertEqual(2, image.metadata.texture_mipmap_bias)
        self.assertEqual(True, image.metadata.normal_maps_enabled)
        self.assertEqual(False, image.metadata.roughness_enabled)
        self.assertEqual(4, image.metadata.geometry_decimation)
        self.assertEqual(1236, image.metadata.procedural_generation_seed)
        self.assertEqual('AIUE_V01_001', image.metadata.simulation_world)

    @mock.patch('dataset.generated.import_generated_dataset.open', mock.mock_open(), create=True)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('dataset.generated.import_generated_dataset.cv2', autospec=cv2)
    def test_import_image_object_loads_stereo_image(self, mock_cv2, mock_isfile, mock_json_load):
        _, images, _ = self.prepare_mocks(mock_isfile=mock_isfile, mock_json_load=mock_json_load, mock_cv2=mock_cv2,
                                          image_metadata={
                                              'Camera Location': {'X': -22, 'Y': 13, 'Z': 4},
                                              'Camera Orientation': {'W': -0.5, 'X': 0.5, 'Y': -0.5, 'Z': -0.5}
                                          }, make_stereo=True)
        (im_data, depth_data, labels_data, normals_data, right_im_data,
         right_depth_data, right_labels_data, right_normals_data) = images

        stereo_image = import_gen.import_image_object(base_path='/home/user',
                                                      filename_format="Test.{frame}.{stereopass}",
                                                      mappings={},
                                                      index_padding=6,
                                                      index=13,
                                                      extension='.png',
                                                      dataset_metadata={
                                                          'world': 'nope',
                                                          'quality': 'overpowered',
                                                          "Material Properties": {
                                                              "RoughnessQuality": 0,
                                                              "BaseMipMapBias": 2,
                                                              "NormalQuality": 1
                                                          },
                                                          "Geometry Detail": {
                                                              "Forced LOD level": 4
                                                          },
                                                          "World Name": "AIUE_V01_001",
                                                          "World Information": {
                                                              "Camera Path": {
                                                                  "Path Generation": {
                                                                      "Random Seed": 12365,
                                                                  }
                                                              }
                                                          }
                                                      })
        self.assertIsInstance(stereo_image, core.image_entity.StereoImageEntity)
        self.assertNPEqual((-22, -13, 4), stereo_image.metadata.camera_pose.location)
        self.assertNPEqual((0.5, 0.5, 0.5, -0.5), stereo_image.metadata.camera_pose.rotation_quat(True),
                           approx=0.000000000000001)
        self.assertNPEqual((-22, -13, 4), stereo_image.metadata.right_camera_pose.location)
        self.assertNPEqual((0.5, 0.5, 0.5, -0.5), stereo_image.metadata.right_camera_pose.rotation_quat(True),
                           approx=0.000000000000001)
        self.assertNPEqual(im_data, stereo_image.left_data)
        self.assertNPEqual(depth_data, stereo_image.left_depth_data)
        self.assertNPEqual(labels_data, stereo_image.left_labels_data)
        self.assertNPEqual(normals_data, stereo_image.left_world_normals_data)
        self.assertNPEqual(right_im_data, stereo_image.right_data)
        self.assertNPEqual(right_depth_data, stereo_image.right_depth_data)
        self.assertNPEqual(right_labels_data, stereo_image.right_labels_data)
        self.assertNPEqual(right_normals_data, stereo_image.right_world_normals_data)
        self.assertEqual(2, stereo_image.metadata.texture_mipmap_bias)
        self.assertEqual(True, stereo_image.metadata.normal_maps_enabled)
        self.assertEqual(False, stereo_image.metadata.roughness_enabled)
        self.assertEqual(4, stereo_image.metadata.geometry_decimation)
        self.assertEqual(12365, stereo_image.metadata.procedural_generation_seed)
        self.assertEqual('AIUE_V01_001', stereo_image.metadata.simulation_world)

    def test_import_dataset_returns_none_if_metadata_doesnt_exist(self):
        self.assertIsNone(import_gen.import_dataset('notafile.fail', mock.create_autospec(database.client)))

    @mock.patch('dataset.generated.import_generated_dataset.glob.glob', autospec=glob.glob)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_dataset_reads_dataset_metadata(self, mock_isfile, mock_json_load, mock_glob):
        mock_glob.return_value = []
        mock_isfile.return_value = True
        mock_json_load.return_value = {
            'World Name': 'test_world',
            'File Extension': '.img'    # Need this so it can call glob and not error
        }
        mock_open = mock.mock_open()
        with mock.patch('dataset.generated.import_generated_dataset.open', mock_open, create=True):
            import_gen.import_dataset('/temp/isfilehonest', mock.create_autospec(database.client))
        self.assertTrue(mock_open.called)
        self.assertTrue('/temp/isfilehonest', mock_open.call_args[0][0])
        self.assertTrue(mock_json_load.called)

    @mock.patch('dataset.generated.import_generated_dataset.import_image_object',
                autospec=import_gen.import_image_object)
    @mock.patch('dataset.generated.import_generated_dataset.glob.glob', autospec=glob.glob)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_dataset_loads_images_for_number_of_image_files(self, mock_isfile, mock_json_load, mock_glob,
                                                                   mock_import_image):
        mock_glob.return_value = list(range(10))    # Values don't matter, only length
        mock_isfile.return_value = True
        mock_json_load.return_value = {
            'World Name': 'test_world',
            'File Extension': '.img',
            'Image Filename Format': '{name}-{color}',
            'Image Filename Format Mappings': {'name': 'yes', 'color': 'red'},
            'Index Padding': 10
        }
        mock_import_image.return_value = mock.create_autospec(core.image_entity.ImageEntity)
        mock_db_client = mock.create_autospec(database.client)
        mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_open = mock.mock_open()
        with mock.patch('dataset.generated.import_generated_dataset.open', mock_open, create=True):
            import_gen.import_dataset('/temp/isfilehonest', mock_db_client)
        self.assertEqual(10, mock_import_image.call_count)

    @mock.patch('dataset.generated.import_generated_dataset.import_image_object',
                autospec=import_gen.import_image_object)
    @mock.patch('dataset.generated.import_generated_dataset.glob.glob', autospec=glob.glob)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_dataset_stops_when_failed_to_load(self, mock_isfile, mock_json_load, mock_glob,
                                                                   mock_import_image):
        mock_glob.return_value = list(range(10))    # Values don't matter, only length
        mock_isfile.return_value = True
        mock_json_load.return_value = {
            'File Extension': '.img',
            'Image Filename Format': '{name}-{color}',
            'Image Filename Format Mappings': {'name': 'yes', 'color': 'red'},
            'Index Padding': 10
        }
        mock_import_image.return_value = None
        mock_db_client = mock.create_autospec(database.client)
        mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_open = mock.mock_open()
        with mock.patch('dataset.generated.import_generated_dataset.open', mock_open, create=True):
            import_gen.import_dataset('/temp/isfilehonest', mock_db_client)
        self.assertEqual(1, mock_import_image.call_count)

    @mock.patch('dataset.generated.import_generated_dataset.import_image_object',
                autospec=import_gen.import_image_object)
    @mock.patch('dataset.generated.import_generated_dataset.glob.glob', autospec=glob.glob)
    @mock.patch('dataset.generated.import_generated_dataset.json.load', autospec=json.load)
    @mock.patch('dataset.generated.import_generated_dataset.os.path.isfile', autospec=os.path.isfile)
    def test_import_dataset_saves_image_collection(self, mock_isfile, mock_json_load, mock_glob,
                                                                   mock_import_image):
        mock_glob.return_value = list(range(10))    # Values don't matter, only length
        mock_isfile.return_value = True
        mock_json_load.return_value = {
            'File Extension': '.img',
            'Image Filename Format': '{name}-{color}',
            'Image Filename Format Mappings': {'name': 'yes', 'color': 'red'},
            'Index Padding': 10
        }
        mock_import_image.return_value = mock.create_autospec(core.image_entity.ImageEntity)
        mock_db_client = mock.create_autospec(database.client)
        mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_source_collection.find_one.return_value = None
        mock_open = mock.mock_open()
        with mock.patch('dataset.generated.import_generated_dataset.open', mock_open, create=True):
            import_gen.import_dataset('/temp/isfilehonest', mock_db_client)
        self.assertTrue(mock_db_client.image_source_collection.find_one.called)
        self.assertTrue(mock_db_client.image_source_collection.insert.called)

    @staticmethod
    def prepare_mocks(mock_cv2=None, mock_isfile=None, mock_json_load=None, mock_pickle=None,
                      image_metadata=None, make_stereo=False):
        """
        Set up mocks for testing loading an image.
        The image loader expects to be able to read several image files.
        We need to set up the mocks to support loading each of these files as expected
        The mocks passed in will be modified, additional mocks will be returned
        :param mock_cv2:
        :param mock_isfile:
        :param mock_json_load:
        :param mock_pickle:
        :param image_metadata:
        :param make_stereo:
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
