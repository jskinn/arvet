#Copyright (c) 2017, John Skinner
import unittest
import bson
import dataset.tum.tum_manager as tum_manager


class TestTUMManager(unittest.TestCase):

    def test_serialize_and_deserialize(self):
        obj1 = tum_manager.TUMManager(
            config={
                'rgbd_dataset_freiburg1_xyz': True,
                'rgbd_dataset_freiburg1_rpy': False,
                'rgbd_dataset_freiburg2_xyz': True,
                'rgbd_dataset_freiburg2_rpy': True
            },
            dataset_ids={
                'rgbd_dataset_freiburg1_xyz': bson.ObjectId()
            }
        )
        s_obj1 = obj1.serialize()

        obj2 = tum_manager.TUMManager.deserialize(s_obj1)
        s_obj2 = obj2.serialize()

        self.assert_managers_equal(obj1, obj2)
        self.assertEqual(s_obj1, s_obj2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            obj2 = tum_manager.TUMManager.deserialize(s_obj2)
            s_obj2 = obj2.serialize()
            self.assert_managers_equal(obj1, obj2)
            self.assertEqual(s_obj1, s_obj2)

    def assert_managers_equal(self, manager1, manager2):
        self.assertIsInstance(manager1, tum_manager.TUMManager)
        self.assertIsInstance(manager2, tum_manager.TUMManager)
        self.assertEqual(manager1._config, manager2._config)
        self.assertEqual(manager1._dataset_ids, manager2._dataset_ids)
