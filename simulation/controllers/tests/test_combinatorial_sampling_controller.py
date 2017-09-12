#Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import core.sequence_type
import util.transform as tf
import simulation.simulator
import simulation.controllers.combinatorial_sampling_controller as comb_controller


class TestCombinatorialSampleController(unittest.TestCase):

    def test_sequence_type_is_nonsequential(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertEqual(core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, subject.sequence_type)

    def test_is_depth_available_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_depth_available)

    def test_is_depth_available_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_depth_available = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_depth_available)
        #self.assertIn(mock.call('is_depth_available'), sim.__getattr__.call_args_list)

    def test_is_per_pixel_labels_available_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_per_pixel_labels_available)

    def test_is_per_pixel_labels_available_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_per_pixel_labels_available = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_per_pixel_labels_available)
        #self.assertIn(mock.call('is_per_pixel_labels_available'), sim.__getattr__.call_args_list)

    def test_is_labels_available_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_labels_available)

    def test_is_labels_available_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_labels_available = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_labels_available)
        #self.assertIn(mock.call('is_labels_available'), sim.__getattr__.call_args_list)

    def test_is_normals_available_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_normals_available)

    def test_is_normals_available_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_normals_available = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_normals_available)
        #self.assertIn(mock.call('is_normals_available'), sim.__getattr__.call_args_list)

    def test_is_stereo_available_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_stereo_available)

    def test_is_stereo_available_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_stereo_available = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_stereo_available)
        #self.assertIn(mock.call('is_stereo_available'), sim.__getattr__.call_args_list)

    def test_is_stored_in_database_false_without_sim(self):
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        self.assertFalse(subject.is_stored_in_database)

    def test_is_stored_in_database_depends_on_simulator(self):
        sim = create_mock_simulator()
        sim.is_stored_in_database = True
        subject = comb_controller.CombinatorialSampleController(10, 10, 10, 0.5, 0.5, 0.5)
        subject.set_simulator(sim)
        self.assertTrue(subject.is_stored_in_database)
        #self.assertIn(mock.call('is_stored_in_database'), sim.__getattr__.call_args_list)

    def test_get_returns_every_combination_of_samples(self):
        sim = create_mock_simulator()
        subject = comb_controller.CombinatorialSampleController(
            (1, 2), (3, 4), (5, 6),
            (0.1, 0.2), (0.3, 0.4), (0.5, 0.6))
        subject.set_simulator(sim)
        self.assertEqual(64, len(subject))
        subject.begin()
        for idx in range(len(subject)):
            subject.get(idx)
        for x in (1, 2):
            for y in (3, 4):
                for z in (5, 6):
                    for roll in (0.1, 0.2):
                        for pitch in (0.3, 0.4):
                            for yaw in (0.5, 0.6):
                                self.assertIn(
                                    mock.call(tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw))),
                                    sim.set_camera_pose.call_args_list,
                                    "No call found for ({0}, {1}, {2}), ({3}, {4}, {5})".format(x, y, z,
                                                                                                roll, pitch, yaw)
                                )

    def test_get_next_image_returns_every_combination_of_samples(self):
        sim = create_mock_simulator()
        subject = comb_controller.CombinatorialSampleController(
            (1, 2), (3, 4), (5, 6),
            (0.1, 0.2), (0.3, 0.4), (0.5, 0.6))
        subject.set_simulator(sim)
        self.assertEqual(64, len(subject))
        subject.begin()
        for idx in range(len(subject)):
            subject.get_next_image()
        self.assertTrue(subject.is_complete())
        for x in (1, 2):
            for y in (3, 4):
                for z in (5, 6):
                    for roll in (0.1, 0.2):
                        for pitch in (0.3, 0.4):
                            for yaw in (0.5, 0.6):
                                self.assertIn(
                                    mock.call(tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw))),
                                    sim.set_camera_pose.call_args_list,
                                    "No call found for ({0}, {1}, {2}), ({3}, {4}, {5})".format(x, y, z,
                                                                                                roll, pitch, yaw)
                                )


def create_mock_simulator():
    mock_simulator = mock.create_autospec(simulation.simulator.Simulator)
    return mock_simulator
