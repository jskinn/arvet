import unittest
import unittest.mock as mock
import core.sequence_type
import simulation.simulator
import simulation.controllers.combinatorial_sampling_controller as comb_controller


class TestCombinatorialSampleController(unittest.TestCase):

    def test_sequence_type_is_nonsequential(self):
        subject = comb_controller.CombinatorialSampleController(
            range(-100, 100, 25),
            range(-100, 100, 25),
            range(-100, 100, 25),
            range(-90, 90, 10),
            range(-90, 90, 10),
            range(-180, 180, 20)
        )
        self.assertEqual(core.sequence_type.ImageSequenceType, subject.sequence_type)


def create_mock_simulator():
    mock_simulator = mock.create_autospec(simulation.simulator.Simulator)
    return mock_simulator
