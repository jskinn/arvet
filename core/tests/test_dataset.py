from unittest import TestCase
from core.dataset import Dataset


class TestDataset(TestCase):

    # Actual metadata from a dataset I generated.
    EXAMPLE_METADATA = {
        "World Information": {},
        "Image Filename Format": "{world}.{frame}",
        "Size": 2502,
        "Framerate": 30,
        "Capture Resolution": {"Height": 720, "Width": 1280},
        "Index Padding": 4,
        "Material Properties": {
            "NormalQuality": False,
            "BaseColorQuality": False,
            "NormalMipMapBias": 0,
            "RoughnessQuality": 0,
            "BaseMipMapBias": 0
        },
        "Image Filename Format Mappings": {"world": "SunTemplePathGeneration"},
        "World Name": "SunTemplePathGeneration",
        "Available Frame Metadata": ["Camera Location", "Camera Orientation"],
        "Path Generation": {
            "Smoothing Iterations": 50,
            "Circuits": 1,
            "Smoothing Learning Rate": 0.05,
            "Random Seed": 7118,
            "Agent Properties": {
                "Nav Agent Radius": 45,
                "Can Fly": False,
                "Nav Agent Height": 200,
                "Can Jump": False,
                "Can Walk": True,
                "Nav Walking Search Height": 0.5,
                "Can Crouch": False,
                "Nav Agent Step Height": 45,
                "Can Swim": False
            },
            "Path Height": 180,
            "Min Path Length": 15000,
            "Bounds Max": {"Z": 1360, "X": 2840, "Y": 11625},
            "Generation Type": "Automatic",
            "Negligible Distance": 10,
            "Max Spline Segment Length": 500,
            "Bounds Min": {"Z": -640, "X": -3160, "Y": -5375}
        },
        "Version": "0.1.2"
    }

    def test_identifier(self):
        dataset = Dataset(id_=123, metadata=TestDataset.EXAMPLE_METADATA)
        self.assertEquals(dataset.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'metadata': TestDataset.EXAMPLE_METADATA, 'a':1, 'b':2, 'c': 3}
        dataset = Dataset(**kwargs)
        self.assertEquals(dataset.identifier, 1234)

    def test_serialize_and_deserialize(self):
        dataset1 = Dataset(id_=12345,metadata=dict(TestDataset.EXAMPLE_METADATA))
        s_dataset1 = dataset1.serialize()

        dataset2 = Dataset.deserialize(s_dataset1)
        s_dataset2 = dataset2.serialize()

        self._assert_models_equal(dataset1, dataset2)
        self.assertEquals(s_dataset1, s_dataset2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            dataset2 = Dataset.deserialize(s_dataset2)
            s_dataset2 = dataset2.serialize()
            self._assert_models_equal(dataset1, dataset2)
            self.assertEquals(s_dataset1, s_dataset2)

    def _assert_models_equal(self, dataset1, dataset2):
        """
        Helper to assert that two dataset models are equal
        :param dataset1: Dataset
        :param dataset2: Dataset
        :return:
        """
        if not isinstance(dataset1, Dataset) or not isinstance(dataset2, Dataset):
            self.fail('object was not a dataset')
        self.assertEquals(dataset1.identifier, dataset2.identifier)
        self.assertEquals(dataset1.type, dataset2.type)
        self.assertEquals(dataset1.size, dataset2.size)
        self.assertEquals(dataset1.duration, dataset2.duration)
        self.assertEquals(dataset1.framerate, dataset2.framerate)
        self.assertEquals(dataset1.resolution, dataset2.resolution)
        self.assertEquals(dataset1.available_frame_metadata, dataset2.available_frame_metadata)
        self.assertEquals(dataset1.world_name, dataset2.world_name)
        self.assertEquals(dataset1.world_information, dataset2.world_information)
        self.assertEquals(dataset1.material_properties, dataset2.material_properties)
        self.assertEquals(dataset1.geometry_properties, dataset2.geometry_properties)
