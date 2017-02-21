from pymongo import ASCENDING
from enum import Enum
from copy import deepcopy
from core.trajectory import TrajectoryBuilder
from database.entity import Entity


class DatasetType(Enum):
    IMAGE_SEQUENCE = 1,
    DISJOINT_COLLECTION = 2


class Dataset(Entity):
    """
    A dataset object, which is a collection of image data
    and accompanying metadata, upon which a vision system can be run

    Datasets are entities and are stored in the database
    """

    def __init__(self, metadata=None, images=None, id_=None, **kwargs):
        super().__init__(id_=id_)

        if images is None:
            self._image_ids = []
        else:
            self._image_ids = images

        self._type = DatasetType.IMAGE_SEQUENCE # TODO: Get the dataset type in the metadata somehow. prolly needs a patch

        # extract the key properties from the metadata
        self._framerate = metadata['Framerate']
        self._resolution = {
            'height': metadata['Capture Resolution']['Height'],
            'width': metadata['Capture Resolution']['Width']
        }
        self._available_frame_metadata = list(metadata['Available Frame Metadata'])

        self._world_name = metadata['World Name']
        self._world_information = deepcopy(metadata['World Information'])
        self._material_properties = deepcopy(metadata['Material Properties'])
        self._geometry_properties = deepcopy(metadata['Geometry Detail'])

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return len(self)

    @property
    def duration(self):
        return self.size / self.framerate

    @property
    def framerate(self):
        return self._framerate

    @property
    def resolution(self):
        return self._resolution

    @property
    def available_frame_metadata(self):
        return self._available_frame_metadata

    @property
    def world_name(self):
        return self._world_name

    @property
    def world_information(self):
        return self._world_information
    
    @property
    def material_properties(self):
        return self._material_properties

    @property
    def geometry_properties(self):
        return self._geometry_properties

    def __len__(self):
        return len(self._image_ids)

    def load_images(self, db_client):
        """
        Load the images in the dataset.
        Operations requiring the images can only be called on the DatasetImageSet

        :param db_client: The database client we're loading images from
        :return: A DatasetImagesSet containing all the images for this dataset
        :rtype DatasetImageSet:
        """
        s_images_cursor = db_client.image_collection.find({'dataset': self.identifier}).sort('index', ASCENDING)
        images = []
        for s_image in s_images_cursor:
            images.append(db_client.deserialize_entity(s_image))
        return DatasetImageSet(self, images)

    def get(self, index):
        return self._image_ids[index]

    def serialize(self):
        serialized = super().serialize()

        serialized['images'] = list(self._image_ids)

        serialized['sequence_type'] = str(self._type)
        serialized['framerate'] = self._framerate
        serialized['resolution'] = self._resolution
        serialized['available_frame_metadata'] = self._available_frame_metadata

        serialized['world_name'] = self._world_name
        serialized['world_information'] = deepcopy(self._world_information)
        serialized['material_properties'] = deepcopy(self.material_properties)
        serialized['geometry_properties'] = deepcopy(self.geometry_properties)

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'images' in serialized_representation:
            kwargs['images'] = serialized_representation['images']

        metadata = {}
        if 'sequence_type'in serialized_representation:
            metadata['Type'] = serialized_representation['sequence_type']
        if 'framerate' in serialized_representation:
            metadata['Framerate'] = serialized_representation['framerate']
        if 'resolution' in serialized_representation:
            metadata['Capture Resolution'] = {
                'Height': serialized_representation['resolution']['height'],
                'Width': serialized_representation['resolution']['width'],
            }
        if 'available_frame_metadata' in serialized_representation:
            metadata['Available Frame Metadata'] = serialized_representation['available_frame_metadata']
        if 'world_name' in serialized_representation:
            metadata['World Name'] = serialized_representation['world_name']
        if 'world_information' in serialized_representation:
            metadata['World Information'] = serialized_representation['world_information']
        if 'material_properties' in serialized_representation:
            metadata['Material Properties'] = serialized_representation['material_properties']
        if 'geometry_properties' in serialized_representation:
            metadata['Geometry Detail'] = serialized_representation['geometry_properties']

        kwargs['metadata'] = metadata
        return super().deserialize(serialized_representation, **kwargs)


class DatasetImageSet:
    """
    A dataset set of images.
    Much of the time, we don't want to actually load all the images in the dataset,
    so the Dataset object has a list of image ids, but no actual image objects.
    When you need the images, use DatasetImageSet instead.
    Operations that depend on the images are on this class.
    """

    def __init__(self, dataset, images):
        self._dataset = dataset
        self._images = images

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        for image in self._images:
            yield image

    def __getitem__(self, item):
        return self._images[item]

    def get_ground_truth_trajectory(self, repeats=0):
        """
        Get the ground truth trajectory for this dataset.
        Requires camera location and orientation for each image
        :param repeats:
        :return:
        """
        builder = TrajectoryBuilder()
        for repeat in range(0, repeats):
            for image in self:
                builder.add_numpy(repeat * self.dataset.duration + image.timestamp,
                                  image.camera_location, image.camera_orientation)
        return builder.build()
