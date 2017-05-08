import abc
import database.entity
import core.system


class VisionSystemTrainer(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A separate class like a builder for training Vision Systems.
    This breaks some awkward dependencies with having the system class
    also be responsible for training.
    
    Like vision systems, all trainers must be entities so that the database can load them.
    """

    @abc.abstractmethod
    def is_image_source_appropriate(self, image_source):
        """
        Is the dataset appropriate for training this vision system.
        This allows more fine-grained filtering of which image sources are used for training particular systems,
        because it can have access to the trainer settings, which may conditionally exclude some sources.

        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def start_training(self, training_image_source_id):
        """
        Start training the system.
        After calling this, train_with_image should be called with images from the image source.
        If we want to train on image data from multiple image sources, call start_training
        for each image source, and only call finish_training when all image sources are complete
        
        :param training_image_source_id: ObjectId
        :return: void
        """
        pass

    @abc.abstractmethod
    def train_with_image(self, image):
        """
        Train the system with a particular image
        :param image: 
        :return: void
        """
        pass

    @abc.abstractmethod
    def finish_training(self):
        """
        Finish training, producing a new vision system instance,
        which will be saved in the database as a new vision system to be tested.
        
        If the training method is not sequential, requiring all images to train,
        then train_with_image should simply accumulate images, and this method
        should actually perform the training.
        
        :return: TrainedVisionSystem  
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_training_dataset_criteria(cls):
        """
        Get search criteria for finding datasets that can be used by this system for training.
        The scheduling process with try and train vision systems using all image sources that match this criteria.

        :return: A set of criteria for finding datasets that match, in mongodb query format
        :rtype: dict
        """
        return {}


class TrainedVisionSystem(core.system.VisionSystem, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A subclass of a vision system that are produced by training.
    Instances of this class are returned by VisionSystemTrainers.
    The only difference here is that they have references back to the image 
    """

    def __init__(self, vision_system_trainer, training_image_source_ids, id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        self._trainer = vision_system_trainer
        self._training_image_sources = training_image_source_ids

    @property
    def vision_system_trainer(self):
        """
        A reference to the training object used to train this system
        :return: 
        """
        return self._trainer

    @property
    def training_image_sources(self):
        """
        A list of all the image sources used to train this system.
        :return: 
        """
        return self._training_image_sources

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['vision_system_trainer'] = self.vision_system_trainer
        serialized['training_image_sources'] = self.training_image_sources
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'vision_system_trainer' in serialized_representation:
            kwargs['vision_system_trainer'] = serialized_representation['vision_system_trainer']
        if 'training_image_sources' in serialized_representation:
            kwargs['training_image_source_ids'] = serialized_representation['training_image_sources']
        return super().deserialize(serialized_representation, db_client, **kwargs)
