import abc
import database.entity
import core.system


class VisionSystemTrainer(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    This class manages image sources, and how they are used for training.
    This includes handling multiple image sources, shuffling,
    presentation order, sampling, and whatnot

    There are two key styles here, systems that require image sequences, and systems that require single images.
    Training on single images does things like shuffle the images, present the same images more than once,
    etc, etc.
    Training on image sequences does similar things, but at the image source level instead,
    presenting each source in order, but potentially randomizing the order the sources is presented.
    They are distinguished by the required_training_type property of the vision system trainee.

    Instances of this class must be entities, with particular settings, so that the experiments can load them.
    """

    @abc.abstractmethod
    def can_train_trainee(self, potential_trainee):
        """
        Is this trainer set-up to train this particular system trainee.
        Should check that it matches the required training type,
        and that it has at least one appropriate image source.
        :param potential_trainee: A VisionSystemTrainee that
        :return:
        """
        pass

    @abc.abstractmethod
    def train_vision_system(self, vision_system_trainee):
        """
        Train a new vision system using the given trainee
        :param vision_system_trainee: The builder-type trainee to feed training images to.
        :return:
        """
        pass


class VisionSystemTrainee(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A separate class like a builder for training Vision Systems.
    It holds all the algorithm specific state during training,
    as opposed to the training approach, which is provided by the Trainer (above),
    and is probably more generic.

    This breaks some awkward dependencies with having the system class
    also be responsible for training.
    
    All trainees must be entities so that the job system can load them when running training tasks.
    However, trainees should not store state generated during training,
    that is the job of the TrainedVisionSytem returned by finish_training
    """

    @property
    @abc.abstractmethod
    def required_training_type(self):
        """
        What kind of training does this system require?

        :return: NON_SEQUENTIAL for systems trained on random images, and SEQUENTIAL for systems trained on video
        :rtype core.sequence_type.ImageSequenceType:
        """
        pass

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
    def start_training(self, num_images=None):
        """
        Start training the system.
        After calling this, train_with_image should be called with images from the image source.
        Only call start_training once, and call finish_training when all image sources are complete.
        Provides the expected number of images if available, for scaling learning rates and such.
        :param num_images: The anticipated total number of images that will be provided, if available.
        :return: void
        """
        pass

    @abc.abstractmethod
    def train_with_image(self, image, index=None):
        """
        Train the system with a particular image
        :param image: The image to train with, and Image object
        :param index: A timestamp or index into the training images, for scaling learning rates
        :return: void
        """
        pass

    @abc.abstractmethod
    def start_validation(self, num_validation_images=None):
        """
        While training, start performing validation
        This is done to reduce overfitting
        :param num_validation_images: The amount of validation data, so we can pre-allocate storage
        :return:
        """
        pass

    @abc.abstractmethod
    def validate_with_image(self, image):
        """
        Measure performance with a particular image. This should accumulate during validation.
        :param image: The image
        :return:
        """
        pass

    @abc.abstractmethod
    def finish_validation(self):
        """
        Finish validation and return to training.
        :return:
        """
        pass


    @abc.abstractmethod
    def finish_training(self, trainer_id, image_source_ids=None, training_settings=None):
        """
        Finish training, producing a new vision system instance,
        which will be saved in the database as a new vision system to be tested.
        
        If the training method is not sequential, requiring all images to train,
        then train_with_image should simply accumulate images, and this method
        should actually perform the training.

        :param trainer_id: The id of the VisionSystemTrainer that provided the images
        :param image_source_ids: The ids of the image sources used by the trainer, if available
        :param training_settings: The settings of the trainer. Optional.
        :return: TrainedVisionSystem A completed TrainedVisionSystem, with state created from the given images.
        """
        pass


class TrainedVisionSystem(core.system.VisionSystem, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A subclass of a vision system that are produced by training.
    Instances of this class are built by VisionSystemTrainers using VisionSystemTrainees.
    The only difference here is that they have references back to the image sources they were trained with
    #TODO: Once we've sorted out universal ids for images (db or generated), store those for all training images
    """

    def __init__(self, vision_system_trainer, training_image_source_ids=None, training_settings=None,
                 id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        self._trainer = vision_system_trainer
        self._training_image_sources = training_image_source_ids
        self._training_settings = training_settings

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

    @property
    def training_settings(self):
        """
        Get the settings used to train this system
        :return:
        """
        return self._training_settings

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['vision_system_trainer'] = self.vision_system_trainer
        serialized['training_image_sources'] = self.training_image_sources
        serialized['training_settings'] = self._training_settings
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'vision_system_trainer' in serialized_representation:
            kwargs['vision_system_trainer'] = serialized_representation['vision_system_trainer']
        if 'training_image_sources' in serialized_representation:
            kwargs['training_image_source_ids'] = serialized_representation['training_image_sources']
        if 'training_settings' in serialized_representation:
            kwargs['training_settings'] = serialized_representation['training_settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
