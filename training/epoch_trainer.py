import itertools
import numpy as np
import metadata.image_metadata as imeta
import core.sequence_type
import core.trained_system
import core.image
import training.image_sampler


class EpochTrainer(core.trained_system.VisionSystemTrainer):
    """
    A Trainer that performs Deep-learning style epoch-based training.
    This is based on parts of the Keras-FRCNN training approach.
    I've tried to generalize it as much as possible though.

    The big difference here is that this can actually use it's validation set.
    It also handles data augmentation differently.
    Rather than randomly augmenting data, augmented versions are shuffled in with the unaugmented versions,
    so that if we loop over the entire training set, we guarantee to get each base image and each augmented image
    exactly once
    """
    def __init__(self, num_epochs=2000, use_source_length=False, epoch_length=1000, image_sources=None,
                 horizontal_flips=False, vertical_flips=False, rot_90=False, validation_fraction=0.3, id_=None,
                 **kwargs):
        """
        Create an Epoch Trainer, with a bunch of settings.
        :param num_epochs: The number of training epochs
        :param use_source_length: Should each epoch be based on the size of the image source.
        If true, will loop through the entire image source exactly once per epoch. Default false.
        :param epoch_length: How many images in each epoch, assuming we're not basing it off the image source
        :param image_sources: The set of initial image sources to use. More can be added.
        :param horizontal_flips: Augment data with horizontal flips? Default False
        :param vertical_flips: Augment data with vertical flips? Default False
        :param rot_90: Augment data with rotations of multiples of 90 degrees. Default False
        :param validation_fraction: The fraction of data to use in the validation set. Default 0.3 (30%)
        :param id_: The database id
        :param kwargs:
        """
        super().__init__(id_=id_, **kwargs)
        self._num_epochs = int(num_epochs)
        self._epoch_length = int(epoch_length)
        self._use_image_source_length = bool(use_source_length)
        self._horizontal_flips = bool(horizontal_flips)
        self._vertical_flips = bool(vertical_flips)
        self._rot_90 = bool(rot_90)
        self._validation_fraction = max(min(float(validation_fraction), 1.0), 0.0)
        self._image_sources = []
        if image_sources is not None:
            for image_source in image_sources:
                self.add_image_source(image_source)

    @property
    def num_image_sources(self):
        """
        The current number of image sources
        :return:
        """
        return len(self._image_sources)

    def add_image_source(self, image_source):
        """
        Add an image source to be used for training.
        Systems will often train from multiple image sources.
        This trainer can only use image sources that allow random access, that is,
        they have a defined get or __getitem__ method. Image collections have this by default.

        :param image_source: An image source object
        :return: True iff the image source was added for use training this object
        """
        if hasattr(image_source, 'begin') and (hasattr(image_source, 'get') or hasattr(image_source, '__getitem__')):
            self._image_sources.append(image_source)
            return True
        return False

    def can_train_trainee(self, potential_trainee):
        """
        Is this trainer set-up to train this particular system trainee.
        Should check that it matches the required training type,
        and that it has at least one appropriate image source.
        :param potential_trainee: A VisionSystemTrainee that
        :return:
        """
        return (potential_trainee.required_training_type == core.sequence_type.ImageSequenceType.NON_SEQUENTIAL and
                any(potential_trainee.is_image_source_appropriate(image_source)
                    for image_source in self._image_sources))

    def train_vision_system(self, vision_system_trainee):
        """
        Train a new vision system using the given trainee
        :param vision_system_trainee: the trainee to train, usually some kind of CNN
        :return: The trained vision system
        """
        # Get all the image sources appropriate for training
        image_sources = [image_source for image_source in self._image_sources
                         if vision_system_trainee.is_image_source_appropriate(image_source)]
        if len(image_sources) <= 0:
            # No appropriate image sources, fail
            return None

        # Create data augmentation
        data_augments = []
        if self._horizontal_flips:
            data_augments.append((None, horizontal_flip))
        if self._vertical_flips:
            data_augments.append((None, vertical_flip))
        if self._rot_90:
            data_augments.append((None, rotate_90, rotate_180, rotate_270))
        data_augments = [DataAugmenter(augs) for augs in itertools.product(*data_augments)
                         if not all(e is None for e in augs)]

        # Create a new image sampler to handle shuffling and unify image sources
        # We want the sampler to loop
        image_sampler = training.image_sampler.ImageSampler(image_sources=image_sources, augmenters=data_augments,
                                                            loop=True,
                                                            default_validation_fraction=self._validation_fraction)

        # Start training!
        epoch_length = image_sampler.num_training if self._use_image_source_length else self._epoch_length
        image_sampler.begin(shuffle=False)
        vision_system_trainee.start_training(num_images=self._num_epochs * epoch_length)
        for epoch_num in range(self._num_epochs):
            # First, re-shuffle the images
            image_sampler.shuffle()

            # Train with the training set
            for image_index in range(epoch_length):
                vision_system_trainee.train_with_image(image_sampler.get(image_index),
                                                       index=image_index + epoch_length * epoch_num)
            # Validate with the validation set
            vision_system_trainee.start_validation(num_validation_images=image_sampler.num_validation)
            for image_index in range(image_sampler.num_validation):
                vision_system_trainee.validate_with_image(image_sampler.get_validation(image_index))
            vision_system_trainee.finish_validation()

        return vision_system_trainee.finish_training(
            trainer_id=self.identifier,
            image_source_ids=[source.identifier for source in image_sources],
            training_settings={
                'num_epochs': self._num_epochs,
                'epoch_length': self._epoch_length,
                'use_source_length': self._use_image_source_length,
                'horizontal_flips': self._horizontal_flips,
                'vertical_flips': self._vertical_flips,
                'rot_90': self._rot_90,
                'validation_fraction': self._validation_fraction
            })

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['num_epochs'] = self._num_epochs
        serialized['epoch_length'] = self._epoch_length
        serialized['use_source_length'] = self._use_image_source_length
        serialized['horizontal_flips'] = self._horizontal_flips
        serialized['vertical_flips'] = self._vertical_flips
        serialized['rot_90'] = self._rot_90
        serialized['validation_fraction'] = self._validation_fraction
        serialized['image_sources'] = [source.identifier for source in self._image_sources]
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        """
        Deserialize the trainer.
        Like image_collection, this gets special logic to deserialize the image sources contained within.
        :param serialized_representation: The serialized epoch trainer
        :param db_client: The database client, we're going to use this
        :param kwargs: keyword arguments to be passed to the constructor, we'll override some of these
        :return: The constructed EpochTrainer object
        """
        if 'num_epochs' in serialized_representation:
            kwargs['num_epochs'] = serialized_representation['num_epochs']
        if 'epoch_length' in serialized_representation:
            kwargs['epoch_length'] = serialized_representation['epoch_length']
        if 'use_source_length' in serialized_representation:
            kwargs['use_source_length'] = serialized_representation['use_source_length']
        if 'horizontal_flips' in serialized_representation:
            kwargs['horizontal_flips'] = serialized_representation['horizontal_flips']
        if 'vertical_flips' in serialized_representation:
            kwargs['vertical_flips'] = serialized_representation['vertical_flips']
        if 'rot_90' in serialized_representation:
            kwargs['rot_90'] = serialized_representation['rot_90']
        if 'validation_fraction' in serialized_representation:
            kwargs['validation_fraction'] = serialized_representation['validation_fraction']
        if 'image_sources' in serialized_representation:
            s_image_sources = db_client.image_source_collection.find({
                '_id': {'$in': serialized_representation['image_sources']}
            })
            kwargs['image_sources'] = [db_client.deserialize_entity(s_image_source)
                                       for s_image_source in s_image_sources]
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def create_serialized(cls, num_epochs=2000, use_source_length=False, epoch_length=1000, image_sources=None,
                          horizontal_flips=False, vertical_flips=False, rot_90=False, validation_fraction=0.3):
        """
        Create an already serialized epoch trainer.
        This saves us having to load all the image sources when trying to build a new trainer for the database.
        :param num_epochs: The number of training epochs
        :param use_source_length: Should each epoch be based on the size of the image source.
        If true, will loop through the entire image source exactly once per epoch. Default false.
        :param epoch_length: How many images in each epoch, assuming we're not basing it off the image source
        :param image_sources: The set of initial image sources to use. More can be added.
        :param horizontal_flips: Augment data with horizontal flips? Default False
        :param vertical_flips: Augment data with vertical flips? Default False
        :param rot_90: Augment data with rotations of multiples of 90 degrees. Default False
        :param validation_fraction: The fraction of data to use in the validation set. Default 0.3 (30%)
        :return:
        """
        return {
            '_type': cls.__module__ + '.' + cls.__name__,
            'num_epochs': num_epochs,
            'epoch_length': epoch_length,
            'use_source_length': use_source_length,
            'horizontal_flips': horizontal_flips,
            'vertical_flips': vertical_flips,
            'rot_90': rot_90,
            'validation_fraction': validation_fraction,
            'image_sources': list(image_sources)
        }


class DataAugmenter:
    """
    A simple helper to perform several different stacked augmentations
    """
    def __init__(self, augments):
        self.augments = (aug for aug in augments if callable(aug))

    def __call__(self, *args, **kwargs):
        if len(args) >= 1:
            image = args[0]
        elif 'image' in kwargs:
            image = kwargs['image']
        else:
            raise TypeError("Missing 1 positional argument 'image'")
        return self.augment(image)

    def augment(self, image):
        for augment in self.augments:
            image = augment(image)
        return image


def horizontal_flip(image):
    return core.image.Image(
        data=np.fliplr(image.data),
        metadata=image.metadata.clone(
            camera_pose=image.camera_pose,
            labelled_objects=(imeta.LabelledObject(
                class_names=obj.class_names,
                bounding_box=(
                    image.metadata.width - obj.bounding_box[0] - obj.bounding_box[2],
                    obj.bounding_box[1],
                    obj.bounding_box[2],
                    obj.bounding_box[3]
                ),
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in image.metadata.labelled_objects)),
        additional_metadata=image.additional_metadata,
        depth_data=np.fliplr(image.depth_data) if image.depth_data is not None else None,
        labels_data=np.fliplr(image.labels_data) if image.labels_data is not None else None,
        world_normals_data=np.fliplr(image.world_normals_data) if image.world_normals_data is not None else None
    )


def vertical_flip(image):
    return core.image.Image(
        data=np.flipud(image.data),
        metadata=image.metadata.clone(
            camera_pose=image.camera_pose,
            labelled_objects=(imeta.LabelledObject(
                class_names=obj.class_names,
                bounding_box=(
                    obj.bounding_box[0],
                    image.metadata.height - obj.bounding_box[1] - obj.bounding_box[3],
                    obj.bounding_box[2],
                    obj.bounding_box[3]
                ),
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in image.metadata.labelled_objects)),
        additional_metadata=image.additional_metadata,
        depth_data=np.flipud(image.depth_data) if image.depth_data is not None else None,
        labels_data=np.flipud(image.labels_data) if image.labels_data is not None else None,
        world_normals_data=np.flipud(image.world_normals_data) if image.world_normals_data is not None else None
    )


def rotate_90(image):
    return core.image.Image(
        data=np.rot90(image.data, k=1),
        metadata=image.metadata.clone(
            camera_pose=image.camera_pose,
            labelled_objects=(imeta.LabelledObject(
                class_names=obj.class_names,
                bounding_box=(
                    obj.bounding_box[1],
                    image.metadata.width - obj.bounding_box[0] - obj.bounding_box[2],
                    obj.bounding_box[3],
                    obj.bounding_box[2]
                ),
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in image.metadata.labelled_objects)),
        additional_metadata=image.additional_metadata,
        depth_data=np.rot90(image.depth_data, k=1) if image.depth_data is not None else None,
        labels_data=np.rot90(image.labels_data, k=1) if image.labels_data is not None else None,
        world_normals_data=np.rot90(image.world_normals_data, k=1) if image.world_normals_data is not None else None
    )


def rotate_180(image):
    return core.image.Image(
        data=np.rot90(image.data, k=2),
        metadata=image.metadata.clone(
            camera_pose=image.camera_pose,
            labelled_objects=(imeta.LabelledObject(
                class_names=obj.class_names,
                bounding_box=(
                    image.metadata.width - obj.bounding_box[0] - obj.bounding_box[2],
                    image.metadata.height - obj.bounding_box[1] - obj.bounding_box[3],
                    obj.bounding_box[2],
                    obj.bounding_box[3]
                ),
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in image.metadata.labelled_objects)),
        additional_metadata=image.additional_metadata,
        depth_data=np.rot90(image.depth_data, k=2) if image.depth_data is not None else None,
        labels_data=np.rot90(image.labels_data, k=2) if image.labels_data is not None else None,
        world_normals_data=np.rot90(image.world_normals_data, k=2) if image.world_normals_data is not None else None
    )


def rotate_270(image):
    return core.image.Image(
        data=np.rot90(image.data, k=3),
        metadata=image.metadata.clone(
            camera_pose=image.camera_pose,
            labelled_objects=(imeta.LabelledObject(
                class_names=obj.class_names,
                bounding_box=(
                    image.metadata.height - obj.bounding_box[1] - obj.bounding_box[3],
                    obj.bounding_box[0],
                    obj.bounding_box[3],
                    obj.bounding_box[2]
                ),
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in image.metadata.labelled_objects)),
        additional_metadata=image.additional_metadata,
        depth_data=np.rot90(image.depth_data, k=3) if image.depth_data is not None else None,
        labels_data=np.rot90(image.labels_data, k=3) if image.labels_data is not None else None,
        world_normals_data=np.rot90(image.world_normals_data, k=3) if image.world_normals_data is not None else None
    )
