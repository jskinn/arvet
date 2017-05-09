

def train_system_with_sources(system_trainer, image_sources):
    """
    Train a system with some image sources.
    
    :param system_trainer: The system trainer which can produce the trained system
    :param image_sources: The image sources with which to train the system
    :return: A TrainedVisionSystem which has been trained on images from the given image sources
    """
    for image_source in image_sources:
        if system_trainer.is_image_source_appropriate(image_source):
            system_trainer.start_training(image_source)
            image_source.begin()
            while not image_source.is_complete():
                system_trainer.train_with_image(image_source.get_next_image())
    return system_trainer.finish_training()
