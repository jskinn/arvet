

def run_system_with_source(system, image_source):
    """
    Run a given vision system with a given image source.
    This is the structure for how image sources and vision systems should be interacted with.
    Both should already be set up and configured.
    :param system: The system to run.
    :param image_source: The image source to get images from
    :return: The TrialResult storing the results of the run. Save it to the database, or None if there's a problem.
    """
    if system.is_image_source_appropriate(image_source):
        system.start_trial()
        image_source.begin()
        while not image_source.is_complete():
            image, timstamp = image_source.get_next_image()
            system.process_image(image)
        return system.finish_trial()
    return None
