import numpy as np

import config.global_configuration as global_conf
import database.client
import dataset.image_collection_builder as collection_builder
import simulation.controllers.combinatorial_sampling_controller as sample_controller
import simulation.unrealcv.unrealcv_simulator as ue_sim
import util.transform as tf


def image_filter(image):
    """
    Function passed to the image collection builder
    used to determine if a given image should be in the dataset
    :param image: The potential image to add
    :return: True iff the image should be included in the generated dataset
    """
    return len(image.metadata.labelled_objects) > 0


def main():
    """
    Use UnrealCV to generate an image dataset
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    controller = sample_controller.CombinatorialSampleController(
        x_samples=range(20, 271, 50),   # 5 samples
        y_samples=range(40, 221, 36),   # 5 samples
        z_samples=range(80, 181, 20),   # 5 samples
        roll_samples=(-np.pi/2, -np.pi/4, -np.pi/6, 0, np.pi/6, np.pi/4, np.pi/2),  # 6 samples
        pitch_samples=np.linspace(-np.pi/2, np.pi/2, 7),    # 7 samples
        yaw_samples=np.linspace(-np.pi, np.pi, 7),          # 7 samples
        fov_samples=(30, 60, 90),
        aperture_samples=(2.2, 22, 120),
        subject_pose=tf.Transform(location=(130.00, 140.00, 70.37))
    )
    simulator = ue_sim.UnrealCVSimulator(controller, config={
        'provide_rgb': True,
        'provide_depth': False,
        'provide_labels': True,
        'provide_world_normals': False,
    })
    builder = collection_builder.ImageCollectionBuilder(db_client)
    builder.add_from_image_source(simulator, filter_function=image_filter)
    builder.save()


if __name__ == '__main__':
    main()
