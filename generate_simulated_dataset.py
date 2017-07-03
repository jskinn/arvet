import config.global_configuration as global_conf
import database.client

import simulation.combinatorial_sampling_controller as sample_controller
import simulation.unrealcv.unrealcv_simulator as ue_sim
import dataset.image_collection_builder as collection_builder


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
        x_samples=range(20, 271, 50),
        y_samples=range(-40, -221, -36),
        z_samples=range(80, 181, 20),
        roll_samples=range(-180, 181, 120),
        pitch_samples=range(-60, 61, 60),
        yaw_samples=range(-180, 181, 90),
        fov_samples=range(30, 91, 30),
        aperture_samples=(2.2, 22, 120)
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
