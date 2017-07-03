import config.global_configuration as global_conf
import database.client

import simulation.combinatorial_sampling_controller as sample_controller


def main():
    """
    Import existing datasets into the database
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    sample_controller.generate_dataset(
        db_client=db_client,
        sim_config={
            'provide_rgb': True,
            'provide_depth': False,
            'provide_labels': True,
            'provide_world_normals': False,
        },
        x_samples=range(20, 271, 50),
        y_samples=range(-40, -221, -36),
        z_samples=range(80, 181, 20),
        roll_samples=range(-180, 181, 120),
        pitch_samples=range(-60, 61, 60),
        yaw_samples=range(-180, 181, 90),
        fov_samples=range(30, 91, 30),
        aperture_samples=(2.2, 22, 120)
    )


if __name__ == '__main__':
    main()
