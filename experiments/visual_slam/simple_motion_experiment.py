# Copyright (c) 2017, John Skinner
import typing
import os
import bson
import logging
import numpy as np
import util.database_helpers as dh
import util.dict_utils as du
import util.transform as tf
import database.client
import core.image_source
import core.sequence_type
import core.benchmark
import metadata.image_metadata as imeta
import batch_analysis.experiment
import batch_analysis.task_manager
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.ate.absolute_trajectory_error as ate
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import benchmarks.tracking.tracking_benchmark as tracking_benchmark
import simulation.unrealcv.unrealcv_simulator as uecv_sim
import simulation.controllers.trajectory_follow_controller as follow_cont


class SimpleMotionExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None, orbslam_systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmark_rpe=None, benchmark_ate=None, benchmark_trajectory_drift=None, benchmark_tracking=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param libviso_system:
        :param simulators:
        :param trajectory_groups:
        :param benchmark_rpe:
        :param benchmark_ate:
        :param benchmark_trajectory_drift:
        :param id_:
        """
        super().__init__(id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)
        # Systems
        self._libviso_system = libviso_system
        self._orbslam_systems = orbslam_systems if orbslam_systems is not None else {}

        # Image sources
        self._simulators = simulators if simulators is not None else {}
        self._trajectory_groups = trajectory_groups if trajectory_groups is not None else {}

        # Benchmarks
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_ate = benchmark_ate
        self._benchmark_trajectory_drift = benchmark_trajectory_drift
        self._benchmark_tracking = benchmark_tracking

    def do_imports(self, task_manager: batch_analysis.task_manager.TaskManager,
                   db_client: database.client.DatabaseClient):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
        # --------- SIMULATORS -----------
        # Add simulators explicitly, they have different metadata, so we can't just search
        for exe, world_name, environment_type, light_level, time_of_day in [
            (
                    '/media/john/Storage/simulators/CorridorWorld/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'CorridorWorld', imeta.EnvironmentType.OUTDOOR_LANDSCAPE, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            )
        ]:
            if world_name not in self._simulators:
                simulator_id = dh.add_unique(db_client.image_source_collection, uecv_sim.UnrealCVSimulator(
                    executable_path=exe,
                    world_name=world_name,
                    environment_type=environment_type,
                    light_level=light_level,
                    time_of_day=time_of_day
                ))
                self._simulators[world_name] = simulator_id
                self._set_property('simulators.{0}'.format(world_name), simulator_id)

        # --------- TRAJECTORY GROUPS -----------

        for name, path in [
            ('forwards', get_forwards_trajectory()),
            ('upwards', get_upward_trajectory()),
            ('left', get_left_trajectory()),
            ('on the spot roll', get_on_the_spot_roll_trajectory()),
            ('on the spot pitch', get_on_the_spot_pitch_trajectory()),
            ('on the spot yaw', get_on_the_spot_yaw_trajectory()),
            ('circle roll', get_circle_roll_trajectory()),
            ('circle pitch', get_circle_pitch_trajectory()),
            ('circle yaw', get_circle_yaw_trajectory()),
        ]:
            if name not in self._trajectory_groups:
                # First, create the trajectory follow controller with the desired trajectory
                controller = follow_cont.TrajectoryFollowController(
                    trajectory=path,
                    trajectory_source='custom {0}'.format(name),
                    sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL)
                controller_id = dh.add_unique(db_client.image_source_collection, controller)

                # Then create a trajectory group for it
                self._trajectory_groups[name] = TrajectoryGroup(
                    name=name, controller_id=controller_id,
                    simulators={'CorridorWorld': self._simulators['CorridorWorld']})
                self._set_property('trajectory_groups.{0}'.format(name), self._trajectory_groups[name].serialize())
        for group in self._trajectory_groups.values():
            if group.do_imports(task_manager, db_client):
                self._set_property('trajectory_groups.{0}'.format(group.name), group.serialize())

        # --------- SYSTEMS -----------
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)

        # ORBSLAM2 - Create orbslam systems in each sensor mode
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}:
            name = 'ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()).replace('.', '-')
            vocab_path = os.path.join('systems', 'slam', 'ORBSLAM2', 'ORBvoc.txt')
            if name not in self._orbslam_systems and os.path.isfile(vocab_path):
                orbslam_id = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
                    settings={
                        'ORBextractor': {'nFeatures': 1500}
                    }, resolution=(752, 480)
                ))
                self._orbslam_systems[name] = orbslam_id
                self._set_property('orbslam_systems.{}'.format(name), orbslam_id)

        # --------- BENCHMARKS -----------
        # Create and store the benchmarks for camera trajectories
        # Just using the default settings for now
        if self._benchmark_rpe is None:
            self._benchmark_rpe = dh.add_unique(db_client.benchmarks_collection, rpe.BenchmarkRPE(
                max_pairs=10000,
                fixed_delta=False,
                delta=1.0,
                delta_unit='s',
                offset=0,
                scale_=1))
            self._set_property('benchmark_rpe', self._benchmark_rpe)
        if self._benchmark_ate is None:
            self._benchmark_ate = dh.add_unique(db_client.benchmarks_collection, ate.BenchmarkATE(
                offset=0,
                max_difference=0.2,
                scale=1))
            self._set_property('benchmark_ate', self._benchmark_ate)
        if self._benchmark_trajectory_drift is None:
            self._benchmark_trajectory_drift = dh.add_unique(
                db_client.benchmarks_collection,
                traj_drift.BenchmarkTrajectoryDrift(
                    segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
                    step_size=10
                ))
            self._set_property('benchmark_trajectory_drift', self._benchmark_trajectory_drift)
        if self._benchmark_tracking is None:
            self._benchmark_tracking = dh.add_unique(db_client.benchmarks_collection,
                                                     tracking_benchmark.TrackingBenchmark(initializing_is_lost=True))
            self._set_property('benchmark_tracking', self._benchmark_tracking)

    def schedule_tasks(self, task_manager: batch_analysis.task_manager.TaskManager,
                       db_client: database.client.DatabaseClient):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :param task_manager:
        :param db_client:
        :return:
        """
        # Group everything up
        # All systems
        systems = [self._libviso_system] + list(self._orbslam_systems.values())
        # All image datasets
        datasets = set()
        for group in self._trajectory_groups.values():
            datasets = datasets | group.get_all_dataset_ids()
        # All benchmarks
        benchmarks = [self._benchmark_rpe, self._benchmark_ate,
                      self._benchmark_trajectory_drift, self._benchmark_tracking]

        # Schedule all combinations of systems with the generated datasets
        self.schedule_all(task_manager=task_manager,
                          db_client=db_client,
                          systems=systems,
                          image_sources=datasets,
                          benchmarks=benchmarks)

    def plot_results(self, db_client: database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_trajectories(db_client)
        #self._plot_relative_pose_error(db_client)

    def _plot_trajectories(self, db_client: database.client.DatabaseClient):
        """
        Plot the ground-truth and computed trajectories for each system for each trajectory.
        This is important for validation
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        logging.getLogger(__name__).info("Plotting trajectories...")
        # Map system ids and simulator ids to printable names
        simulator_names = {v: k for k, v in self._simulators.items()}
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)

        for trajectory_group in self._trajectory_groups.values():
            # Collect all the image sources for this trajectory group
            image_sources = {}
            for simulator_id, dataset_id in trajectory_group.generated_datasets.items():
                if simulator_id in simulator_names:
                    image_sources[simulator_names[simulator_id]] = dataset_id
                else:
                    image_sources[simulator_id] = dataset_id

            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in systems.items():
                for dataset_name, dataset_id in image_sources.items():
                    trial_result_id = self.get_trial_result(system_id, dataset_id)
                    if trial_result_id is not None:
                        label = "{0} on {1}".format(system_name, dataset_name)
                        trial_results[label] = trial_result_id

            # Make sure we have at least one result to plot
            if len(trial_results) >= 1:
                figure = pyplot.figure(figsize=(14, 10), dpi=80)
                figure.suptitle("Computed trajectories for {0}".format(trajectory_group.name))
                ax = figure.add_subplot(111, projection='3d')
                ax.set_xlabel('x-location')
                ax.set_ylabel('y-location')
                ax.set_zlabel('z-location')

                oax = figure.add_subplot(111, projection='3d')
                oax.set_xlabel('x-location')
                oax.set_ylabel('y-location')
                oax.set_zlabel('z-location')

                added_ground_truth = False
                lowest = -0.001
                highest = 0.001
                cmap = pyplot.get_cmap('Set1')
                colour_index = 0

                # For each trial result
                for label, trial_result_id in trial_results.items():
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None:
                        if trial_result.success:
                            if not added_ground_truth:
                                trajectory = trial_result.get_ground_truth_camera_poses()
                                lower, upper = plot_trajectory(ax, trajectory, 'ground truth trajectory', style='k--')
                                lowest = min(lowest, lower)
                                highest = max(highest, upper)
                                added_ground_truth = True
                            trajectory = trial_result.get_computed_camera_poses()
                            lower, upper = plot_trajectory(ax, trajectory, label=label, style='-')
                            plot_forward(oax, trajectory, label=label, colors=[cmap(colour_index / 9, alpha=0.5)])
                            lowest = min(lowest, lower)
                            highest = max(highest, upper)
                            colour_index += 1
                        else:
                            print("Got failed trial: {0}".format(trial_result.reason))

                logging.getLogger(__name__).info("... plotted trajectories for {0}".format(trajectory_group.name))
                ax.legend()
                ax.set_xlim(lowest, highest)
                ax.set_ylim(lowest, highest)
                ax.set_zlim(lowest, highest)

                oax.legend()
                oax.set_xlim(lowest, highest)
                oax.set_ylim(lowest, highest)
                oax.set_zlim(lowest, highest)

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()

    def _plot_relative_pose_error(self, db_client: database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        logging.getLogger(__name__).info("Plotting relative pose error...")
        # Map system ids and simulator ids to printable names
        simulator_names = {v: k for k, v in self._simulators.items()}
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)

        for trajectory_group in self._trajectory_groups.values():
            # Collect all the image sources for this trajectory group
            image_sources = {}
            for simulator_id, dataset_id in trajectory_group.generated_datasets.items():
                if simulator_id in simulator_names:
                    image_sources[simulator_names[simulator_id]] = dataset_id
                else:
                    image_sources[simulator_id] = dataset_id

            if len(image_sources) <= 1:
                # Skip where we've only got one image source, it's not interesting.
                continue

            # Collect the results for each image source in this group
            results = {}
            for system_name, system_id in systems.items():
                for dataset_name, dataset_id in image_sources.items():
                    trial_result_id = self.get_trial_result(system_id, dataset_id)
                    if trial_result_id is not None:
                        result_id = self.get_benchmark_result(trial_result_id, self._benchmark_rpe)
                        if result_id is not None:
                            label = "{0} on {1}".format(system_name, dataset_name)
                            results[label] = result_id

            if len(results) > 1:
                figure = pyplot.figure(figsize=(14, 10), dpi=80)
                figure.suptitle("Relative pose error for {0}".format(trajectory_group.name))
                ax = figure.add_subplot(111)
                ax.set_xlabel('time')
                ax.set_ylabel('relative pose error')

                # For each trial result
                for label, result_id in results.items():
                    result = dh.load_object(db_client, db_client.results_collection, result_id)
                    if result is not None:
                        if result.success:
                            x = []
                            y = []
                            times = sorted(result.translational_error.keys())
                            for time in times:
                                error = result.translational_error[time]
                                if error < 100:
                                    x.append(time - times[0])
                                    y.append(error)
                            ax.plot(x, y, '-', label=label, alpha=0.7)
                        else:
                            print("Got failed result: {0}".format(result.reason))

                logging.getLogger(__name__).info("... plotted rpe for {0}".format(trajectory_group.name))
                ax.legend()
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()

    def serialize(self):
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'experiments:visual_slam:SimpleMotionExperiment', 1)

        # Systems
        serialized['libviso'] = self._libviso_system
        serialized['orbslam_systems'] = self._orbslam_systems

        # Image Sources
        serialized['simulators'] = self._simulators
        serialized['trajectory_groups'] = {str(name): group.serialize()
                                           for name, group in self._trajectory_groups.items()}

        # Benchmarks
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_ate'] = self._benchmark_ate
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        serialized['benchmark_tracking'] = self._benchmark_tracking

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        update_schema(serialized_representation, db_client)

        # Systems
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_systems' in serialized_representation:
            kwargs['orbslam_systems'] = serialized_representation['orbslam_systems']

        # Generated datasets
        if 'simulators' in serialized_representation:
            kwargs['simulators'] = serialized_representation['simulators']
        if 'trajectory_groups' in serialized_representation:
            kwargs['trajectory_groups'] = {name: TrajectoryGroup.deserialize(s_group)
                                           for name, s_group in
                                           serialized_representation['trajectory_groups'].items()}

        # Benchmarks
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_ate' in serialized_representation:
            kwargs['benchmark_ate'] = serialized_representation['benchmark_ate']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        if 'benchmark_tracking' in serialized_representation:
            kwargs['benchmark_tracking'] = serialized_representation['benchmark_tracking']

        return super().deserialize(serialized_representation, db_client, **kwargs)


def plot_trajectory(axis, trajectory, label, style='-'):
    """
    Simple helper to plot a trajectory on a 3D axis.
    Will normalise the trajectory to start at (0,0,0) and facing down the x axis,
    that is, all poses are relative to the first one.
    :param axis: The axis on which to plot
    :param trajectory: A map of timestamps to camera poses
    :param label: The label for the series
    :param style: The line style to use for the trajectory. Lets us distinguish virtual and real world results.
    :return: The minimum and maximum coordinate values, for axis sizing.
    """
    x = []
    y = []
    z = []
    max_point = 0
    min_point = 0
    times = sorted(trajectory.keys())
    first_pose = None
    for timestamp in times:
        pose = trajectory[timestamp]
        if first_pose is None:
            first_pose = pose
            x.append(0)
            y.append(0)
            z.append(0)
        else:
            pose = first_pose.find_relative(pose)
            max_point = max(max_point, pose.location[0], pose.location[1], pose.location[2])
            min_point = min(min_point, pose.location[0], pose.location[1], pose.location[2])
            x.append(pose.location[0])
            y.append(pose.location[1])
            z.append(pose.location[2])
    axis.plot(x, y, z, style, label=label, alpha=0.7)
    return min_point, max_point


def plot_forward(axis, trajectory, label, linestyles='solid', colors=None):
    x = []
    y = []
    z = []
    u = []
    v = []
    w = []
    times = sorted(trajectory.keys())
    first_pose = None
    prev_forward = (1, 0, 0)
    lowest = -0.001
    highest = 0.001
    for timestamp in times:
        pose = trajectory[timestamp]
        if first_pose is None:
            first_pose = pose
            x.append(0)
            y.append(0)
            z.append(0)
            u.append(1)
            v.append(0)
            w.append(0)
        else:
            pose = first_pose.find_relative(pose)
            forward = pose.forward
            forward = forward / np.linalg.norm(forward)
            if np.arccos(np.dot(forward, prev_forward)) < 0.95:
                prev_forward = forward
                lowest = min(lowest, pose.location[0], pose.location[1], pose.location[2])
                highest = max(highest, pose.location[0], pose.location[1], pose.location[2])
                x.append(pose.location[0])
                y.append(pose.location[1])
                z.append(pose.location[2])
                u.append(forward[0])
                v.append(forward[1])
                w.append(forward[2])
    axis.quiver(x, y, z, u, v, w, linestyles=linestyles, colors=colors, normalize=True, label=label,
                length=1)


def update_schema(serialized: dict, db_client: database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:SimpleMotionExperiment')
    if 'libviso' in serialized and not dh.check_reference_is_valid(db_client.system_collection, serialized['libviso']):
        del serialized['libviso']
    if 'orbslam_systems' in serialized:
        keys = list(serialized['orbslam_systems'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.system_collection, serialized['orbslam_systems'][key]):
                del serialized['orbslam_systems'][key]
    if 'simulators' in serialized:
        keys = list(serialized['simulators'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['simulators'][key]):
                del serialized['simulators'][key]
    if 'benchmark_rpe' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['benchmark_rpe']):
        del serialized['benchmark_rpe']
    if 'benchmark_ate' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['benchmark_ate']):
        del serialized['benchmark_ate']
    if 'benchmark_trajectory_drift' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['benchmark_trajectory_drift']):
        del serialized['benchmark_trajectory_drift']
    if 'benchmark_tracking' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['benchmark_tracking']):
        del serialized['benchmark_tracking']


def get_forwards_trajectory() -> typing.Mapping[float, tf.Transform]:
    """
    Get a simple linear trajectory, moving in a straight line forwards from -10m to 10m
    :return: A map from timestamps to camera poses, with which we can create a controller.
    """
    framerate = 30  # frames per second
    speed = 1.2     # meters per second (walking pace)
    return {time: tf.Transform(location=(time * speed - 10, 0, 0)) for time in np.arange(0, 20 / speed, 1 / framerate)}


def get_upward_trajectory() -> typing.Mapping[float, tf.Transform]:
    """
    Get a simple linear trajectory, moving in a straight line upwards from -10m to 10m
    :return: A map from timestamps to camera poses, with which we can create a controller.
    """
    framerate = 30  # frames per second
    speed = 1.2     # meters per second
    return {time: tf.Transform(location=(0, 0, time * speed - 10)) for time in np.arange(0, 20 / speed, 1 / framerate)}


def get_left_trajectory() -> typing.Mapping[float, tf.Transform]:
    """
    Get a simple linear trajectory, moving in a straight line left from -10m to 10m
    :return: A map from timestamps to camera poses, with which we can create a controller.
    """
    framerate = 30  # frames per second
    speed = 1.2     # meters per second
    return {time: tf.Transform(location=(0, time * speed - 10, 0)) for time in np.arange(0, 20 / speed, 1 / framerate)}


def get_on_the_spot_roll_trajectory():
    """
    Get a trajectory that turns a circle on the spot. Pure yaw, no translational component.
    :return:
    """
    framerate = 30  # frames per second
    angular_vel = np.pi / 36   # radians per second
    return {time: tf.Transform(rotation=(time * angular_vel, 0, 0))
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


def get_on_the_spot_pitch_trajectory():
    """
    Get a trajectory that turns a circle on the spot. Pure yaw, no translational component.
    :return:
    """
    framerate = 30  # frames per second
    angular_vel = np.pi / 36   # radians per second
    return {time: tf.Transform(rotation=(0, 0, time * angular_vel))
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


def get_on_the_spot_yaw_trajectory():
    """
    Get a trajectory that turns a circle on the spot. Pure yaw, no translational component.
    :return:
    """
    framerate = 30  # frames per second
    angular_vel = np.pi / 36   # radians per second
    return {time: tf.Transform(rotation=(0, 0, time * angular_vel))
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


def get_circle_roll_trajectory():
    """
    Get a trajectory that moves in a 1m diameter circle, facing outwards
    :return:
    """
    framerate = 30            # frames per second
    angular_vel = np.pi / 36  # radians per second
    relative_pose = tf.Transform(location=(0, 0, 0.5), rotation=(0, 0, 0))
    return {time: (tf.Transform(rotation=(time * angular_vel, 0, 0))).find_independent(relative_pose)
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


def get_circle_pitch_trajectory():
    """
    Get a trajectory that moves in a 1m diameter circle, facing outwards
    :return:
    """
    framerate = 30            # frames per second
    angular_vel = np.pi / 36  # radians per second
    relative_pose = tf.Transform(location=(0.5, 0, 0), rotation=(0, 0, 0))
    return {time: (tf.Transform(rotation=(0, time * angular_vel, 0))).find_independent(relative_pose)
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


def get_circle_yaw_trajectory():
    """
    Get a trajectory that moves in a 1m diameter circle, facing outwards
    :return:
    """
    framerate = 30            # frames per second
    angular_vel = np.pi / 36  # radians per second
    relative_pose = tf.Transform(location=(0.5, 0, 0), rotation=(0, 0, 0))
    return {time: (tf.Transform(rotation=(0, 0, time * angular_vel))).find_independent(relative_pose)
            for time in np.arange(0, 2 * np.pi / angular_vel, 1 / framerate)}


class TrajectoryGroup:
    """
    A Trajectory Group is a helper structure to manage image datasets grouped by camera trajectory.
    In this experiment, it is created from a single reference dataset,
    and produces many synthetic datasets with the same camera trajectory.

    For convenience, it serializes and deserialzes as a group.
    """

    def __init__(self, name: str, controller_id: bson.ObjectId,
                 simulators: dict = None, generated_datasets: dict = None):
        self.name = name
        self.simulators = simulators if simulators is not None else {}

        self.controller_id = controller_id
        self.generated_datasets = generated_datasets if generated_datasets is not None else {}

    def get_all_dataset_ids(self) -> set:
        """
        Get all the datasets in this group, as a set
        :return:
        """
        return set(self.generated_datasets.values())

    def do_imports(self, task_manager: batch_analysis.task_manager.TaskManager,
                   db_client: database.client.DatabaseClient) -> bool:
        """
        Do imports and dataset generation for this trajectory group.
        Will create a controller, and then generate reduced quality synthetic datasets.
        :param task_manager:
        :param db_client:
        :return: True if part of the group has changed, and it needs to be re-saved
        """
        changed = False
        # For each combination of simulator id and config, generate a dataset
        for sim_name, simulator_id in self.simulators.items():
            # Schedule generation of quality variations that don't exist yet
            if sim_name not in self.generated_datasets:
                generate_dataset_task = task_manager.get_generate_dataset_task(
                    controller_id=self.controller_id,
                    simulator_id=simulator_id,
                    simulator_config={
                        # Simulation execution config
                        'stereo_offset': 0.48,  # Same as KITTI data
                        'provide_rgb': True,
                        'provide_depth': True,
                        'provide_labels': False,
                        'provide_world_normals': False,

                        # Simulator camera settings, be similar to the reference dataset
                        'resolution': {'width': 752, 'height': 480},
                        'fov': 81.4,    # Same as KITTI data
                        'depth_of_field_enabled': False,
                        'focus_distance': None,
                        'aperture': 2.2,

                        # Quality settings - Maximum quality
                        'lit_mode': True,
                        'texture_mipmap_bias': 0,
                        'normal_maps_enabled': True,
                        'roughness_enabled': True,
                        'geometry_decimation': 0,
                        'depth_noise_quality': 1,

                        # Simulation server config
                        'host': 'localhost',
                        'port': 9000,
                    },
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if generate_dataset_task.is_finished:
                    self.generated_datasets[sim_name] = generate_dataset_task.result
                    changed = True
                else:
                    task_manager.do_task(generate_dataset_task)
        return changed

    def serialize(self) -> dict:
        return {
            'name': self.name,
            'simulators': self.simulators,
            'controller_id': self.controller_id,
            'generated_datasets': self.generated_datasets
        }

    @classmethod
    def deserialize(cls, serialized_representation: dict) -> 'TrajectoryGroup':
        return cls(
            name=serialized_representation['name'],
            simulators=serialized_representation['simulators'],
            controller_id=serialized_representation['controller_id'],
            generated_datasets=serialized_representation['generated_datasets']
        )
