# Copyright (c) 2017, John Skinner
import os
import bson
import logging
import util.database_helpers as dh
import database.client
import core.image_source
import core.sequence_type
import core.benchmark
import metadata.image_metadata as imeta
import batch_analysis.experiment
import batch_analysis.task_manager
import systems.visual_odometry.libviso2.libviso2 as libviso2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.ate.absolute_trajectory_error as ate
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import simulation.unrealcv.unrealcv_simulator as uecv_sim
import simulation.controllers.trajectory_follow_controller as follow_cont


class VisualOdometryExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None,
                 simulators=None, trajectory_groups=None,
                 benchmark_rpe=None, benchmark_ate=None, benchmark_trajectory_drift=None,
                 trial_map=None, result_map=None, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param libviso_system:
        :param simulators:
        :param trajectory_groups:
        :param benchmark_rpe:
        :param benchmark_ate:
        :param benchmark_trajectory_drift:
        :param trial_map:
        :param result_map:
        :param id_:
        """
        super().__init__(id_=id_)
        # Systems
        self._libviso_system = libviso_system

        # Image sources
        self._simulators = simulators if simulators is not None else {}
        self._trajectory_groups = trajectory_groups if trajectory_groups is not None else {}

        # Benchmarks
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_ate = benchmark_ate
        self._benchmark_trajectory_drift = benchmark_trajectory_drift

        # Trials and results
        self._trial_map = trial_map if trial_map is not None else {}
        self._result_map = result_map if result_map is not None else {}
        self._placeholder_image_collections = {}

    def do_imports(self, task_manager, db_client):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
        # --------- REAL WORLD DATASETS -----------
        # Import KITTI dataset
        for sequence_num in range(11):
            task = task_manager.get_import_dataset_task(
                module_name='dataset.kitti.kitti_loader',
                path=os.path.expanduser(os.path.join('~', 'datasets', 'KITTI', 'dataset')),
                additional_args={
                    'sequence_number': sequence_num
                },
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='72:00:00'
            )
            if task.is_finished:
                self._trajectory_groups[task.result] = TrajectoryGroup(
                    name='KITTI trajectory {}'.format(sequence_num),
                    real_world_id=task.result
                )
                self._set_property('trajectory_groups.{0}'.format(task.result),
                                   self._trajectory_groups[task.result].serialize())
            else:
                task_manager.do_task(task)

        # --------- SIMULATORS -----------
        # Add simulators explicitly, they have different metadata, so we can't just search
        for exe, world_name, environment_type, light_level, time_of_day in [
            (
                '/media/john/Storage/simulators/BlockWorld/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                'BlockWorld', imeta.EnvironmentType.OUTDOOR_LANDSCAPE, imeta.LightingLevel.WELL_LIT, imeta.TimeOfDay.DAY
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
                self._set_property('simulators.{}'.format(world_name), simulator_id)

        # --------- TRAJECTORY GROUPS -----------
        for key, trajectory_group in self._trajectory_groups.items():
            changed = False
            # Add the simulators to the group, with particular config
            for simulator_id in self._simulators.values():
                changed |= trajectory_group.add_simulator(simulator_id, {
                    # Simulation execution config
                    'stereo_offset': 0.15,  # meters
                    'provide_rgb': True,
                    'provide_depth': False,
                    'provide_labels': False,
                    'provide_world_normals': False,

                    # Simulator camera settings, need to disable DOF because the autofocus doesn't work  in BlockWorld
                    'resolution': {'width': 640, 'height': 480},
                    'fov': 90,
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
                })
            changed |= trajectory_group.do_imports(task_manager, db_client)
            # If the group has had something change, push the changes
            if changed:
                self._set_property('trajectory_groups.{0}'.format(key), trajectory_group.serialize())

        # --------- SYSTEMS -----------
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)

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

    def schedule_tasks(self, task_manager, db_client):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :param task_manager:
        :param db_client:
        :return:
        """
        # Group everything up
        # All systems
        systems = [
            dh.load_object(db_client, db_client.system_collection, self._libviso_system)
        ]
        # All image datasets
        datasets = set()
        for group in self._trajectory_groups.values():
            datasets = datasets | group.get_all_dataset_ids()
        # All benchmarks
        benchmarks = [
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_rpe),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_ate),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_trajectory_drift)
        ]
        # Trial results will be collected as we go
        trial_results = set()

        # For each image dataset, run libviso with that dataset, and store the result in the trial map
        for image_source_id in datasets:
            image_source = dh.load_object(db_client, db_client.image_source_collection, image_source_id)
            for system in systems:
                if system.is_image_source_appropriate(image_source):
                    task = task_manager.get_run_system_task(
                        system_id=system.identifier,
                        image_source_id=image_source.identifier,
                        expected_duration='8:00:00',
                        memory_requirements='12GB'
                    )
                    if not task.is_finished:
                        task_manager.do_task(task)
                    else:
                        trial_results.add(task.result)
                        if self._libviso_system not in self._trial_map:
                            self._trial_map[system.identifier] = {}
                        self._trial_map[system.identifier][image_source_id] = task.result
                        self._set_property('trial_map.{0}.{1}'.format(system.identifier, image_source_id),
                                           task.result)

        # Benchmark trial results
        for trial_result_id in trial_results:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            for benchmark in benchmarks:
                if benchmark.is_trial_appropriate(trial_result):
                    task = task_manager.get_benchmark_task(
                        trial_result_id=trial_result.identifier,
                        benchmark_id=benchmark.identifier,
                        expected_duration='6:00:00',
                        memory_requirements='6GB'
                    )
                    if not task.is_finished:
                        task_manager.do_task(task)
                    else:
                        if trial_result_id not in self._result_map:
                            self._result_map[trial_result_id] = {}
                        self._result_map[trial_result_id][benchmark.identifier] = task.result
                        self._set_property('result_map.{0}.{1}'.format(trial_result_id, benchmark.identifier),
                                           task.result)

    def plot_results(self, db_client):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_trajectories(db_client)

    def _plot_trajectories(self, db_client):
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

        systems = {'LIBVISO 2': self._libviso_system}
        for system_name, system_id in systems.items():
            if system_id not in self._trial_map:
                continue    # No trials for this system, skip it
            for trajectory_group in self._trajectory_groups.values():
                figure = pyplot.figure(figsize=(14, 10), dpi=80)
                figure.suptitle("Computed trajectories for {0} on {1}".format(system_name, trajectory_group.name))
                ax = figure.add_subplot(111, projection='3d')
                ax.set_xlabel('x-location')
                ax.set_ylabel('y-location')
                ax.set_zlabel('z-location')
                ax.plot([0], [0], [0], 'ko', label='origin')
                lower_limit = 0
                upper_limit = 0

                image_sources = {'KITTI sequence': trajectory_group.real_world_dataset}
                for simulator_id, dataset_id in trajectory_group.generated_datasets.items():
                    image_sources[simulator_names[simulator_id]] = dataset_id

                # For each image source in this group
                for dataset_name, dataset_id in image_sources.items():
                    if dataset_id in self._trial_map[system_id]:
                        trial_result = dh.load_object(db_client, db_client.trials_collection,
                                                      self._trial_map[system_id][dataset_id])
                        if trial_result is not None:
                            minp, maxp = plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                                         dataset_name)
                            lower_limit = min(lower_limit, minp)
                            upper_limit = max(upper_limit, maxp)

                logging.getLogger(__name__).info("... plotted trajectories for {0}".format(trajectory_group.name))
                ax.legend()
                ax.set_xlim(lower_limit, upper_limit)
                ax.set_ylim(lower_limit, upper_limit)
                ax.set_zlim(lower_limit, upper_limit)
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

    def serialize(self):
        serialized = super().serialize()
        # Systems
        serialized['libviso'] = self._libviso_system

        # Image Sources
        serialized['simulators'] = self._simulators
        serialized['trajectory_groups'] = {str(max_id): group.serialize()
                                           for max_id, group in self._trajectory_groups.items()}

        # Benchmarks
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_ate'] = self._benchmark_ate
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift

        # Trials
        serialized['trial_map'] = {str(sys_id): {str(source_id): trial_id for source_id, trial_id in inner_map.items()}
                                   for sys_id, inner_map in self._trial_map.items()}
        serialized['result_map'] = {str(trial_id): {str(bench_id): res_id for bench_id, res_id in inner_map.items()}
                                    for trial_id, inner_map in self._result_map.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        # Systems
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']

        # Generated datasets
        if 'simulators' in serialized_representation:
            kwargs['simulators'] = serialized_representation['simulators']
        if 'trajectory_groups' in serialized_representation:
            kwargs['trajectory_groups'] = {bson.ObjectId(max_id): TrajectoryGroup.deserialize(s_group)
                                           for max_id, s_group in
                                           serialized_representation['trajectory_groups'].items()}

        # Benchmarks
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_ate' in serialized_representation:
            kwargs['benchmark_ate'] = serialized_representation['benchmark_ate']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']

        # Trials and results
        if 'trial_map' in serialized_representation:
            kwargs['trial_map'] = {bson.ObjectId(sys_id): {bson.ObjectId(source_id): trial_id
                                                           for source_id, trial_id in inner_map.items()}
                                   for sys_id, inner_map in serialized_representation['trial_map'].items()}
        if 'result_map' in serialized_representation:
            kwargs['result_map'] = {bson.ObjectId(trial_id): {bson.ObjectId(bench_id): res_id
                                                              for bench_id, res_id in inner_map.items()}
                                    for trial_id, inner_map in serialized_representation['result_map'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)


def plot_trajectory(axis, trajectory, label):
    """
    Simple helper to plot a trajectory on a 3D axis.
    Will normalise the trajectory to start at (0,0,0) and facing down the x axis,
    that is, all poses are relative to the first one.
    :param axis: The axis on which to plot
    :param trajectory: A map of timestamps to camera poses
    :param label: The label for the series
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
    axis.plot(x, y, z, '--', label=label)
    return min_point, max_point


class TrajectoryGroup:
    """
    A Trajectory Group is a helper structure to manage image datasets grouped by camera trajectory.
    In this experiment, it is created from a single real world dataset (A KITTI sequence),
    and produces many synthetic datasets with the same camera trajectory.

    For convenience, it serializes and deserialzes as a group.
    """

    def __init__(self,  name, real_world_id: bson.ObjectId, simulator_ids=None, simulator_configs=None,
                 controller_id=None, generated_datasets=None):
        self.name = name
        self.real_world_dataset = real_world_id
        self.simulator_ids = simulator_ids if simulator_ids is not None else []
        self.simulator_configs = simulator_configs if simulator_configs is not None else {}

        self.follow_controller_id = controller_id
        self.generated_datasets = generated_datasets if generated_datasets is not None else {}

    def add_simulator(self, simulator_id: bson.ObjectId, simulator_config: dict) -> bool:
        """
        Add a simulator to the group, from which a dataset will be generated
        :param simulator_id: The id of the simulator
        :param simulator_config: Configuration used to generate the dataset
        :return: True if that simulator was not already part of the group, so the group needs to be re-saved
        """
        if simulator_id not in self.simulator_ids:
            self.simulator_ids.append(simulator_id)
            self.simulator_configs[simulator_id] = simulator_config
            return True
        return False

    def get_all_dataset_ids(self) -> set:
        """
        Get all the datasets in this group, as a set
        :return:
        """
        return set(self.generated_datasets.values()) | {self.real_world_dataset}

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
        # First, make a follow controller for the base dataset if we don't have one.
        # This will be used to generate reduced-quality datasets following the same trajectory
        # as the root dataset
        if self.follow_controller_id is None:
            self.follow_controller_id = follow_cont.create_follow_controller(
                db_client, self.real_world_dataset, sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL)
            changed = True
        for simulator_id in self.simulator_ids:
            # Schedule generation of quality variations that don't exist yet
            if simulator_id not in self.generated_datasets:
                generate_dataset_task = task_manager.get_generate_dataset_task(
                    controller_id=self.follow_controller_id,
                    simulator_id=simulator_id,
                    simulator_config=self.simulator_configs[simulator_id],
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if generate_dataset_task.is_finished:
                    self.generated_datasets[simulator_id] = generate_dataset_task.result
                else:
                    task_manager.do_task(generate_dataset_task)
        return changed

    def serialize(self):
        return {
            'name': self.name,
            'real_world_dataset': self.real_world_dataset,
            'simulator_ids': self.simulator_ids,
            'simulator_configs': {str(id_): config for id_, config in self.simulator_configs.items()},
            'controller_id': self.follow_controller_id,
            'generated_datasets': {str(sim_id): dataset_id for sim_id, dataset_id in self.generated_datasets.items()}
        }

    @classmethod
    def deserialize(cls, serialized_representation):
        return cls(
            name=serialized_representation['name'],
            real_world_id=serialized_representation['real_world_dataset'],
            simulator_ids=serialized_representation['simulator_ids'],
            simulator_configs={bson.ObjectId(id_): config for id_, config
                               in serialized_representation['simulator_configs'].items()},
            controller_id=serialized_representation['controller_id'],
            generated_datasets={bson.ObjectId(sim_id): dataset_id for sim_id, dataset_id
                                in serialized_representation['generated_datasets'].items()}
        )
