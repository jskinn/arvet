# Copyright (c) 2017, John Skinner
import os
import bson
import logging
import util.database_helpers as dh
import util.dict_utils as du
import util.trajectory_helpers as traj_help
import util.unreal_transform as uetf
import database.client
import core.image_source
import core.sequence_type
import core.benchmark
import metadata.image_metadata as imeta
import batch_analysis.experiment
import batch_analysis.task_manager
import dataset.tum.tum_manager
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.ate.absolute_trajectory_error as ate
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import benchmarks.tracking.tracking_benchmark as tracking_benchmark
import simulation.unrealcv.unrealcv_simulator as uecv_sim
import simulation.controllers.trajectory_follow_controller as follow_cont


class VisualOdometryExperiment(batch_analysis.experiment.Experiment):

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
                    '/media/john/Storage/simulators/BlockWorld/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'BlockWorld', imeta.EnvironmentType.OUTDOOR_LANDSCAPE, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ),
            (
                    '/media/john/Storage/simulators/AIUE_V01_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ),
            (
                    '/media/john/Storage/simulators/AIUE_V01_002/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_002', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
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

        # --------- REAL WORLD DATASETS -----------

        # Import KITTI dataset
        for sequence_num in range(11):
            path = os.path.expanduser(os.path.join('~', 'datasets', 'KITTI', 'dataset'))
            if os.path.isdir(path) and os.path.isdir(os.path.join(path, "{0:02}".format(sequence_num))):
                task = task_manager.get_import_dataset_task(
                    module_name='dataset.kitti.kitti_loader',
                    path=path,
                    additional_args={'sequence_number': sequence_num},
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='12:00:00'
                )
                if task.is_finished:
                    trajectory_group = self._add_trajectory_group(
                        'KITTI trajectory {}'.format(sequence_num), task.result)
                    self._update_trajectory_group(trajectory_group, task_manager, db_client)
                else:
                    task_manager.do_task(task)

        # Import EuRoC datasets
        for name, path in [
            ('EuRoC MH_01_easy', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_01_easy'))),
            ('EuRoC MH_02_easy', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_02_easy'))),
            ('EuRoC MH_02_medium', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_03_medium'))),
            ('EuRoC MH_04_difficult', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_04_difficult'))),
            ('EuRoC MH_05_difficult', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_05_difficult'))),
            ('EuRoC V1_01_easy', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V1_01_easy'))),
            ('EuRoC V1_02_medium', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V1_02_medium'))),
            ('EuRoC V1_03_difficult', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V1_03_difficult'))),
            ('EuRoC V2_01_easy', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V2_01_easy'))),
            ('EuRoC V2_02_medium', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V2_02_medium'))),
            ('EuRoC V2_03_difficult', os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'V2_03_difficult')))
        ]:
            if os.path.isdir(path):
                task = task_manager.get_import_dataset_task(
                    module_name='dataset.euroc.euroc_loader',
                    path=path,
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if task.is_finished:
                    trajectory_group = self._add_trajectory_group(name, task.result)
                    self._update_trajectory_group(trajectory_group, task_manager, db_client)
                else:
                    task_manager.do_task(task)

        # Import TUM datasets using the manager.
        tum_manager = dataset.tum.tum_manager.TUMManager({
            'rgbd_dataset_freiburg1_xyz': True,
            'rgbd_dataset_freiburg1_rpy': True,
            'rgbd_dataset_freiburg2_xyz': True,
            'rgbd_dataset_freiburg2_rpy': True
        })
        tum_manager.do_imports(os.path.expanduser(os.path.join('~', 'datasets', 'TUM')), task_manager)
        for name, dataset_id in tum_manager.datasets:
            trajectory_group = self._add_trajectory_group("TUM {}".format(name), dataset_id)
            self._update_trajectory_group(trajectory_group, task_manager, db_client)

        # --------- SYSTEMS -----------
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)

        # ORBSLAM2 - Create 9 variants, with different procesing modes
        settings_list = [
            (sensor_mode, n_features, resolution)
            for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}
            for n_features in {1000, 1500, 2500}
            for resolution in {(640, 480)}
        ]
        if len(self._orbslam_systems) < len(settings_list):
            for settings in settings_list:
                name = 'ORBSLAM2 {mode} - {resolution} - {features} features'.format(
                    mode=settings[0].name.lower(),
                    features=settings[1],
                    resolution=settings[2]
                )
                vocab_path = os.path.expanduser(os.path.join('~', 'code', 'ORBSLAM2', 'Vocabulary', 'ORBvoc.txt'))
                if name not in self._orbslam_systems and os.path.isfile(vocab_path):
                    orbslam_id = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                        vocabulary_file=vocab_path,
                        mode=settings[0],
                        settings={
                            'ORBextractor': {
                                'nFeatures': settings[1]
                            }
                        }, resolution=settings[2]
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

    def _add_trajectory_group(self, name: str, reference_dataset_id: bson.ObjectId) -> 'TrajectoryGroup':
        """
        Add a trajectory group to the experiment based around a particular dataset.
        :param name:
        :param reference_dataset_id:
        :return:
        """
        if name not in self._trajectory_groups:
            self._trajectory_groups[name] = TrajectoryGroup(name=name, reference_id=reference_dataset_id)
            self._set_property('trajectory_groups.{0}'.format(name), self._trajectory_groups[name].serialize())
        return self._trajectory_groups[name]

    def _update_trajectory_group(self, trajectory_group: 'TrajectoryGroup',
                                 task_manager: batch_analysis.task_manager.TaskManager,
                                 db_client: database.client.DatabaseClient):
        """
        Update KITTI trajectory groups. Different datasets get different updates because they use different
        sets of simulators.
        :param trajectory_group:
        :param task_manager:
        :param db_client:
        :return:
        """
        quality_variations = [('max quality', {
        #}), ('reduced resolution', {
        #    'resolution': {'width': 256, 'height': 144}  # Extremely low res
        #}), ('narrow fov', {
        #    'fov': 15
        #}), ('no depth-of-field', {
        #    'depth_of_field_enabled': False
        #}), ('no textures', {
        #    'texture_mipmap_bias': 8
        #}), ('no normal maps', {
        #    'normal_maps_enabled': False,  # No normal maps
        #}), ('no reflections', {
        #    'roughness_enabled': False  # No reflections
        #}), ('simple geometry', {
        #    'geometry_decimation': 4,   # Simple geometry
        #}), ('low quality', {
        #    # low quality
        #    'depth_of_field_enabled': False,
        #    'texture_mipmap_bias': 8,
        #    'normal_maps_enabled': False,
        #    'roughness_enabled': False,
        #    'geometry_decimation': 4,
        #}), ('worst quality', {
        #    # absolute minimum quality
        #    'resolution': {'width': 256, 'height': 144},
        #    'fov': 15,
        #    'depth_of_field_enabled': False,
        #    'texture_mipmap_bias': 8,
        #    'normal_maps_enabled': False,
        #    'roughness_enabled': False,
        #    'geometry_decimation': 4,
        })]

        changed = False
        # Add simulators for the different trajectory groups with different settings
        if 'KITTI' in trajectory_group.name and 'BlockWorld' in self._simulators:
            changed |= trajectory_group.add_simulator('BlockWorld', self._simulators['BlockWorld'], {})
        if trajectory_group.name == 'TUM rgbd_dataset_freiburg1_xyz' or \
                trajectory_group.name == 'TUM rgbd_dataset_freiburg1_rpy':
            for sim_name, variant, offset in [
                ('AIUE_V01_001', 1, uetf.create_serialized((-100, 300, 0), (0, 0, 90))),
                ('AIUE_V01_001', 2, uetf.create_serialized((-300, 600, 0), (0, 0, -90))),
                ('AIUE_V01_001', 3, uetf.create_serialized((-200, -100, 0), (0, 0, -90))),
                ('AIUE_V01_002', 1, uetf.create_serialized((-100, -200, 0), (0, 0, 0))),
                ('AIUE_V01_002', 2, uetf.create_serialized((-1000, 300, -150), (0, 0, -150))),
                ('AIUE_V01_002', 3, uetf.create_serialized((-1150, -300, 150), (0, 0, 30))),
                ('AIUE_V01_003', 1, uetf.create_serialized((-70, -70, 10), (0, 0, -90))),
                ('AIUE_V01_003', 2, uetf.create_serialized((-300, 760, 0), (0, 0, 60))),
                ('AIUE_V01_004', 1, uetf.create_serialized((410, 150, -40), (0, 0, 30))),
                ('AIUE_V01_004', 2, uetf.create_serialized((410, 150, 320), (0, 0, 10))),
                ('AIUE_V01_004', 3, uetf.create_serialized((-90, -100, -10), (0, 0, -140))),
                ('AIUE_V01_005', 1, uetf.create_serialized((-180, -1350, -80), (0, 0, -70))),
                ('AIUE_V01_005', 2, uetf.create_serialized((-660, 60, -80), (0, 0, -130))),
                ('AIUE_V01_005', 3, uetf.create_serialized((-710, -1380, 310), (0, 0, -100)))
            ]:
                if sim_name in self._simulators:
                    for quality_name, quality_settings in quality_variations:
                        changed |= trajectory_group.add_simulator(
                            "{0} variant {1} - {2}".format(sim_name, variant, quality_name),
                            self._simulators[sim_name], du.defaults({'origin': offset}, quality_settings))
        if trajectory_group.name == 'EuRoC MH_01_easy':
            for sim_name, variant, offset in [
                ('AIUE_V01_002', 1, uetf.create_serialized((500, -100, 150), (0, 0, 130))),
                ('AIUE_V01_002', 2, uetf.create_serialized((2500, 650, 100), (0, 0, -70))),
                ('AIUE_V01_003', 1, uetf.create_serialized((-350, 600, 150), (0, 0, 30))),
                ('AIUE_V01_003', 2, uetf.create_serialized((-70, -70, 180), (0, 0, -160))),
                ('AIUE_V01_004', 1, uetf.create_serialized((-290, -430, 170), (0, 0, 130))),
                ('AIUE_V01_004', 2, uetf.create_serialized((-430, 380, 380), (-6.46637, 7.644286, -40.43219))),
                ('AIUE_V01_005', 1, uetf.create_serialized((-220, -140, 250), (0, 0, 0))),
                ('AIUE_V01_005', 2, uetf.create_serialized((470, -810, 370), (0, 0, -130))),
                ('AIUE_V01_005', 3, uetf.create_serialized((-520, -1280, 120), (0, 0, 170)))
            ]:
                if sim_name in self._simulators:
                    for quality_name, quality_settings in quality_variations:
                        changed |= trajectory_group.add_simulator(
                            "{0} variant {1} - {2}".format(sim_name, variant, quality_name),
                            self._simulators[sim_name], du.defaults({'origin': offset}, quality_settings))
        if trajectory_group.name == 'EuRoC MH_04_hard':
            for sim_name, variant, offset in [
                ('AIUE_V01_005', 1, uetf.create_serialized((660, -1050, 70), (0, 0, 170))),
                ('AIUE_V01_005', 2, uetf.create_serialized((-890, -190, 90), (0, 0, 0))),
            ]:
                if sim_name in self._simulators:
                    for quality_name, quality_settings in quality_variations:
                        changed |= trajectory_group.add_simulator(
                            "{0} variant {1} - {2}".format(sim_name, variant, quality_name),
                            self._simulators[sim_name], du.defaults({'origin': offset}, quality_settings))

        # Do the imports for the group, and save any changes
        changed |= trajectory_group.do_imports(task_manager, db_client)
        if changed:
            self._set_property('trajectory_groups.{0}'.format(trajectory_group.name), trajectory_group.serialize())

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
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Computed trajectories for {0}".format(trajectory_group.name))
            ax = figure.add_subplot(111, projection='3d')
            ax.set_xlabel('x-location')
            ax.set_ylabel('y-location')
            ax.set_zlabel('z-location')
            ax.plot([0], [0], [0], 'ko', label='origin')
            added_ground_truth = False

            image_sources = {'reference dataset': trajectory_group.reference_dataset}
            for simulator_id, dataset_id in trajectory_group.generated_datasets.items():
                if simulator_id in simulator_names:
                    image_sources[simulator_names[simulator_id]] = dataset_id
                else:
                    image_sources[simulator_id] = dataset_id

            for system_name, system_id in systems.items():
                # For each image source in this group
                for dataset_name, dataset_id in image_sources.items():
                    trial_result_id = self.get_trial_result(system_id, dataset_id)
                    if trial_result_id is not None:
                        trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                        if trial_result is not None:
                            if trial_result.success:
                                if not added_ground_truth:
                                    lower, upper = plot_trajectory(ax, trial_result.get_ground_truth_camera_poses(),
                                                                   'ground truth trajectory')
                                    mean = (upper + lower) / 2
                                    lower = 2 * lower - mean
                                    upper = 2 * upper - mean
                                    ax.set_xlim(lower, upper)
                                    ax.set_ylim(lower, upper)
                                    ax.set_zlim(lower, upper)
                                    added_ground_truth = True
                                plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                                "{0} on {1}".format(system_name, dataset_name))
                            else:
                                print("Got failed trial: {0}".format(trial_result.reason))

            logging.getLogger(__name__).info("... plotted trajectories for {0}".format(trajectory_group.name))
            ax.legend()
            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()

    def export_data(self, db_client: database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to
        :param db_client:
        :return:
        """
        for trajectory_group in self._trajectory_groups.values():
            trajectory = traj_help.get_trajectory_for_image_source(db_client, trajectory_group.reference_dataset)
            with open('trajectory_{0}.csv'.format(trajectory_group.name), 'w') as output_file:
                output_file.write('Name,X,Y,Z,Roll,Pitch,Yaw\n')
                for idx, timestamp in enumerate(sorted(trajectory.keys())):
                    ue_pose = uetf.transform_to_unreal(trajectory[timestamp])
                    output_file.write('{name},{x},{y},{z},{roll},{pitch},{yaw}\n'.format(
                        name=idx,
                        x=ue_pose.location[0],
                        y=ue_pose.location[1],
                        z=ue_pose.location[2],
                        roll=ue_pose.euler[0],
                        pitch=ue_pose.euler[1],
                        yaw=ue_pose.euler[2]))

    def serialize(self):
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment', 2)

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
        update_schema(serialized_representation)

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


def update_schema(serialized: dict):
    version = dh.get_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment')
    if version < 1:  # unversioned -> 1
        if 'simulators' in serialized:
            serialized['kitti_simulators'] = serialized['simulators']
            del serialized['simulators']
            version = 1
    if version < 2:
        simulators = {}
        if 'kitti_simulators' in serialized:
            simulators = du.defaults(simulators, serialized['kitti_simulators'])
        if 'euroc_simulators' in serialized:
            simulators = du.defaults(simulators, serialized['euroc_simulators'])
        if 'tum_simulators' in serialized:
            simulators = du.defaults(simulators, serialized['tum_simulators'])
        serialized['simulators'] = simulators


class TrajectoryGroup:
    """
    A Trajectory Group is a helper structure to manage image datasets grouped by camera trajectory.
    In this experiment, it is created from a single reference dataset,
    and produces many synthetic datasets with the same camera trajectory.

    For convenience, it serializes and deserialzes as a group.
    """

    def __init__(self, name: str, reference_id: bson.ObjectId, baseline_configuration: dict = None,
                 simulators: dict = None, controller_id: bson.ObjectId = None, generated_datasets: dict = None):
        self.name = name
        self.reference_dataset = reference_id
        self.baseline_configuration = baseline_configuration if baseline_configuration is not None else {}
        self.simulators = simulators if simulators is not None else {}

        self.follow_controller_id = controller_id
        self.generated_datasets = generated_datasets if generated_datasets is not None else {}

    def add_simulator(self, name: str, simulator_id: bson.ObjectId, simulator_config: dict = None) -> bool:
        """
        Add a simulator to the group, from which a dataset will be generated
        :param name: The name of the simulator and simulator config combination
        :param simulator_id: The id of the simulator
        :param simulator_config: Configuration used to generate the dataset
        :return: True if that simulator was not already part of the group, so the group needs to be re-saved
        """
        if name not in self.simulators:
            self.simulators[name] = (simulator_id, simulator_config)
            return True
        return False

    def get_all_dataset_ids(self) -> set:
        """
        Get all the datasets in this group, as a set
        :return:
        """
        return set(self.generated_datasets.values()) | {self.reference_dataset}

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
                db_client, self.reference_dataset, sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL)
            changed = True
        # Next, if we haven't already, compute baseline configuration from the reference dataset
        if self.baseline_configuration is None:
            reference_dataset = dh.load_object(db_client, db_client.image_source_collection, self.reference_dataset)
            intrinsics = reference_dataset.get_camera_intrinsics()
            self.baseline_configuration = {
                    # Simulation execution config
                    'stereo_offset': reference_dataset.get_stereo_baseline() \
                    if reference_dataset.is_stereo_available else 0,
                    'provide_rgb': True,
                    'provide_depth': reference_dataset.is_depth_available,
                    'provide_labels': reference_dataset.is_labels_available,
                    'provide_world_normals': reference_dataset.is_normals_available,

                    # Simulator camera settings, be similar to the reference dataset
                    'resolution': {'width': intrinsics.width, 'height': intrinsics.height},
                    'fov': max(intrinsics.horizontal_fov, intrinsics.vertical_fov),
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
                }
            changed = True

        # Then, for each combination of simulator id and config, generate a dataset
        for sim_name, (simulator_id, simulator_config) in self.simulators.items():
            # Schedule generation of quality variations that don't exist yet
            if sim_name not in self.generated_datasets:
                generate_dataset_task = task_manager.get_generate_dataset_task(
                    controller_id=self.follow_controller_id,
                    simulator_id=simulator_id,
                    simulator_config=du.defaults({}, simulator_config, self.baseline_configuration),
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
            'reference_id': self.reference_dataset,
            'baseline_configuration': self.baseline_configuration,
            'simulators': self.simulators,
            'controller_id': self.follow_controller_id,
            'generated_datasets': self.generated_datasets
        }

    @classmethod
    def deserialize(cls, serialized_representation: dict) -> 'TrajectoryGroup':
        if 'reference_id' not in serialized_representation and 'real_world_dataset' in serialized_representation:
            serialized_representation['reference_id'] = serialized_representation['real_world_dataset']
        return cls(
            name=serialized_representation['name'],
            reference_id=serialized_representation['reference_id'],
            baseline_configuration=serialized_representation['baseline_configuration'],
            simulators={name: tuple(data) for name, data in serialized_representation['simulators'].items()},
            controller_id=serialized_representation['controller_id'],
            generated_datasets=serialized_representation['generated_datasets']
        )
