#Copyright (c) 2017, John Skinner
import os
import bson
import numpy as np
import operator
import logging
import util.database_helpers as dh
import util.associate
import util.transform as tf
import util.dict_utils as du
import core.image_source
import core.sequence_type
import core.benchmark
import metadata.camera_intrinsics as cam_intr
import metadata.image_metadata as imeta
import batch_analysis.experiment
import dataset.tum.tum_manager
import systems.feature.detectors.sift_detector as sift_detector
import systems.feature.detectors.orb_detector as orb_detector
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.ate.absolute_trajectory_error as ate
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import benchmarks.tracking.tracking_benchmark as tracking_benchmark
import benchmarks.feature.detection_comparison as detection_comparison
import simulation.unrealcv.unrealcv_simulator as uecv_sim
import simulation.controllers.flythrough_controller as fly_cont
import simulation.controllers.trajectory_follow_controller as follow_cont


class TrajectoryGroup:

    def __init__(self, simulator_id, group_name, default_simulator_config, max_quality_id,
                 controller_id=None, quality_variations=None,):
        self.simulator_id = simulator_id
        self.name = group_name
        self.default_simulator_config = default_simulator_config
        self.max_quality_id = max_quality_id
        self.follow_controller_id = controller_id
        self.quality_variations = quality_variations if quality_variations is not None else []

    def __eq__(self, other):
        if isinstance(other, TrajectoryGroup):
            return (self.simulator_id == other.simulator_id and
                    self.default_simulator_config == other.default_simulator_config and
                    self.max_quality_id == other.max_quality_id and
                    self.follow_controller_id == other.follow_controller_id and
                    self.quality_variations == other.quality_variations)
        return NotImplemented

    def get_all_dataset_ids(self):
        """
        Get all the image dataset ids in this trajectory group.
        Used for marshalling trials.
        :return:
        """
        datasets = {variation['dataset'] for variation in self.quality_variations}
        datasets.add(self.max_quality_id)
        return datasets

    def do_imports(self, task_manager, db_client):
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
                db_client, self.max_quality_id, sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL)
            changed = True
        # Next, schedule reduced quality dataset generation for each desired config
        # These are all the quality variations we're going to do for now.
        #for config in [{
        #    'resolution': {'width': 256, 'height': 144}  # Extremely low res
        #}, {
        #    'fov': 15   # Narrow field of view
        #}, {
        #    'depth_of_field_enabled': False   # No dof
        #}, {
        #    'lit_mode': False  # Unlit
        #}, {
        #    'texture_mipmap_bias': 8  # No textures
        #}, {
        #    'normal_maps_enabled': False,  # No normal maps
        #}, {
        #    'roughness_enabled': False  # No reflections
        #}, {
        #    'geometry_decimation': 4,   # Simple geometry
        #}, {
        #    # low quality
        #    'depth_of_field_enabled': False,
        #    'lit_mode': False,
        #    'texture_mipmap_bias': 8,
        #    'normal_maps_enabled': False,
        #    'roughness_enabled': False,
        #    'geometry_decimation': 4,
        #}, {
        #    # absolute minimum quality - yall got any more of them pixels
        #    'resolution': {'width': 256, 'height': 144},
        #    'fov': 15,
        #    'depth_of_field_enabled': False,
        #    'lit_mode': False,
        #    'texture_mipmap_bias': 8,
        #    'normal_maps_enabled': False,
        #    'roughness_enabled': False,
        #    'geometry_decimation': 4,
        #}]:
        for config in [{
            # low quality
            'depth_of_field_enabled': False,
            'texture_mipmap_bias': 8,
            'normal_maps_enabled': False,
            'roughness_enabled': False,
            'geometry_decimation': 2,
        }]:
            # check for an existing run using this config
            found = False
            for variation in self.quality_variations:
                if variation['config'] == config:
                    found = True
                    break
            # Schedule generation of quality variations that don't exist yet
            if not found:
                generate_dataset_task = task_manager.get_generate_dataset_task(
                    controller_id=self.follow_controller_id,
                    simulator_id=self.simulator_id,
                    simulator_config=du.defaults({}, config, self.default_simulator_config),
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if generate_dataset_task.is_finished:
                    result_ids = generate_dataset_task.result
                    if isinstance(result_ids, bson.ObjectId):  # we got a single id
                        result_ids = [result_ids]
                    for result_id in result_ids:
                        self.quality_variations.append({
                            'config': config,
                            'dataset': result_id
                        })
                        changed = True
                else:
                    task_manager.do_task(generate_dataset_task)
        return changed

    def serialize(self):
        return {
            'simulator_id': self.simulator_id,
            'name': self.name,
            'default_simulator_config': self.default_simulator_config,
            'max_quality_id': self.max_quality_id,
            'controller_id': self.follow_controller_id,
            'quality_variations': self.quality_variations
        }

    @classmethod
    def deserialize(cls, serialized_representation):
        return cls(
            simulator_id=serialized_representation['simulator_id'],
            group_name=serialized_representation['name'],
            default_simulator_config=serialized_representation['default_simulator_config'],
            max_quality_id=serialized_representation['max_quality_id'],
            controller_id=serialized_representation['controller_id'],
            quality_variations=serialized_representation['quality_variations']
        )


class VisualSlamExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, feature_detectors=None, libviso_system=None, orbslam_systems=None,
                 tum_manager=None, kitti_datasets=None, euroc_datasets=None,
                 simulators=None, flythrough_controller=None, trajectory_groups=None,
                 benchmark_feature_diff=None,
                 benchmark_rpe=None, benchmark_ate=None, benchmark_trajectory_drift=None, benchmark_tracking=None,
                 trial_map=None, result_map=None, id_=None):
        super().__init__(id_=id_)
        # Systems
        self._feature_detectors = feature_detectors if feature_detectors is not None else {}
        self._libviso_system = libviso_system
        self._orbslam_systems = dict(orbslam_systems) if orbslam_systems is not None else {}

        # Real-world image sources
        self._kitti_datasets = set(kitti_datasets) if kitti_datasets is not None else set()
        self._euroc_datasets = dict(euroc_datasets) if euroc_datasets is not None else {}
        if isinstance(tum_manager, dataset.tum.tum_manager.TUMManager):
            self._tum_manager = tum_manager
        else:
            self._tum_manager = dataset.tum.tum_manager.TUMManager({
                'rgbd_dataset_freiburg1_xyz': True,
                'rgbd_dataset_freiburg1_rpy': True,
                'rgbd_dataset_freiburg2_xyz': True,
                'rgbd_dataset_freiburg2_rpy': True
            })
        # Generated image sources
        self._simulators = simulators if simulators is not None else {}
        self._flythrough_controller = flythrough_controller
        self._trajectory_groups = trajectory_groups if trajectory_groups is not None else {}

        # Benchmarks
        self._benchmark_feature_diff = benchmark_feature_diff
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_ate = benchmark_ate
        self._benchmark_trajectory_drift = benchmark_trajectory_drift
        self._benchmark_tracking = benchmark_tracking

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
        task = task_manager.get_import_dataset_task(
            module_name='dataset.kitti.kitti_loader',
            path=os.path.expanduser(os.path.join('~', 'datasets', 'KITTI', 'dataset')),
            num_cpus=1,
            num_gpus=0,
            memory_requirements='3GB',
            expected_duration='72:00:00'
        )
        if task.is_finished:
            self._kitti_datasets |= set(task.result)
            self._add_to_set('kitti_datasets', task.result)
        else:
            task_manager.do_task(task)

        # Import EuRoC datasets
        for path in [
            os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_01_easy')),
            os.path.expanduser(os.path.join('~', 'datasets', 'EuRoC', 'MH_04_difficult'))
        ]:
            path_key = path.replace('/', '_')
            if path_key not in self._euroc_datasets and os.path.isdir(path):
                task = task_manager.get_import_dataset_task(
                    module_name='dataset.euroc.euroc_loader',
                    path=path,
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if task.is_finished:
                    self._euroc_datasets[path_key] = task.result
                    self._set_property('euroc_datasets.{0}'.format(path_key), task.result)
                else:
                    task_manager.do_task(task)

        # Import TUM datasets using the manager
        #self._tum_manager.do_imports(os.path.expanduser(os.path.join('~', 'datasets', 'TUM')), task_manager)

        # --------- SYNTHETIC DATASETS -----------
        # Add simulators explicitly, they have different metadata, so we can't just search
        for exe, world_name, environment_type, light_level, time_of_day in [
            (
                '/media/john/Storage/simulators/AIUE_V01_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                'AIUE_V01_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT, imeta.TimeOfDay.DAY
            ), (
                '/media/john/Storage/simulators/AIUE_V01_002/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                'AIUE_V01_002', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT, imeta.TimeOfDay.DAY
            ), (
                '/media/john/Storage/simulators/AIUE_V01_003/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                'AIUE_V01_003', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT, imeta.TimeOfDay.DAY
            ), (
                '/media/john/Storage/simulators/AIUE_V01_004/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                'AIUE_V01_004', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT, imeta.TimeOfDay.DAY
            )
        ]:
            sim_id = dh.add_unique(db_client.image_source_collection, uecv_sim.UnrealCVSimulator(
                executable_path=exe,
                world_name=world_name,
                environment_type=environment_type,
                light_level=light_level,
                time_of_day=time_of_day
            ))
            self._simulators[world_name] = sim_id
            self._set_property('simulators.{}'.format(world_name), sim_id)

        # Add controllers
        if self._flythrough_controller is None:
            self._flythrough_controller = dh.add_unique(db_client.image_source_collection,
                                                        fly_cont.FlythroughController(
                                                            max_speed=0.5,
                                                            acceleration=0.1,
                                                            acceleration_noise=0.01,
                                                            max_turn_angle=np.pi / 72,
                                                            avoidance_radius=1,
                                                            avoidance_scale=1,
                                                            length=1000,
                                                            seconds_per_frame=0.1
                                                        ))
            self._set_property('flythrough_controller', self._flythrough_controller)

        # Generate max-quality flythroughs
        trajectories_per_environment = 3
        for world_name, simulator_id in self._simulators.items():
            for repeat in range(trajectories_per_environment):
                # Default, maximum quality settings. We override specific settings in the quality group
                simulator_config = {
                    # Simulation execution config
                    'stereo_offset': 0.15,  # meters
                    'provide_rgb': True,
                    'provide_depth': False,
                    'provide_labels': False,
                    'provide_world_normals': False,

                    # Simulator settings - No clear maximum, but these are the best we'll use
                    'resolution': {'width': 640, 'height': 480},
                    'fov': 90,
                    'depth_of_field_enabled': True,
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
                generate_dataset_task = task_manager.get_generate_dataset_task(
                    controller_id=self._flythrough_controller,
                    simulator_id=simulator_id,
                    simulator_config=simulator_config,
                    repeat=repeat,
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='4:00:00'
                )
                if not generate_dataset_task.is_finished:
                    task_manager.do_task(generate_dataset_task)
                else:
                    result_ids = generate_dataset_task.result
                    if isinstance(result_ids, bson.ObjectId):   # we got a single id
                        result_ids = [result_ids]
                    for result_id in result_ids:
                        if result_id not in self._trajectory_groups:
                            self._trajectory_groups[result_id] = TrajectoryGroup(
                                simulator_id=simulator_id,
                                group_name="{world} {repeat}".format(world=world_name, repeat=repeat),
                                default_simulator_config=simulator_config,
                                max_quality_id=result_id
                            )
                            self._set_property('trajectory_groups.{0}'.format(result_id),
                                               self._trajectory_groups[result_id].serialize())

        # Schedule dataset generation for lower-quality datasets in each trajectory group
        for trajectory_group in self._trajectory_groups.values():
            if trajectory_group.do_imports(task_manager, db_client):
                self._set_property('trajectory_groups.{0}'.format(trajectory_group.max_quality_id),
                                   trajectory_group.serialize())

        # --------- SYSTEMS -----------
        # Import the systems under test for this experiment.
        # They are: sundry feature detectors, libviso2, orbslam2

        # Feature detectors
        if 'SIFT detector' not in self._feature_detectors:
            self._feature_detectors['SIFT detector'] = dh.add_unique(
                db_client.system_collection, sift_detector.SiftDetector({
                    'num_features': 0,
                    'num_octave_layers': 4,
                    'contrast_threshold': 0.04,
                    'edge_threshold': 10,
                    'sigma': 1.6
                }))
            self._set_property('feature_detectors.SIFT detector', self._feature_detectors['SIFT detector'])
        if 'ORB detector' not in self._feature_detectors:
            self._feature_detectors['ORB detector'] = dh.add_unique(
                db_client.system_collection, orb_detector.ORBDetector({
                    'num_features': 1000,
                    'scale_factor': 1.2,
                    'num_levels': 8,
                    'edge_threshold': 31,
                    'patch_size': 31,
                    'fast_threshold': 20
                }))
            self._set_property('feature_detectors.ORB detector', self._feature_detectors['ORB detector'])

        # LIBVISO2
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)

        # ORBSLAM2 - now only a single variant
        settings_list = [
            (sensor_mode, n_features, resolution)
            for sensor_mode in {orbslam2.SensorMode.STEREO}
            for n_features in {1500}
            for resolution in {(640, 480)}
        ]
        if len(self._orbslam_systems) < len(settings_list):
            for settings in settings_list:
                name = 'ORBSLAM2 {mode} - {resolution} - {features} features'.format(
                    mode=settings[0].name.lower(),
                    features=settings[1],
                    resolution=settings[2]
                )
                orbslam_id = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                    vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
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
        if self._benchmark_feature_diff is None:
            self._benchmark_feature_diff = dh.add_unique(
                db_client.benchmarks_collection, detection_comparison.FeatureDetectionComparison(acceptable_radius=4))
            self._set_property('benchmark_feature_diff', self._benchmark_feature_diff)
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

    def schedule_tasks(self, task_manager, db_client):
        """

        :param task_manager:
        :param db_client:
        :return:
        """
        # Feature Detectors
        for feature_detector_id in self._feature_detectors.values():
            # Make sure we've got a place to store trials for this feature detector
            if feature_detector_id not in self._trial_map:
                self._trial_map[feature_detector_id] = {}

            # Run the feature detector on each image source in each trajectory group
            for group in self._trajectory_groups.values():
                max_quality_trial = None

                # Run the feature detector on the top quality run
                task = task_manager.get_run_system_task(
                    system_id=feature_detector_id,
                    image_source_id=group.max_quality_id,
                    expected_duration='2:00:00'
                )
                if not task.is_finished:
                    task_manager.do_task(task)
                else:
                    # We've already run it, add to the trial map, and get it ready for comparison
                    max_quality_trial = task.result
                    self._trial_map[feature_detector_id][group.max_quality_id] = task.result
                    self._set_property('trial_map.{0}.{1}'.format(feature_detector_id, group.max_quality_id),
                                       task.result)

                for variation in group.quality_variations:
                    # Run with each reduced quality variation
                    task = task_manager.get_run_system_task(
                        system_id=feature_detector_id,
                        image_source_id=variation['dataset'],
                        expected_duration='2:00:00'
                    )
                    if not task.is_finished:
                        # Run the task if we haven't yet
                        task_manager.do_task(task)
                    else:
                        # If task is complete, benchmark it
                        self._trial_map[feature_detector_id][variation['dataset']] = task.result
                        self._set_property('trial_map.{0}.{1}'.format(feature_detector_id, variation['dataset']),
                                           task.result)

                        # Task is complete, perform comparison benchmark
                        if max_quality_trial is not None:
                            variation_trial_result = task.result
                            task = task_manager.get_trial_comparison_task(
                                trial_result1_id=task.result,
                                trial_result2_id=max_quality_trial,
                                comparison_id=self._benchmark_feature_diff,
                                memory_requirements='6GB',
                                expected_duration='4:00:00'
                            )
                            if task.is_finished:
                                if variation_trial_result not in self._result_map:
                                    self._result_map[variation_trial_result] = {}
                                self._result_map[variation_trial_result][self._benchmark_feature_diff] = task.result
                                self._set_property(
                                    'result_map.{0}.{1}'.format(variation_trial_result,
                                                                self._benchmark_feature_diff),
                                    task.result)
                            else:
                                task_manager.do_task(task)

        # Groups of things for testing all the systems
        libviso_system = dh.load_object(db_client, db_client.system_collection, self._libviso_system)
        #orbslam_systems = [
        #    dh.load_object(db_client, db_client.system_collection, system_id)
        #    for system_id in self._orbslam_systems.values()
        #]
        benchmarks = [
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_rpe),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_ate),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_trajectory_drift),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_tracking)
        ]
        datasets = set(self._kitti_datasets) | self._tum_manager.all_datasets | set(self._euroc_datasets.values())
        for group in self._trajectory_groups.values():
            datasets = datasets | group.get_all_dataset_ids()
        system_trials = set()

        # For each of all the datasets, run LIBVISO and ORBSLAM on the dataset
        for image_source_id in datasets:
            image_source = self._load_image_source(db_client, image_source_id)

            # Libviso2
            if libviso_system.is_image_source_appropriate(image_source):
                task = task_manager.get_run_system_task(
                    system_id=libviso_system.identifier,
                    image_source_id=image_source.identifier,
                    expected_duration='8:00:00',
                    memory_requirements='12GB'
                )
                if not task.is_finished:
                    task_manager.do_task(task)
                else:
                    system_trials.add(task.result)
                    if self._libviso_system not in self._trial_map:
                        self._trial_map[self._libviso_system] = {}
                    self._trial_map[self._libviso_system][image_source_id] = task.result
                    self._set_property('trial_map.{0}.{1}'.format(self._libviso_system, image_source_id), task.result)

            # ORBSLAM2
            #for orbslam_system in orbslam_systems:
            #    if (orbslam_system.is_image_source_appropriate(image_source) and
            #            orbslam_system.mode != orbslam2.SensorMode.RGBD):
            #        task = task_manager.get_run_system_task(
            #            system_id=orbslam_system.identifier,
            #            image_source_id=image_source.identifier,
            #            expected_duration='4:00:00',
            #            memory_requirements='4GB'
            #        )
            #        if not task.is_finished:
            #            #task_manager.do_task(task)
            #            pass
            #        else:
            #            system_trials.add(task.result)
            #            if orbslam_system.identifier not in self._trial_map:
            #                self._trial_map[orbslam_system.identifier] = {}
            #            self._trial_map[orbslam_system.identifier][image_source_id] = task.result
            #            self._set_property('trial_map.{0}.{1}'.format(orbslam_system.identifier, image_source_id),
            #                               task.result)

        # Benchmark system results
        for trial_result_id in system_trials:
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
        import matplotlib.pyplot as pyplot
        # Step 1 - Show some aggregate statistics on image feature detection
        #self._plot_feature_changes(db_client)

        # Step 2 - Plot images with interesting changes
        #self._plot_interesting_feature_changes(db_client)

        # Step 3 - Plot the changes for each image
        #self._plot_per_image_feature_changes(db_client)

        # Step 4 - Trajectory visualization: For each system and each trajectory, plot the different paths
        #self._plot_trajectories(db_client)

        # Step 5 - Aggregation: For each benchmark, compare real-world and different qualities
        self._plot_aggregate_performance(db_client)

        # Step 6 - detailed analysis of performance vs time for each trajectory
        #self._plot_performance_vs_time(db_client)

        # final figure configuration, and show the figures
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()

    def _load_image_source(self, db_client, image_source_id):
        if image_source_id not in self._placeholder_image_collections:
            s_image_collection = db_client.image_source_collection.find_one({'_id': image_source_id})
            image_ids = [img_id for _, img_id in s_image_collection['images']]
            self._placeholder_image_collections[image_source_id] = PlaceholderImageCollection(
                id_=s_image_collection['_id'],
                is_labels_available=db_client.image_collection.find({
                    '_id': {'$in': image_ids},
                    'metadata.labelled_objects': {'$in': [[], None]}
                }).count() <= 0,
                is_per_pixel_labels_available=db_client.image_collection.find({
                    '_id': {'$in': image_ids},
                    'labels_data': None,
                    'left_labels_data': None
                }).count() <= 0,
                is_normals_available=db_client.image_collection.find({
                    '_id': {'$in': image_ids},
                    'world_normals_data': None,
                    'left_world_normals_data': None
                }).count() <= 0,
                is_depth_available=db_client.image_collection.find({
                    '_id': {'$in': image_ids},
                    'depth_data': None,
                    'left_depth_data': None
                }).count() <= 0,
                is_stereo_available=db_client.image_collection.find({
                    '_id': {'$in': image_ids},
                    'right_data': None
                }).count() <= 0,
                sequence_type=(core.sequence_type.ImageSequenceType.SEQUENTIAL
                               if s_image_collection['sequence_type'] == 'SEQ'
                               else core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
            )
        return self._placeholder_image_collections[image_source_id]

    def _plot_feature_changes(self, db_client):
        """
        Plot new vs missing features for each feature detector type
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Plotting general feature changes")
        for detector_name, detector_id in self._feature_detectors.items():
            if detector_id not in self._trial_map:
                continue
            # Track outstanding images
            highest_iou_points = None
            highest_iou = -1
            most_missing_points = None
            most_missing = 1
            x = []
            y = []
            for trajectory_group in self._trajectory_groups.values():
                for variation in trajectory_group.quality_variations:
                    if variation['dataset'] not in self._trial_map[detector_id]:
                        continue
                    trial_result_id = self._trial_map[detector_id][variation['dataset']]
                    if (trial_result_id in self._result_map and
                            self._benchmark_feature_diff in self._result_map[trial_result_id]):
                        benchmark_result_id = self._result_map[trial_result_id][self._benchmark_feature_diff]
                        benchmark_result = dh.load_object(db_client, db_client.results_collection, benchmark_result_id)
                        for image_changes in benchmark_result.changes:
                            x.append(len(image_changes['new_trial_points']))
                            y.append(len(image_changes['missing_reference_points']))
                            iou = len(image_changes['point_matches']) / (len(image_changes['point_matches']) +
                                                                         len(image_changes['new_trial_points']) +
                                                                         len(image_changes['missing_reference_points']))
                            if iou > highest_iou:
                                highest_iou = iou
                                highest_iou_points = (image_changes['trial_image_id'],
                                                      image_changes['point_matches'],
                                                      image_changes['new_trial_points'],
                                                      image_changes['missing_reference_points'])
                            if len(image_changes['missing_reference_points']) > most_missing:
                                most_missing = len(image_changes['missing_reference_points'])
                                most_missing_points = (image_changes['trial_image_id'],
                                                       image_changes['point_matches'],
                                                       image_changes['new_trial_points'],
                                                       image_changes['missing_reference_points'])

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Number of changes for {0}".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('new features')
            ax.set_ylabel('missing features')
            ax.plot(x, y, 'o')

            # Step 1a - Show some outstanding example images
            for name, points in [('{0} highest IoU ({1})'.format(detector_name, highest_iou), highest_iou_points),
                                 ('{0} most lost points ({1})'.format(detector_name, most_missing),
                                  most_missing_points)]:
                if points is not None:
                    image_id, matches, missing_trial, missing_reference = points
                    image = dh.load_object(db_client, db_client.image_collection, image_id)
                    figure = pyplot.figure()
                    figure.suptitle(name)
                    ax = figure.add_subplot(111)
                    ax.imshow(image.data)
                    ax.plot([match[0][0] for match in matches], [match[0][1] for match in matches], 'go')
                    ax.plot([point[0] for point in missing_trial], [point[1] for point in missing_trial], 'rx')

    def _plot_interesting_feature_changes(self, db_client):
        """
        Search through the results for interesting
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Looking for interesting images...")
        for detector_name, detector_id in self._feature_detectors.items():
            if detector_id not in self._trial_map:
                continue
            # Track outstanding images
            highest_iou_points = None
            highest_iou = -1
            lowest_iou_points = None
            lowest_iou = 1
            most_missing_points = None
            most_missing = -1
            least_missing_points = None
            least_missing = 10000000
            most_new_points = None
            most_new = -1
            least_new_points = None
            least_new = 1000000000
            for trajectory_group in self._trajectory_groups.values():
                for variation in trajectory_group.quality_variations:
                    if variation['dataset'] not in self._trial_map[detector_id]:
                        continue
                    trial_result_id = self._trial_map[detector_id][variation['dataset']]
                    if (trial_result_id in self._result_map and
                                self._benchmark_feature_diff in self._result_map[trial_result_id]):
                        benchmark_result_id = self._result_map[trial_result_id][self._benchmark_feature_diff]
                        benchmark_result = dh.load_object(db_client, db_client.results_collection, benchmark_result_id)
                        for image_changes in benchmark_result.changes:
                            iou = len(image_changes['point_matches']) / (len(image_changes['point_matches']) +
                                                                         len(image_changes['new_trial_points']) +
                                                                         len(image_changes['missing_reference_points']))
                            points = (image_changes['trial_image_id'],
                                      image_changes['point_matches'],
                                      image_changes['new_trial_points'],
                                      image_changes['missing_reference_points'])
                            if iou > highest_iou:
                                highest_iou = iou
                                highest_iou_points = points
                            if iou < lowest_iou:
                                lowest_iou = iou
                                lowest_iou_points = points
                            if len(image_changes['missing_reference_points']) > most_missing:
                                most_missing = len(image_changes['missing_reference_points'])
                                most_missing_points = points
                            if len(image_changes['missing_reference_points']) < least_missing:
                                least_missing = len(image_changes['missing_reference_points'])
                                least_missing_points = points
                            if len(image_changes['new_trial_points']) > most_new:
                                most_new = len(image_changes['new_trial_points'])
                                most_new_points = points
                            if len(image_changes['new_trial_points']) < least_new:
                                least_new = len(image_changes['new_trial_points'])
                                least_new_points = points
                    logging.getLogger(__name__).info("... searched trajectory {0}".format(trajectory_group.name))

            # Show the selected outstanding example images
            for name, points in [('{0} highest IoU ({1})'.format(detector_name, highest_iou), highest_iou_points),
                                 ('{0} lowest IoU ({1})'.format(detector_name, lowest_iou), lowest_iou_points),
                                 ('{0} most lost points ({1})'.format(detector_name, most_missing),
                                  most_missing_points),
                                 ('{0} fewest lost points ({1})'.format(detector_name, least_missing),
                                  least_missing_points),
                                 ('{0} most new points ({1})'.format(detector_name, most_new), most_new_points),
                                 ('{0} fewest new points ({1})'.format(detector_name, least_new), least_new_points)]:
                if points is not None:
                    image_id, matches, missing_trial, missing_reference = points
                    image = dh.load_object(db_client, db_client.image_collection, image_id)
                    figure = pyplot.figure()
                    figure.suptitle(name)
                    ax = figure.add_subplot(111)
                    ax.imshow(image.data)
                    ax.plot([match[0][0] for match in matches], [match[0][1] for match in matches], 'go')
                    ax.plot([point[0] for point in missing_trial], [point[1] for point in missing_trial], 'rx')

    def _plot_per_image_feature_changes(self, db_client):
        """
        Plot the changes in detected features for all the images.
        This illustrates the size of the intersection between the high and low quality detected features.
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Plotting per-image feature change counts...")
        for detector_name, detector_id in self._feature_detectors.items():
            if detector_id not in self._trial_map:
                continue

            changes = []
            grouped_changes = []
            where = []
            annotations = []
            for trajectory_group in self._trajectory_groups.values():
                for variation in trajectory_group.quality_variations:
                    if variation['dataset'] not in self._trial_map[detector_id]:
                        continue
                    trial_result_id = self._trial_map[detector_id][variation['dataset']]
                    if (trial_result_id in self._result_map and
                            self._benchmark_feature_diff in self._result_map[trial_result_id]):
                        benchmark_result_id = self._result_map[trial_result_id][self._benchmark_feature_diff]
                        benchmark_result = dh.load_object(db_client, db_client.results_collection, benchmark_result_id)
                        for image_changes in benchmark_result.changes:
                            changes.append((
                                len(image_changes['point_matches']),
                                len(image_changes['new_trial_points']),
                                len(image_changes['missing_reference_points'])
                            ))
                            grouped_changes.append((
                                len(image_changes['point_matches']),
                                len(image_changes['new_trial_points']),
                                len(image_changes['missing_reference_points'])
                            ))
                            where.append(True)
                logging.getLogger(__name__).info("... added changes for {0}".format(trajectory_group.name))
                annotations.append(((0, len(grouped_changes)), trajectory_group.name))
                for _ in range(10):
                    grouped_changes.append((0,0,0))
                    where.append(False)

            # Plot an area overlap graph
            #changes.sort(key=lambda d: d[1], reverse=True)
            x = np.arange(len(changes))
            changes = np.array(changes)
            grouped_changes = np.array(grouped_changes)

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Number of changes for {0}".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('number of features')
            ax.set_ylabel('image')
            #ax.fill_between(x, changes[:, 0], color='b', label='matching features')
            ax.fill_betweenx(np.arange(len(grouped_changes)), -1 * grouped_changes[:, 0] / 2,
                             grouped_changes[:, 1] + grouped_changes[:, 0] / 2, where=where,
                             color='b', label='low-quality features', alpha=0.3)
            ax.fill_betweenx(np.arange(len(grouped_changes)), -1 * grouped_changes[:, 2] - grouped_changes[:, 0] / 2,
                             grouped_changes[:, 0] / 2, where=where,
                             color='r', label='high-quality features', alpha=0.3)
            for point, label in annotations:
                ax.annotate(label, xy=point)

            # Plot just the total new/missing
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} total new/missing features".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('image')
            ax.set_ylabel('number of features')
            ax.plot(x, changes[:, 1], label='new low-quality features', linewidth=1)
            ax.plot(x, changes[:, 2], label='missing high-quality features', linewidth=1)
            ax.legend()

            # Plot the relative number of new missing
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} fraction of features matched".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('image')
            ax.set_ylabel('fraction of features')
            ax.plot(x, np.divide(changes[:, 0], changes[:, 0] + changes[:, 1]),
                    label='fraction of low-quality features', linewidth=1)
            ax.plot(x, np.divide(changes[:, 0], changes[:, 0] + changes[:, 2]),
                    label='fraction of high-quality features', linewidth=1)
            ax.legend()

            # Plot precision/recall, just because
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} precision/recall".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('fraction of low-quality features')
            ax.set_ylabel('fraction of high-quality features')
            ax.plot(np.divide(changes[:, 0], changes[:, 0] + changes[:, 1]),
                    np.divide(changes[:, 0], changes[:, 0] + changes[:, 2]), 'bo', markersize=0.5)

            # Plot IoU
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} Intersection over Union".format(detector_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('image')
            ax.set_ylabel('Intersection over Union')
            y = np.divide(changes[:, 0], np.sum(changes, axis=1))
            y.sort()
            ax.plot(x, y, linewidth=1)

    def _plot_trajectories(self, db_client):
        """
        Plot the ground-truth and computed trajectories for each system for each trajectory.
        This is important for validation
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Plotting trajectories...")
        from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plots
        # Make a list of systems and system names to plot.
        systems = [(self._libviso_system, 'LIBVISO 2')]

        # Plot Real-world datasets
        for image_source_id in set(self._kitti_datasets) | set(self._euroc_datasets.values()):
            # Make the trajectory comparison figure
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Computed trajectories for {0}".format(image_source_id))
            ax = figure.add_subplot(111, projection='3d')
            ax.set_xlabel('x-location')
            ax.set_ylabel('y-location')
            ax.set_zlabel('z-location')
            ax.plot([0], [0], [0], 'ko', label='origin')
            lower_limit = 0
            upper_limit = 0
            added_ground_truth = False

            for system_id, system_name in systems:
                if (system_id not in self._trial_map or image_source_id not in self._trial_map[system_id]):
                    # Skip systems that have not run on this image source
                    continue

                # Plot the max quality trajectory
                trial_result = dh.load_object(db_client, db_client.trials_collection,
                                              self._trial_map[system_id][image_source_id])
                if trial_result is None:
                    continue
                if not added_ground_truth:
                    minp, maxp = plot_trajectory(ax, trial_result.get_ground_truth_camera_poses(),
                                                 'ground-truth trajectory')
                    lower_limit = min(lower_limit, minp)
                    upper_limit = max(upper_limit, maxp)
                    added_ground_truth = True  # Ground truth trajectory should be the same for all in this group.
                minp, maxp = plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                             'max-quality trajectory for {}'.format(system_name))
                lower_limit = min(lower_limit, minp)
                upper_limit = max(upper_limit, maxp)
            if added_ground_truth:
                logging.getLogger(__name__).info("... plotted trajectories for {0}".format(image_source_id))
                ax.legend()
                ax.set_xlim(lower_limit, upper_limit)
                ax.set_ylim(lower_limit, upper_limit)
                ax.set_zlim(lower_limit, upper_limit)
            else:
                logging.getLogger(__name__).warning("Failed to get trajectories for {0}".format(image_source_id))

        # Trajectory visualization: For each system and each trajectory, plot the different paths
        for trajectory_group in self._trajectory_groups.values():
            # Make the trajectory comparison figure
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Computed trajectories for {0}".format(trajectory_group.name))
            ax = figure.add_subplot(111, projection='3d')
            ax.set_xlabel('x-location')
            ax.set_ylabel('y-location')
            ax.set_zlabel('z-location')
            ax.plot([0], [0], [0], 'ko', label='origin')
            lower_limit = 0
            upper_limit = 0
            added_ground_truth = False

            # For each system variation over this trajectory
            for system_id, system_name in systems:
                if (system_id not in self._trial_map or
                        trajectory_group.max_quality_id not in self._trial_map[system_id]):
                    # Skip systems that have not run on this trajectory group
                    continue

                # Plot the max quality trajectory
                trial_result = dh.load_object(db_client, db_client.trials_collection,
                                              self._trial_map[system_id][trajectory_group.max_quality_id])
                if trial_result is None:
                    continue
                if not added_ground_truth:
                    minp, maxp = plot_trajectory(ax, trial_result.get_ground_truth_camera_poses(),
                                                 'ground-truth trajectory')
                    lower_limit = min(lower_limit, minp)
                    upper_limit = max(upper_limit, maxp)
                    added_ground_truth = True  # Ground truth trajectory should be the same for all in this group.
                minp, maxp = plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                'max-quality trajectory for {}'.format(system_name))
                lower_limit = min(lower_limit, minp)
                upper_limit = max(upper_limit, maxp)

                # Plot the trajectories for each quality variant. We've only got one right now
                for variation in trajectory_group.quality_variations:
                    if variation['dataset'] in self._trial_map[system_id]:
                        trial_result = dh.load_object(db_client, db_client.trials_collection,
                                                      self._trial_map[system_id][variation['dataset']])
                        minp, maxp = plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                                     'min-quality trajectory for {}'.format(system_name))
                        lower_limit = min(lower_limit, minp)
                        upper_limit = max(upper_limit, maxp)
            if added_ground_truth:
                logging.getLogger(__name__).info("... plotted trajectories for {0}".format(trajectory_group.name))
                ax.legend()
                ax.set_xlim(lower_limit, upper_limit)
                ax.set_ylim(lower_limit, upper_limit)
                ax.set_zlim(lower_limit, upper_limit)
            else:
                logging.getLogger(__name__).warning("Failed to get trajectories for {0}".format(trajectory_group.name))

    def _plot_aggregate_performance(self, db_client):
        """
        Plot the aggregate performance of real world, high-quality and low quality
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Plotting aggregate performance...")
        # Make a list of systems and system names to plot.
        systems = [(self._libviso_system, 'LIBVISO 2')]

        # Make a list of benchmarks, names, and lambdas for aggregate statistic extraction
        benchmarks = [
            (self._benchmark_rpe, 'Relative Pose Error (Translation)', lambda r: list(r.translational_error.values())),
            (self._benchmark_rpe, 'Relative Pose Error (Rotation)', lambda r: list(r.rotational_error.values())),
            (self._benchmark_ate, 'Absolute Trajectory Error', lambda r: list(r.translational_error.values())),
            (self._benchmark_trajectory_drift, 'Trajectory Drift (Translation)',
             operator.attrgetter('translational_error')),
            (self._benchmark_trajectory_drift, 'Trajectory Drift (Rotation)',
             operator.attrgetter('rotational_error')),
#            (self._benchmark_tracking, 'Tracking Failure', lambda r: list(r.distances))
        ]

        # Make a list of real-world datasets
        real_world_datasets = self._kitti_datasets | self._tum_manager.all_datasets | set(self._euroc_datasets.values())

        # Step 3 - Aggregation: For each benchmark, compare real-world and different qualities
        for benchmark_id, benchmark_name, values_list_getter in benchmarks:
            data = []
            labels = []
            for system_id, system_name in systems:
                if system_id not in self._trial_map:
                    continue  # Skip systems for which we have no trials on record.
                real_world_results = []
                max_quality_results = []
                min_quality_results = []

                # Add results for real-world data
                for image_source_id in real_world_datasets:
                    if image_source_id in self._trial_map[system_id]:
                        trial_result_id = self._trial_map[system_id][image_source_id]
                        if trial_result_id in self._result_map and benchmark_id in self._result_map[trial_result_id]:
                            benchmark_result = dh.load_object(db_client, db_client.results_collection,
                                                              self._result_map[trial_result_id][benchmark_id])
                            if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                print(benchmark_result.reason)
                            else:
                                real_world_results += values_list_getter(benchmark_result)

                # Add results for synthetic data
                for trajectory_group in self._trajectory_groups.values():
                    if trajectory_group.max_quality_id in self._trial_map[system_id]:
                        trial_result_id = self._trial_map[system_id][trajectory_group.max_quality_id]
                        if trial_result_id in self._result_map and benchmark_id in self._result_map[trial_result_id]:
                            benchmark_result = dh.load_object(db_client, db_client.results_collection,
                                                              self._result_map[trial_result_id][benchmark_id])
                            if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                print(benchmark_result.reason)
                            else:
                                max_quality_results += values_list_getter(benchmark_result)

                    for variation in trajectory_group.quality_variations:
                        if variation['dataset'] in self._trial_map[system_id]:
                            trial_result_id = self._trial_map[system_id][variation['dataset']]
                            if (trial_result_id in self._result_map and
                                    benchmark_id in self._result_map[trial_result_id]):
                                benchmark_result = dh.load_object(db_client, db_client.results_collection,
                                                                  self._result_map[trial_result_id][benchmark_id])
                                if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                    print(benchmark_result.reason)
                                else:
                                    min_quality_results += values_list_getter(benchmark_result)

                data.append(real_world_results)
                labels.append('{} - Real world'.format(system_name))
                data.append(max_quality_results)
                labels.append('{} - Max quality'.format(system_name))
                data.append(min_quality_results)
                labels.append('{} - Min quality'.format(system_name))

            logging.getLogger(__name__).info("Completed plot for {0}".format(benchmark_name))
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0}".format(benchmark_name))
            ax = figure.add_subplot(111)
            ax.boxplot(data)
            ax.set_xticklabels(labels)

    def _plot_performance_vs_time(self, db_client):
        """
        Perform detailed plots of performance vs time for high and low quality.
        This lets us look at places where the difference between high and low quality is pronounced
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        logging.getLogger(__name__).info("Plottng performance over time for all systems...")
        # Make a list of systems and system names to plot.
        systems = [(self._libviso_system, 'LIBVISO 2')]

        # Benchmark results over time, for each trajectory and system
        for system_id, system_name in systems:
            if system_id not in self._trial_map:  # Skip systems with no trials
                continue
            for trajectory_group in self._trajectory_groups.values():

                # Make figures for each benchmark
                figure = pyplot.figure(figsize=(14, 10), dpi=80)
                figure.suptitle("{0} changes for {1}".format(system_name, trajectory_group.name))
                rpe_ax = figure.add_subplot(211)
                rpe_ax.set_xlabel('Timestamp')
                rpe_ax.set_ylabel('Relative Pose Error (translational)')
                ate_ax = figure.add_subplot(212)
                ate_ax.set_xlabel('Timestamp')
                ate_ax.set_ylabel('Absolute Trajectory Error')

                if trajectory_group.max_quality_id in self._trial_map[system_id]:
                    # Results for max quality run
                    trial_result_id = self._trial_map[system_id][trajectory_group.max_quality_id]
                    if trial_result_id in self._result_map:
                        # RPE translational results
                        if self._benchmark_rpe in self._result_map[trial_result_id]:
                            benchmark_result = dh.load_object(
                                db_client, db_client.results_collection,
                                self._result_map[trial_result_id][self._benchmark_rpe]
                            )
                            if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                print(benchmark_result.reason)
                            else:
                                times = sorted(benchmark_result.translational_error.keys())
                                rpe_ax.plot(times, [benchmark_result.translational_error[time] for time in times],
                                            label='Max quality')
                        # ATE results
                        if self._benchmark_ate in self._result_map[trial_result_id]:
                            benchmark_result = dh.load_object(
                                db_client, db_client.results_collection,
                                self._result_map[trial_result_id][self._benchmark_ate]
                            )
                            if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                print(benchmark_result.reason)
                            else:
                                times = sorted(benchmark_result.translational_error.keys())
                                ate_ax.plot(times, [benchmark_result.translational_error[time] for time in times],
                                            label='Max quality')

                # Results for lower quality variations
                for variation in trajectory_group.quality_variations:
                    if variation['dataset'] in self._trial_map[system_id]:
                        trial_result_id = self._trial_map[system_id][variation['dataset']]
                        if trial_result_id in self._result_map:
                            # RPE translational results
                            if self._benchmark_rpe in self._result_map[trial_result_id]:
                                benchmark_result = dh.load_object(
                                    db_client, db_client.results_collection,
                                    self._result_map[trial_result_id][self._benchmark_rpe]
                                )
                                if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                    print(benchmark_result.reason)
                                else:
                                    times = sorted(benchmark_result.translational_error.keys())
                                    rpe_ax.plot(times, [benchmark_result.translational_error[time] for time in times],
                                                label='Min quality')
                            # ATE results
                            if self._benchmark_ate in self._result_map[trial_result_id]:
                                benchmark_result = dh.load_object(
                                    db_client, db_client.results_collection,
                                    self._result_map[trial_result_id][self._benchmark_ate]
                                )
                                if isinstance(benchmark_result, core.benchmark.FailedBenchmark):
                                    print(benchmark_result.reason)
                                else:
                                    times = sorted(benchmark_result.translational_error.keys())
                                    ate_ax.plot(times, [benchmark_result.translational_error[time] for time in times],
                                                label='Min quality')

    def serialize(self):
        serialized = super().serialize()
        # Systems
        serialized['feature_detectors'] = self._feature_detectors
        serialized['libviso'] = self._libviso_system
        serialized['orbslam_systems'] = self._orbslam_systems

        # Real-world datasets
        serialized['kitti_datasets'] = list(self._kitti_datasets)
        serialized['tum_datasets'] = self._tum_manager.serialize()
        serialized['euroc_datasets'] = self._euroc_datasets

        # Generated Datasets
        serialized['simulators'] = self._simulators
        serialized['flythrough_controller'] = self._flythrough_controller
        serialized['trajectory_groups'] = {str(max_id): group.serialize()
                                           for max_id, group in self._trajectory_groups.items()}

        # Benchmarks
        serialized['benchmark_feature_diff'] = self._benchmark_feature_diff
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_ate'] = self._benchmark_ate
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        serialized['benchmark_tracking'] = self._benchmark_tracking

        # Trials
        serialized['trial_map'] = {str(sys_id): {str(source_id): trial_id for source_id, trial_id in inner_map.items()}
                                   for sys_id, inner_map in self._trial_map.items()}
        serialized['result_map'] = {str(trial_id): {str(bench_id): res_id for bench_id, res_id in inner_map.items()}
                                    for trial_id, inner_map in self._result_map.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        # Systems
        if 'feature_detectors' in serialized_representation:
            kwargs['feature_detectors'] = serialized_representation['feature_detectors']
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_systems' in serialized_representation:
            kwargs['orbslam_systems'] = serialized_representation['orbslam_systems']

        # Real-world datasets
        if 'kitti_datasets' in serialized_representation:
            kwargs['kitti_datasets'] = set(serialized_representation['kitti_datasets'])
        if 'tum_datasets' in serialized_representation:
            kwargs['tum_manager'] = dataset.tum.tum_manager.TUMManager.deserialize(
                serialized_representation['tum_datasets'])
        if 'euroc_datasets' in serialized_representation:
            kwargs['euroc_datasets'] = serialized_representation['euroc_datasets']

        # Generated datasets
        if 'simulators' in serialized_representation:
            kwargs['simulators'] = serialized_representation['simulators']
        if 'flythrough_controller' in serialized_representation:
            kwargs['flythrough_controller'] = serialized_representation['flythrough_controller']
        if 'trajectory_groups' in serialized_representation:
            kwargs['trajectory_groups'] = {bson.ObjectId(max_id): TrajectoryGroup.deserialize(s_group)
                                           for max_id, s_group in
                                           serialized_representation['trajectory_groups'].items()}

        # Benchmarks
        if 'benchmark_feature_diff' in serialized_representation:
            kwargs['benchmark_feature_diff'] = serialized_representation['benchmark_feature_diff']
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_ate' in serialized_representation:
            kwargs['benchmark_ate'] = serialized_representation['benchmark_ate']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        if 'benchmark_tracking' in serialized_representation:
            kwargs['benchmark_tracking'] = serialized_representation['benchmark_tracking']

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


class PlaceholderImageCollection(core.image_source.ImageSource):
    """
    A placeholder for an image collection which doesn't actually have any images,
    but will give the same answers to things like is_stored_in_database and is_labels_available.
    Used for preliminary checking if an image collection is appropriate
    """

    def __init__(self, id_, is_labels_available, is_per_pixel_labels_available, is_normals_available,
                 is_depth_available, is_stereo_available, sequence_type):
        self._id = id_
        self._is_labels_available = is_labels_available
        self._is_per_pixel_labels_available = is_per_pixel_labels_available
        self._is_normals_available = is_normals_available
        self._is_depth_available = is_depth_available
        self._is_stereo_available = is_stereo_available
        self._sequence_type = core.sequence_type.ImageSequenceType(sequence_type)

    @property
    def identifier(self):
        return self._id

    @property
    def is_stored_in_database(self):
        return True

    @property
    def is_labels_available(self):
        return self._is_labels_available

    @property
    def is_per_pixel_labels_available(self):
        return self._is_per_pixel_labels_available

    @property
    def is_normals_available(self):
        return self._is_normals_available

    @property
    def is_depth_available(self):
        return self._is_depth_available

    @property
    def is_stereo_available(self):
        return self._is_stereo_available

    @property
    def supports_random_access(self):
        return True

    def get_camera_intrinsics(self):
        return cam_intr.CameraIntrinsics(fx=1, fy=1, cx=0.5, cy=0.5), (640, 480)

    @property
    def sequence_type(self):
        return self._sequence_type

    def get_next_image(self):
        return None, None

    def begin(self):
        pass

    def get(self, index):
        return None

    def is_complete(self):
        return True


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
    axis.plot(x, y, z, label=label)
    return min_point, max_point


def get_trajectory_for_image_source(db_client, image_collection_id):
    """
    Image collections are too large for us to load into memory here,
    but we need to be able to do logic on their trajectores.
    This utility uses the database to get just the trajectory for a given image collection.
    Only works for image collections, due to their database structure.
    :param db_client: The database client
    :param image_collection_id: The id of the image collection to load
    :return: A trajectory, a map of timestamp to camera pose. Ignores right-camera for stereo
    """
    images = db_client.image_source_collection.find_one({'_id': image_collection_id, 'images': {'$exists': True}},
                                                        {'images': True})
    trajectory = {}
    if images is not None:
        for timestamp, image_id in images['images']:
            position_result = db_client.image_collection.find_one({'_id': image_id}, {'metadata.camera_pose': True})
            if position_result is not None:
                trajectory[timestamp] = tf.Transform.deserialize(position_result['metadata']['camera_pose'])
    return trajectory


def are_matching_trajectories(trajectory1, trajectory2, location_threshold=10, time_threshold=1):
    """
    For the purposes of this experiment, do two image sources follow the same trajectory.
    Because of the simulation, we expect trajectories to be either exactly the same or completely different
    :param trajectory1: map of timestamps to poses
    :param trajectory2: map of timestamps to poses
    :param location_threshold: The maximum location error before trajectories are considered different. Default 10.
    :param time_threshold: The maximum difference in timestamp for matching trajectories. Default 1.
    :return:
    """
    timestamp_matches = util.associate.associate(trajectory1, trajectory2, offset=0, max_difference=time_threshold)
    if float(len(timestamp_matches)) < 0.9 * min(len(trajectory1), len(trajectory2)):
        return False
    square_threshold = location_threshold * location_threshold
    for traj1_stamp, traj2_stamp in timestamp_matches:
        dist = trajectory1[traj1_stamp].location - trajectory2[traj2_stamp].location
        if np.dot(dist, dist) > square_threshold:
            return False
    return True
