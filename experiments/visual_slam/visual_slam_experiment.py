import os
import bson
import numpy as np
import util.database_helpers as dh
import util.associate
import util.transform as tf
import util.dict_utils as du
import core.image_source
import core.sequence_type
import metadata.camera_intrinsics as cam_intr
import metadata.image_metadata as imeta
import batch_analysis.experiment
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import simulation.unrealcv.unrealcv_simulator as uecv_sim
import simulation.controllers.flythrough_controller as fly_cont
import simulation.controllers.trajectory_follow_controller as follow_cont


class TrajectoryGroup:

    def __init__(self, simulator_id, default_simulator_config, max_quality_id,
                 controller_id=None, quality_variations=None):
        self.simulator_id = simulator_id
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
            trajectory = get_trajectory_for_image_source(db_client, self.max_quality_id)
            self.follow_controller_id = dh.add_unique(db_client.image_source_collection,
                                                      follow_cont.TrajectoryFollowController(
                                                          trajectory=trajectory,
                                                          sequence_type=core.sequence_type.ImageSequenceType.SEQUENTIAL
                                                      ))
            changed = True
        # Next, schedule reduced quality dataset generation for each desired config
        # These are all the quality variations we're going to do for now.
        for config in [{
            'resolution': {'width': 256, 'height': 144}  # Extremely low res
        }, {
            'fov': 15   # Narrow field of view
        }, {
            'depth_of_field_enabled': False   # No dof
        }, {
            'lit_mode': False  # Unlit
        }, {
            'texture_mipmap_bias': 8  # No textures
        }, {
            'normal_maps_enabled': False,  # No normal maps
        }, {
            'roughness_enabled': False  # No reflections
        }, {
            'geometry_decimation': 4,   # Simple geometry
        }, {
            # low quality
            'depth_of_field_enabled': False,
            'lit_mode': False,
            'texture_mipmap_bias': 8,
            'normal_maps_enabled': False,
            'roughness_enabled': False,
            'geometry_decimation': 4,
        }, {
            # absolute minimum quality - yall got any more of them pixels
            'resolution': {'width': 256, 'height': 144},
            'fov': 15,
            'depth_of_field_enabled': False,
            'lit_mode': False,
            'texture_mipmap_bias': 8,
            'normal_maps_enabled': False,
            'roughness_enabled': False,
            'geometry_decimation': 4,
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
                    simulator_config=du.defaults(config, self.default_simulator_config),
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
            'default_simulator_config': self.default_simulator_config,
            'max_quality_id': self.max_quality_id,
            'controller_id': self.follow_controller_id,
            'quality_variations': self.quality_variations
        }

    @classmethod
    def deserialize(cls, serialized_representation):
        return cls(
            simulator_id=serialized_representation['simulator_id'],
            default_simulator_config=serialized_representation['default_simulator_config'],
            max_quality_id=serialized_representation['max_quality_id'],
            controller_id=serialized_representation['controller_id'],
            quality_variations=serialized_representation['quality_variations']
        )


class VisualSlamExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None, orbslam_systems=None, benchmark_rpe=None, benchmark_trajectory_drift=None,
                 simulators=None, flythrough_controller=None, trajectory_groups=None,
                 real_world_datasets=None, trial_list=None, result_list=None, id_=None):
        super().__init__(id_=id_)
        self._libviso_system = libviso_system
        self._orbslam_systems = set(orbslam_systems) if orbslam_systems is not None else set()
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_trajectory_drift = benchmark_trajectory_drift
        self._simulators = set(simulators) if simulators is not None else set()
        self._flythrough_controller = flythrough_controller
        self._trajectory_groups = trajectory_groups if trajectory_groups is not None else {}
        self._real_world_datasets = set(real_world_datasets) if real_world_datasets is not None else set()
        self._trial_list = trial_list if trial_list is not None else []
        self._result_list = result_list if result_list is not None else []
        self._placeholder_image_collections = {}

    def do_imports(self, task_manager, db_client):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
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
            self._simulators.add(sim_id)
            self._add_to_set('simulators', {sim_id})

        # Add controllers
        if self._flythrough_controller is None:
            self._flythrough_controller = dh.add_unique(db_client.image_source_collection,
                                                        fly_cont.FlythroughController(
                                                            max_speed=0.2,
                                                            max_turn_angle=np.pi / 36,
                                                            avoidance_radius=1,
                                                            avoidance_scale=1,
                                                            length=100,
                                                            seconds_per_frame=1/10
                                                        ))
            self._set_property('flythrough_controller', self._flythrough_controller)

        # Generate max-quality flythroughs
        trajectories_per_environment = 3
        for simulator_id in self._simulators:
            for repeat in range(trajectories_per_environment):
                # Default, maximum quality settings. We override specific settings in the quality group
                simulator_config = {
                    # Simulation execution config
                    'stereo_offset': 0.15,  # meters
                    'provide_rgb': True,
                    'provide_depth': True,
                    'provide_labels': False,
                    'provide_world_normals': False,

                    # Simulator settings - No clear maximum, but these are the best we'll use
                    'resolution': {'width': 1280, 'height': 720},
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
                if generate_dataset_task.is_finished:
                    result_ids = generate_dataset_task.result
                    if isinstance(result_ids, bson.ObjectId):   # we got a single id
                        result_ids = [result_ids]
                    for result_id in result_ids:
                        if result_id not in self._trajectory_groups:
                            self._trajectory_groups[result_id] = TrajectoryGroup(
                                simulator_id=simulator_id,
                                default_simulator_config=simulator_config,
                                max_quality_id=result_id)
                            self._set_property('trajectory_groups.{0}'.format(result_id),
                                               self._trajectory_groups[result_id].serialize())
                else:
                    task_manager.do_task(generate_dataset_task)

        # Schedule dataset generation for lower-quality datasets in each trajectory group
        for trajectory_group in self._trajectory_groups.values():
            if trajectory_group.do_imports(task_manager, db_client):
                self._set_property('trajectory_groups.{0}'.format(trajectory_group.max_quality_id),
                                   trajectory_group.serialize())

        # --------- REAL WORLD DATASETS -----------
        # Import KITTI dataset
        import_dataset_task = task_manager.get_import_dataset_task(
            module_name='dataset.kitti.kitti_loader',
            path=os.path.expanduser(os.path.join('~', 'datasets', 'KITTI', 'dataset')),
            num_cpus=1,
            num_gpus=0,
            memory_requirements='3GB',
            expected_duration='72:00:00'
        )
        if import_dataset_task.is_finished:
            imported_ids = import_dataset_task.result
            if not isinstance(import_dataset_task.result, list):
                imported_ids = [imported_ids]
            for imported_id in imported_ids:
                if imported_id not in self._real_world_datasets:
                    self._real_world_datasets.add(imported_id)
                    self._add_to_set('real_world_datasets', {imported_id})
        else:
            task_manager.do_task(import_dataset_task)

        # --------- SYSTEMS -----------
        # Import the systems under test for this experiment.
        # They are: libviso2, orbslam2
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)
        # ORBSLAM2 - 12 variants for parameter sweep
        settings_list = [
            (sensor_mode, n_features, scale_factor, resolution)
            for sensor_mode in {orbslam2.SensorMode.MONOCULAR, orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD}
            for n_features in {1000, 2000}
            for scale_factor in {1.2}
            for resolution in {(1280, 720), (640, 480)}
        ]
        if len(self._orbslam_systems) < len(settings_list):
            for settings in settings_list:
                orbslam_id = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                    vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
                    mode=settings[0],
                    settings={
                        'ORBextractor': {
                            'nFeatures': settings[1],
                            'scaleFactor': settings[2]
                        }
                    }, resolution=settings[3]
                ))
                self._orbslam_systems.add(orbslam_id)
                self._add_to_set('orbslam_systems', {orbslam_id})

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
        if self._benchmark_trajectory_drift is None:
            self._benchmark_trajectory_drift = dh.add_unique(
                db_client.benchmarks_collection,
                traj_drift.BenchmarkTrajectoryDrift(
                    segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
                    step_size=10
                ))
            self._set_property('benchmark_trajectory_drift', self._benchmark_trajectory_drift)

    def schedule_tasks(self, task_manager, db_client):
        libviso_system = dh.load_object(db_client, db_client.system_collection, self._libviso_system)
        orbslam_systems = [
            dh.load_object(db_client, db_client.system_collection, system_id)
            for system_id in self._orbslam_systems
        ]
        benchmarks = [
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_rpe),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_trajectory_drift)
        ]
        datasets = set(self._real_world_datasets)
        for group in self._trajectory_groups.values():
            datasets = datasets | group.get_all_dataset_ids()

        # Schedule trials
        for image_source_id in datasets:
            image_source = self._load_image_source(db_client, image_source_id)
            # Libviso2
            if libviso_system.is_image_source_appropriate(image_source):
                task = task_manager.get_run_system_task(
                    system_id=libviso_system.identifier,
                    image_source_id=image_source.identifier,
                    expected_duration='4:00:00'
                )
                if task.is_finished:
                    found = False
                    for _, _, trial_result_id in self._trial_list:
                        if trial_result_id == task.result:
                            found = True
                            break
                    if not found:
                        trial_tuple = (image_source_id, self._libviso_system, task.result)
                        self._trial_list.append(trial_tuple)
                        self._add_to_list('trial_list', [trial_tuple])
                else:
                    task_manager.do_task(task)

            # ORBSLAM2
            for orbslam_system in orbslam_systems:
                if orbslam_system.is_image_source_appropriate(image_source):
                    task = task_manager.get_run_system_task(
                        system_id=orbslam_system.identifier,
                        image_source_id=image_source.identifier,
                        expected_duration='4:00:00'
                    )
                    if task.is_finished:
                        found = False
                        for _, _, trial_result_id in self._trial_list:
                            if trial_result_id == task.result:
                                found = True
                                break
                        if not found:
                            trial_tuple = (image_source_id, orbslam_system.identifier, task.result)
                            self._trial_list.append(trial_tuple)
                            self._add_to_list('trial_list', [trial_tuple])
                    else:
                        task_manager.do_task(task)

        # Benchmark results
        for image_source_id, system_id, trial_result_id in self._trial_list:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            for benchmark in benchmarks:
                if benchmark.is_trial_appropriate(trial_result):
                    task = task_manager.get_benchmark_task(
                        trial_result_id=trial_result.identifier,
                        benchmark_id=benchmark.identifier,
                        expected_duration='4:00:00'
                    )
                    if task.is_finished:
                        found = False
                        for _, _, _, benchmark_id in self._result_list:
                            if benchmark_id == task.result:
                                found = True
                                break
                        if not found:
                            result_tuple = (image_source_id, system_id, trial_result_id, task.result)
                            self._result_list.append(result_tuple)
                            self._add_to_list('result_list', [result_tuple])
                    else:
                        task_manager.do_task(task)

    def plot_results(self, db_client):
        """
        Plot the results for this experiment.
        I don't know how we want to do this yet.
        :param db_client:
        :return:
        """
        # Plotting imports are here only
        import pandas
        import matplotlib.pyplot as pyplot
        from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plots

        # Step 1: Group trials and results by trajectory, and by each image quality
        trajectory_map = []
        trials_by_trajectory = []
        results_by_trajectory = []

        for image_source_id, system_id, trial_result_id in self._trial_list:
            image_source = self._load_image_source(db_client, image_source_id)
            metadata_key = self._make_metadata_key(image_source, trajectory_map)
            trials_by_trajectory[metadata_key[0]].append(
                dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            )
        for image_source_id, system_id, trial_result_id, benchmark_result_id in self._result_list:
            image_source = self._load_image_source(db_client, image_source_id)
            metadata_key = self._make_metadata_key(image_source, trajectory_map)
            results_by_trajectory[metadata_key[0]].append(
                dh.load_object(db_client, db_client.trials_collection, benchmark_result_id)
            )

        # TODO: Once we've grouped the trial results, plot them by group
        trial_results = []
        for trial_result_id in trial_results:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
            result_traj = trial_result.get_computed_camera_poses()
            matches = util.associate.associate(ground_truth_traj, result_traj, offset=0, max_difference=1)
            x = []
            y = []
            z = []
            x_gt = []
            y_gt = []
            z_gt = []
            gt_start = ground_truth_traj[min(ground_truth_traj.keys())]
            for gt_stamp, result_stamp in matches:
                gt_relative_pose = gt_start.find_relative(ground_truth_traj[gt_stamp])
                x_gt.append(gt_relative_pose.location[0])
                y_gt.append(gt_relative_pose.location[1])
                z_gt.append(gt_relative_pose.location[2])
                x.append(result_traj[result_stamp].location[0])
                y.append(result_traj[result_stamp].location[1])
                z.append(result_traj[result_stamp].location[2])

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            ax = figure.add_subplot(111, projection='3d')
            ax.plot(x, y, z, label='computed trajectory')
            ax.plot(x_gt, y_gt, z_gt, label='ground-truth trajectory')
            ax.set_xlabel('x-location')
            ax.set_ylabel('y-location')
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

    def _get_trajectory_id(self, trajectory, trajectory_map):
        """
        Get an id number in the trajectory map matching the given trajectory.
        If there are no matching trajectories, store it in the map and return the new index
        :param trajectory: A trajectory, as timestamps to poses
        :return: Integer trajectory id which is the index to the trajectory map
        """
        for idx, mapped_trajectory in enumerate(trajectory_map):
            if are_matching_trajectories(trajectory, mapped_trajectory):
                return idx
        trajectory_map.append(trajectory)
        return len(trajectory_map) - 1

    def _make_metadata_key(self, dataset, trajectory_map):
        """
        Make a vector of the properties we care about for associating results from a dataset
        :param dataset: The image collection
        :param trajectory_map: A collection of existing trajectories, for grouping
        :return:
        """
        trajectory = extract_trajectory(dataset)
        trajectory_id = self._get_trajectory_id(trajectory, trajectory_map)
        first_image = dataset.get(0)
        return (
            trajectory_id,
            first_image.metadata.source_type.value,
            first_image.metadata.texture_mipmap_bias,
            int(first_image.metadata.texture_mipmap_bias),
            int(first_image.metadata.roughness_enabled),
            first_image.metadata.geometry_decimation
        )

    def serialize(self):
        serialized = super().serialize()
        serialized['libviso'] = self._libviso_system
        serialized['orbslam_systems'] = list(self._orbslam_systems)
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        serialized['simulators'] = list(self._simulators)
        serialized['flythrough_controller'] = self._flythrough_controller
        serialized['trajectory_groups'] = {str(max_id): group.serialize()
                                           for max_id, group in self._trajectory_groups.items()}
        serialized['real_world_datasets'] = list(self._real_world_datasets)
        serialized['trial_list'] = self._trial_list
        serialized['result_list'] = self._result_list
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_systems' in serialized_representation:
            kwargs['orbslam_systems'] = serialized_representation['orbslam_systems']
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        if 'simulators' in serialized_representation:
            kwargs['simulators'] = serialized_representation['simulators']
        if 'flythrough_controller' in serialized_representation:
            kwargs['flythrough_controller'] = serialized_representation['flythrough_controller']
        if 'trajectory_groups' in serialized_representation:
            kwargs['trajectory_groups'] = {bson.ObjectId(max_id): TrajectoryGroup.deserialize(s_group)
                                           for max_id, s_group in
                                           serialized_representation['trajectory_groups'].items()}
        if 'real_world_datasets' in serialized_representation:
            kwargs['real_world_datasets'] = serialized_representation['real_world_datasets']
        if 'trial_list' in serialized_representation:
            kwargs['trial_list'] = serialized_representation['trial_list']
        if 'result_list' in serialized_representation:
            kwargs['result_list'] = serialized_representation['result_list']
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
        return cam_intr.CameraIntrinsics(fx=1, fy=1, cx=0.5, cy=0.5)

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


def extract_trajectory(image_source):
    """
    Extract a trajectory from an image source, which will almost always be an image collection.
    :param image_source: The image source to extract from
    :return: The ground-truth camera trajectory, as a dict of timestamp to pose
    """
    trajectory = {}
    with image_source:
        while not image_source.is_complete():
            image, timestamp = image_source.get_next_image()
            trajectory[timestamp] = image.metadata.camera_pose
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
