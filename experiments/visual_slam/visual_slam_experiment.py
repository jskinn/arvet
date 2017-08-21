import os
import glob
import numpy as np
import pandas
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D     # Necessary for 3D plots
import util.database_helpers as dh
import util.associate
import core.sequence_type
import batch_analysis.experiment
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.trajectory_drift.trajectory_drift as traj_drift
import image_collections.looping_collection


class VisualSlamExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None, orbslam_map=None, benchmark_rpe=None, benchmark_trajectory_drift=None,
                 datasets=None, looped_datasets=None, trajectory_map=None, trial_list=None, result_list=None, id_=None):
        super().__init__(id_=id_)
        self._libviso_system = libviso_system
        self._orbslam_map = orbslam_map if orbslam_map is not None else {}
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_trajectory_drift = benchmark_trajectory_drift
        self._datasets = set(datasets) if datasets is not None else set()
        self._looped_datasets = set(looped_datasets) if looped_datasets is not None else set()
        self._trajectory_map = trajectory_map if trajectory_map is not None else []
        self._trial_list = trial_list if trial_list is not None else []
        self._result_list = result_list if result_list is not None else []
        self._image_source_cache = {}

    def do_imports(self, task_manager, db_client):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
        # Import existing datasets
        for path in glob.iglob(os.path.expanduser('~/Renders/Visual Realism/Experiment 1/**/**/metadata.json')):
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name='dataset.generated.import_generated_dataset',
                path=path,
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='4:00:00'
            )
            if import_dataset_task.is_finished:
                if import_dataset_task.result not in self._datasets:
                    imported_ids = import_dataset_task.result
                    if not isinstance(import_dataset_task.result, list):
                        imported_ids = [imported_ids]
                    for imported_id in imported_ids:
                        self._datasets.add(imported_id)
                        image_source = self._load_image_source(db_client, imported_id)
                        self._looped_datasets.add(dh.add_unique(
                            db_client.image_source_collection,
                            image_collections.looping_collection.LoopingCollection(
                                image_source, 3, core.sequence_type.ImageSequenceType.SEQUENTIAL
                            )
                        ))
            else:
                task_manager.do_task(import_dataset_task)

        # Import the systems under test for this experiment.
        # They are: libviso2, orbslam2
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem(
                # TODO: Get accurate camera focal information.
                # Alternately, get this ground truth from the image source when it is run.
                focal_distance=1,
                cu=640,
                cv=360,
                base=30
            ))
            self._set_property('libviso', self._libviso_system)
        # ORBSLAM2 - 12 variants for parameter sweep
        settings_list = [
            (sensor_mode, n_features, scale_factor, resolution)
            for sensor_mode in {orbslam2.SensorMode.MONOCULAR, orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD}
            for n_features in {1000, 2000}
            for scale_factor in {1.2}
            for resolution in {(1280, 720), (640, 480)}
        ]
        for settings in settings_list:
            settings_key = (settings[0].value, settings[1], settings[2], settings[3][0], settings[3][1])
            if settings_key not in self._orbslam_map:
                self._orbslam_map[settings_key] = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                    vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
                    mode=settings[0],
                    settings={
                        'ORBextractor': {
                            'nFeatures': settings[1],
                            'scaleFactor': settings[2]
                        }
                    }, resolution=settings[3]
                ))

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
            for system_key, system_id in self._orbslam_map.items()
        ]
        benchmarks = [
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_rpe),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_trajectory_drift)
        ]

        # Schedule trials
        for image_source_id in self._looped_datasets:
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
                        self._trial_list.append((image_source_id, self._libviso_system, task.result))
                else:
                    task_manager.do_task(task)

            # ORBSLAM2
            for orbslam_system in orbslam_systems:
                if orbslam_system.is_image_source_appropriate(image_source):
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
                            self._trial_list.append((image_source_id, orbslam_system.identifier, task.result))
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
                            self._result_list.append((image_source_id, system_id, trial_result_id, task.result))
                    else:
                        task_manager.do_task(task)

    def plot_results(self, db_client):
        """
        Plot the results for this experiment.
        I don't know how we want to do this yet.
        :param db_client:
        :return:
        """
        # Step 1: Group trials and results by trajectory, and by each image quality


        trials_by_trajectory = [[] for _ in range(len(self._trajectory_map))]
        results_by_trajectory = [[] for _ in range(len(self._trajectory_map))]

        for image_source_id, system_id, trial_result_id in self._trial_list:
            image_source = self._load_image_source(db_client, image_source_id)
            metadata_key = self._make_metadata_key(image_source)
            trials_by_trajectory[metadata_key[0]].append(
                dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            )
        for image_source_id, system_id, trial_result_id, benchmark_result_id in self._result_list:
            image_source = self._load_image_source(db_client, image_source_id)
            metadata_key = self._make_metadata_key(image_source)
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
        if image_source_id not in self._image_source_cache:
            self._image_source_cache[image_source_id] = dh.load_object(db_client, db_client.image_source_collection,
                                                                       image_source_id)
        return self._image_source_cache[image_source_id]

    def _get_trajectory_id(self, trajectory):
        """
        Get an id number in the trajectory map matching the given trajectory.
        If there are no matching trajectories, store it in the map and return the new index
        :param trajectory: A trajectory, as timestamps to poses
        :return: Integer trajectory id which is the index to the trajectory map
        """
        for idx, mapped_trajectory in enumerate(self._trajectory_map):
            if are_matching_trajectories(trajectory, mapped_trajectory):
                return idx
        self._trajectory_map.append(trajectory)
        return len(self._trajectory_map) - 1

    def _make_metadata_key(self, dataset):
        """
        Make a vector of the properties we care about for associating results from a dataset
        :param dataset: The image collection
        :return:
        """
        trajectory = extract_trajectory(dataset)
        trajectory_id = self._get_trajectory_id(trajectory)
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
        serialized['orbslam_map'] = self._orbslam_map
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        serialized['datasets'] = list(self._datasets)
        serialized['looped_datasets'] = list(self._looped_datasets)
        serialized['trajectory_map'] = [[(stamp, pose.serialize()) for stamp, pose in traj]
                                        for traj in self._trajectory_map]
        serialized['trial_list'] = self._trial_list
        serialized['result_list'] = self._result_list
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_map' in serialized_representation:
            kwargs['orbslam_map'] = serialized_representation['orbslam_map']
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        if 'datasets' in serialized_representation:
            kwargs['datasets'] = serialized_representation['datasets']
        if 'looped_datasets' in serialized_representation:
            kwargs['looped_datasets'] = serialized_representation['looped_datasets']
        if 'trajectory_map' in serialized_representation:
            kwargs['trajectory_map'] = serialized_representation['trajectory_map']
        if 'trial_list' in serialized_representation:
            kwargs['trial_list'] = serialized_representation['trial_list']
        if 'result_list' in serialized_representation:
            kwargs['result_list'] = serialized_representation['result_list']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def extract_trajectory(image_source):
    """
    Extract a trajectory from an image source, which will almost always be an image collection.
    :param image_source: The image source to extract from
    :return: The ground-truth camera trajectory, as a dict of timestamp to pose
    """
    trajectory = {}
    image_source.begin()
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
