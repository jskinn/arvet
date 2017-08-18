import os
import glob
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D     # Necessary for 3D plots
import util.database_helpers as dh
import util.associate
import batch_analysis.experiment
import systems.visual_odometry.libviso2.libviso2 as libviso2
import systems.slam.orbslam2 as orbslam2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.trajectory_drift.trajectory_drift as traj_drift


class VisualSlamExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None, orbslam_monocular_system=None, orbslam_stereo_system=None,
                 orbslam_rgbd_system=None, benchmark_rpe=None, benchmark_trajectory_drift=None, id_=None):
        super().__init__(id_=id_)
        self._libviso_system = libviso_system
        self._orbslam_monocular = orbslam_monocular_system
        self._orbslam_stereo = orbslam_stereo_system
        self._orbslam_rgbd = orbslam_rgbd_system
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_trajectory_drift = benchmark_trajectory_drift

    def do_imports(self, task_manager, db_client):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
        # Import existing datasets
        datasets = set()
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
                # TODO: Map out the datasets by world, trajectory, and quality change. Needs data structure
                datasets.add(import_dataset_task.result)
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
        if self._orbslam_monocular is None:
            self._orbslam_monocular = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
                settings={}, mode=orbslam2.SensorMode.MONOCULAR, resolution=(1280, 720)
            ))
            self._set_property('orbslam_monocular', self._orbslam_monocular)
        if self._orbslam_stereo is None:
            self._orbslam_stereo = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
                settings={}, mode=orbslam2.SensorMode.STEREO, resolution=(1280, 720)
            ))
            self._set_property('orbslam_stereo', self._orbslam_stereo)
        if self._orbslam_rgbd is None:
            self._orbslam_rgbd = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                vocabulary_file='/opt/ORBSLAM2/Vocabulary/ORBvoc.txt',
                settings={}, mode=orbslam2.SensorMode.RGBD, resolution=(1280, 720)
            ))
            self._set_property('orbslam_stereo', self._orbslam_rgbd)

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
        # TODO: Flatten the map of image sources to a list we can loop over
        test_image_sources = []
        systems = [
            dh.load_object(db_client, db_client.system_collection, self._libviso_system),
            dh.load_object(db_client, db_client.system_collection, self._orbslam_monocular),
            dh.load_object(db_client, db_client.system_collection, self._orbslam_stereo),
            dh.load_object(db_client, db_client.system_collection, self._orbslam_rgbd)
        ]
        benchmarks = [
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_rpe),
            dh.load_object(db_client, db_client.benchmarks_collection, self._benchmark_trajectory_drift)
        ]

        # Schedule trials
        trial_results = []
        for system in systems:
            for image_source in test_image_sources:
                if system.is_image_source_appropriate(image_source):
                    task = task_manager.get_run_system_task(
                        system_id=system.identifier,
                        image_source_id=image_source.identifier,
                        expected_duration='4:00:00'
                    )
                    if task.is_finished:
                        # TODO: Store the trial results in a similar map to the datasets
                        trial_results.append(dh.load_object(db_client, db_client.trials_collection, task.result))
                    else:
                        task_manager.do_task(task)

        # Benchmark results
        # TODO: We need a similar structure for benchmark results as trial results
        results = []
        for trial_result in trial_results:
            for benchmark in benchmarks:
                if benchmark.is_trial_appropriate(trial_result):
                    task = task_manager.get_benchmark_task(
                        trial_result_id=trial_result.identifier,
                        benchmark_id=benchmark.identifier,
                        expected_duration='4:00:00'
                    )
                    if task.is_finished:
                        results.append(task.result)
                    else:
                        task_manager.do_task(task)

    def plot_results(self, db_client):
        """
        Plot the results for this experiment.
        I don't know how we want to do this yet.
        :param db_client:
        :return:
        """
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

    def _get_representative_metadata(self, image_source_id, db_client):
        # TODO: To sort the image sources, we need a way of getting aggregate metadata
        # This function should somehow retrieve that given a dataset id.
        images_list = db_client.image_sources_collection.find_one({'_id': image_source_id}, {
            'images': True
        })
        images_list = sorted(images_list)

    def serialize(self):
        serialized = super().serialize()
        serialized['libviso'] = self._libviso_system
        serialized['orbslam_monocular'] = self._orbslam_monocular
        serialized['orbslam_stereo'] = self._orbslam_stereo
        serialized['orbslam_rgbd'] = self._orbslam_rgbd
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_monocular' in serialized_representation:
            kwargs['orbslam_monocular_system'] = serialized_representation['orbslam_monocular']
        if 'orbslam_stereo' in serialized_representation:
            kwargs['orbslam_stereo_system'] = serialized_representation['orbslam_stereo']
        if 'orbslam_rgbd' in serialized_representation:
            kwargs['orbslam_rgbd_system'] = serialized_representation['orbslam_rgbd']
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        return super().deserialize(serialized_representation, db_client, **kwargs)
