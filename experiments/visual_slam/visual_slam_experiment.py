import matplotlib.pyplot as pyplot
import util.database_helpers as dh
import util.associate
import batch_analysis.experiment
import systems.visual_odometry.libviso2.libviso2 as libviso2
import benchmarks.rpe.relative_pose_error as rpe
import benchmarks.trajectory_drift.trajectory_drift as traj_drift


class VisualSlamExperiment(batch_analysis.experiment.Experiment):

    def __init__(self, *args, dataset_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_map = dataset_map if dataset_map is not None else {}

    def import_trainers(self, db_client):
        """
        Import trainers for this experiment.
        Trainers are city simulators of different kinds & qualities
        :param db_client:
        :return:
        """
        return set()

    def import_trainees(self, db_client):
        """
        Import trainees for this experiment.
        This will be the learned VO system from feras
        :param db_client:
        :return:
        """
        return set()

    def import_image_sources(self, db_client, job_system):
        """
        Import image sources for evaluation in this experiment
        :param db_client:
        :param job_system: The job system
        :return:
        """
        for path in {'/home/john/Renders/Dataset 1/metadata.json'}:
            map_key = path.replace('.', '-')
            if map_key not in self._dataset_map:
                if job_system.queue_import_dataset('dataset.generated.import_generated_dataset', path, self.identifier):
                    self._dataset_map[map_key] = False
                    if '$set' not in self._updates:
                        self._updates['$set'] = {}
                    self._updates['$set']['dataset_map.{0}'.format(map_key)] = False
        return set()

    def import_systems(self, db_client):
        """
        Import the systems under test for this experiment.
        They are: libviso2
        The trained systems we want to use will be added automatically when they are trained.
        TODO: Add ORBSLAM2 as well.
        :param db_client:
        :return:
        """
        systems = set()
        systems.add(dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem(
            # TODO: Get accurate camera focal information.
            # Alternately, get this ground truth from the image source when it is run.
            focal_distance=1,
            cu=640,
            cv=360,
            base=30
        )))
        return systems

    def import_benchmarks(self, db_client):
        """
        Create and store the benchmarks for bounding boxes.
        Just using the default settings for now
        :param db_client:
        :return: The set of benchmark ids.
        """
        c = db_client.benchmarks_collection
        benchmarks = set()
        benchmarks.add(dh.add_unique(c, rpe.BenchmarkRPE(
            max_pairs=10000,
            fixed_delta=False,
            delta=1.0,
            delta_unit='s',
            offset=0,
            scale_=1)))
        benchmarks.add(dh.add_unique(c, traj_drift.BenchmarkTrajectoryDrift(
            segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
            step_size=10
        )))
        return benchmarks

    def add_image_source(self, image_source_id, folder, db_client=None):
        # TODO: Grab out any dataset we only want to use for training
        map_key = folder.replace('.', '-')
        self._dataset_map[map_key] = image_source_id
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        self._updates['$set']['dataset_map.{0}'.format(map_key)] = image_source_id
        super().add_image_source(image_source_id, folder, db_client)

    def plot_results(self, db_client):
        """
        Plot the results for this experiment.
        I don't know how we want to do this yet.
        :param db_client:
        :return:
        """
        for trial_result_id in self._trial_results:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
            result_traj = trial_result.get_computed_camera_poses()
            matches = util.associate.associate(ground_truth_traj, result_traj, offset=0, max_difference=1)
            x = []
            y = []
            x_gt = []
            y_gt = []
            gt_start = ground_truth_traj[min(ground_truth_traj.keys())]
            for gt_stamp, result_stamp in matches:
                gt_relative_pose = gt_start.find_relative(ground_truth_traj[gt_stamp])
                x_gt.append(gt_relative_pose.location[0])
                y_gt.append(gt_relative_pose.location[1])
                x.append(result_traj[result_stamp].location[0])
                y.append(result_traj[result_stamp].location[1])

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            ax = figure.add_subplot(111)
            ax.plot(x, y)
            ax.plot(x_gt, y_gt)
            ax.set_xlabel('x-location')
            ax.set_ylabel('y-location')
            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)
        pyplot.show()


    def serialize(self):
        serialized = super().serialize()
        serialized['dataset_map'] = {folder: dataset_id for folder, dataset_id in self._dataset_map.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'dataset_map' in serialized_representation:
            kwargs['dataset_map'] = serialized_representation['dataset_map']
        return super().deserialize(serialized_representation, db_client, **kwargs)
