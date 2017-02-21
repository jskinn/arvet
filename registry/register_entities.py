from database.entity import Entity

from core.dataset import Dataset
from core.image import Image

from core.trial_result import TrialResult
from core.trained_system import TrainedState
from core.visual_slam import SLAMTrialResult
from core.loop_closure_detection import LoopClosureTrialResult
from libviso2.libviso2 import LibVisOTrialResult

from core.benchmark import BenchmarkResult, FailedBenchmark
from benchmarks.absolute_trajectory_error import BenchmarkATEResult
from benchmarks.mean_time_lost import BenchmarkMTLResult
from benchmarks.mean_distance_lost import BenchmarkMDLResult
from benchmarks.frame_delta_error import BenchmarkFrameDeltaResult
from benchmarks.frame_delta_error_filtered import BenchmarkFrameDeltaFilteredResult
from benchmarks.precision_recall import BenchmarkPrecisionRecallResult
from benchmarks.tracking_comparison_benchmark import BenchmarkTrackingComparisonResult
from benchmarks.absolute_trajectory_error_comparison import BenchmarkATEComparisonResult
from benchmarks.pose_error_comparison import BenchmarkPoseErrorComparisonResult
from benchmarks.pose_error_comparison_filtered import BenchmarkPoseErrorComparisonResultFiltered
from benchmarks.match_comparison import BenchmarkMatchingComparisonResult

from openfabmap.openfabmap import OpenFABMAPTrainedState


def register_entities(db_client):
    """
    Register all the different entity types to the DB client for deserialization
    I can't think of a nicer solution than having a big list here,
    I'm not sold on having each child of Entity register itself with a singleton
    when it is declared. It feels nasty to have modules imported and never used, explicitly
    :param db_client:
    :return:
    """
    db_client.register_entity(Entity)
    db_client.register_entity(Dataset)
    db_client.register_entity(Image)
    db_client.register_entity(TrialResult)
    db_client.register_entity(SLAMTrialResult)
    db_client.register_entity(LoopClosureTrialResult)
    db_client.register_entity(LibVisOTrialResult)
    db_client.register_entity(BenchmarkResult)
    db_client.register_entity(FailedBenchmark)
    db_client.register_entity(BenchmarkATEResult)
    db_client.register_entity(BenchmarkMTLResult)
    db_client.register_entity(BenchmarkMDLResult)
    db_client.register_entity(BenchmarkFrameDeltaResult)
    db_client.register_entity(BenchmarkFrameDeltaFilteredResult)
    db_client.register_entity(BenchmarkPrecisionRecallResult)
    db_client.register_entity(BenchmarkTrackingComparisonResult)
    db_client.register_entity(BenchmarkATEComparisonResult)
    db_client.register_entity(BenchmarkPoseErrorComparisonResult)
    db_client.register_entity(BenchmarkPoseErrorComparisonResultFiltered)
    db_client.register_entity(BenchmarkMatchingComparisonResult)
    db_client.register_entity(TrainedState)
    db_client.register_entity(OpenFABMAPTrainedState)