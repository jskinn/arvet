# Copyright (c) 2017, John Skinner
import unittest
import os
import time
import numpy as np
import pymodm.fields as fields
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.core.tests.mock_types as mock_types

import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image import Image
from arvet.core.image_source import ImageSource
from arvet.core.image_collection import ImageCollection
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.batch_analysis.task import Task
import arvet.batch_analysis.task_manager as task_manager


class TimerVisionSystem(VisionSystem):

    def __init__(self, *args, **kwargs):
        super(TimerVisionSystem, self).__init__(*args, **kwargs)
        self.start_time = None
        self.actual_times = []

    @property
    def is_deterministic(self) -> bool:
        return False

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        return True

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics) -> None:
        pass

    def start_trial(self, sequence_type: ImageSequenceType) -> None:
        self.actual_times = []
        self.start_time = time.time()

    def process_image(self, image: Image, timestamp: float) -> None:
        self.actual_times.append(
            (timestamp, time.time() - self.start_time)
        )
        # Make sure we read the pixels, to force  a load
        _ = image.pixels[0]

    def finish_trial(self) -> TrialResult:
        finish_time = time.time()
        result = TimerTrialResult(
            system=self,
            success=True,
            total_time=finish_time - self.start_time,
            timestamps=[timestamp for timestamp, _ in self.actual_times],
            actual_times=[actual_time for _, actual_time in self.actual_times]
        )
        self.start_time = None
        self.actual_times = []
        return result


class TimerTrialResult(TrialResult):
    total_time = fields.FloatField()
    timestamps = fields.ListField(fields.FloatField())
    actual_times = fields.ListField(fields.FloatField())


@unittest.skip("Not running performance tests")
class TestRunSystemTaskPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        pass

    def tearDown(self):
        # Remove all the data from the database after each test
        Task.objects.all().delete()
        TrialResult.objects.all().delete()
        VisionSystem.objects.all().delete()
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_runs_real_time_10_fps_320x240(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 10, 320, 240, 600)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_10_fps_640x480(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 10, 640, 480, 600)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_30_fps_640x480(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 30, 640, 480, 1800)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_60_fps_640x480(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 60, 640, 480, 3600)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_10_fps_1920x1080(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 10, 1920, 1080, 600)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_30_fps_1920x1080(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 30, 1920, 1080, 1800)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)

    def test_runs_real_time_60_fps_1920x1080(self):
        path_manager = PathManager(['~'])
        task_id = make_task_for_test(1 / 60, 1920, 1080, 3600)
        task = Task.objects.get({'_id': task_id})
        task.mark_job_started('unittest', 0)
        task.run_task(path_manager)

        # Check that the task has finished
        self.assertTrue(task.is_finished)

        # Check the actual times don't overtake the supposed times
        result = task.result
        self.assertIsInstance(result, TimerTrialResult)
        for delta_timestamp, delta_actual in zip(
                delta_iter(iter(result.timestamps)),
                delta_iter(iter(result.actual_times)),
        ):
            self.assertLessEqual(delta_actual, delta_timestamp)


def make_task_for_test(timestep, width, height, length):
    """
    Make the prerequisites and the task for a test
    Done in a helper to send all the prerequistes out of scope before actually running the test.
    Should force everything to load from the db.
    :param timestep:
    :param width:
    :param height:
    :param length:
    :return:
    """
    # Make the system and the sequence
    sequence = make_image_sequence(timestep, width, height, length)
    system = TimerVisionSystem()
    system.save()

    # Make the task
    task = task_manager.get_run_system_task(
        system=system,
        image_source=sequence,
        repeat=0
    )
    task.mark_job_started('unittest', 0)
    task.save()
    return task.pk


def make_image_sequence(timestep, width, height, length):
    images = []
    for _ in range(length):
        pixels = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
        image = Image(
            pixels=pixels,
            metadata=imeta.make_metadata(
                pixels,
                source_type=imeta.ImageSourceType.SYNTHETIC,
                intrinsics=CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            ),
            additional_metadata={'test': True}
        )
        image.save()
        images.append(image)
    sequence = ImageCollection(
        images=images,
        timestamps=[idx * timestep for idx in range(length)],
        sequence_type=ImageSequenceType.SEQUENTIAL
    )
    sequence.save()
    return sequence


def delta_iter(it):
    """
    A tiny helper for finding the difference between successive times
    :param it:
    :return:
    """
    prev = next(it)
    for val in it:
        yield val - prev
        prev = val
