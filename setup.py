#!/usr/bin/env python

from distutils.core import setup

setup(
    name='argus',
    version='0.1.0',
    description='Framework and utilities for performing robotic vision experiments',
    author='John Skinner',
    author_email='jskinn@protonmail.com',
    url='https://gitub.com/jskinn/argus',
    packages=['argus.batch_analysis', 'argus.batch_analysis.job_systems', 'argus.batch_analysis.tasks', 'argus.config',
              'argus.core', 'argus.database',
              'argus.image_collections', 'argus.image_collections.image_augmentations', 'argus.metadata', 'argus.simulation',
              'argus.simulation.controllers', 'argus.simulation.unrealcv', 'argus.training', 'argus.trials', 'argus.trials.feature_detection',
              'argus.trials.loop_closure_detection', 'argus.trials.object_detection', 'argus.trials.slam', 'argus.trials.visual_odometry',
              'argus.util'],
    requires=['pymongo', 'numpy', 'transforms3d', 'mongomock', 'xxhash', 'PyYAML']
)
