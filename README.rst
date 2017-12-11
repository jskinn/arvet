=======================================
ARVET: A Robotic Vision Evaluation Tool
=======================================

A framework for testing and training robot vision systems, as part of a robot vision experiment.
Handles logic around importing datasets, intializing and calling different robot vision systems,
feeding them images, collecting the output, and then evaluating the system output.
This is a work in progress, so it will be hard to use without reading the code.

Handles monocular, stereo and RGB-D systems, only running the systems with datasets that have the appropriate data.

Implements a task management system, to manage which tasks have been completed,
and to interface with external job managers and batch systems (such as exist on High-Performance computing servers).
Experiment data and state is stored in a MongoDB database.
Several separate nodes can run the code from the same MongoDB database, to distribute computation.

This module defines core types and automates the evaluation process, but actually conducting experiments requires
explicit creation of systems, datasets, and benchmarks under test;
as well as intermediate types for storing their output.
This is made easier through extension modules such as `arvet_slam`_

.. _arvet_slam: https://github.com/jskinn/arvet-slam

Usage
=====

Creating an experiment
----------------------

To define an experiment, override `arvet.batch_analysis.experiment.Experiment`,
in particular the `do_imports`, `schedule_tasks`, and `plot_results` methods.

- `do_imports` method should import and store references to whatever image datasets
- `schedule_tasks` indicates which systems should be run with which image datasets, and how each result should be assessed
- `plot results` visualizes the performance output

Lastly, create and store an instance of the experiment in `add_initial_entities.py`.

Running experiments
-------------------

Run `add_initial_entities.py` to create the experiments.
Then, run `scheduler.py` repeatedly to incrementally schedule systems to be run with image sources


Configuration
=============

Configuration information is stored in config.yml,
allowing you to specify how to connect and structure data in the MongoDB database,
which types of tasks to run on this node, what kind of job system should be used to run
the jobs (see `arvet.batch_analysis.job_systems`), and the log output.

Structure
=========

- The key abstractions are in the core module, with implementations of the base types in the `benchmarks`, `dataset`, `simulation`, `systems`, and `trials` modules.
- the `arvet.batch_analysis` module contains code to perform analysis based on these abstractions, including defining experiments, the task management system, and the different job systems.
- The `database` module contains util code for connecting to the MongoDB database, and saving and loading objects from it.
- The `metadata` module contains management structures for image metadata, including camera intrinsics, object labels, camera pose, and many other properties. See `metadata.image_metadata` for the full implementation.

Data and state is stored in a MongoDB database, which can be configured in config.yml.

License
=======

Except where otherwise noted in the relevant file, this code is licensed under the BSD 2-Clause licence, see LICENSE.

Python Dependencies
===================

The code is in python 3, tested with both 3.4 and 3.5. Does not support python 2.
The core structure depends on the following python libraries, which will be installed automatically by pip:

- pymongo
- numpy
- transforms3d
- mongomock
- xxhash
- PyYAML
- pillow
- unrealcv

Additionally, some of the possible image augmentations (`arvet.image_collections.image_augmentations.opencv_augmentations`),
and certain visualizations (`arvet.visualize_dataset` and `arvet.verify_bounding_boxes_manually` require OpenCV.
They should not occur as part of normal operation, but will produce ImportErrors if OpenCV is unavailable.
