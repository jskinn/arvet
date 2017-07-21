"""
A set of mass imports for entities, so that the classes are defined
when we deserialize entities.
Add new entity classes here, only leaf classes matter though
"""
import core.image_entity
import core.image_collection

import training.epoch_trainer

import systems.deep_learning.keras_frcnn_trainee

import systems.deep_learning.keras_frcnn

import benchmarks.bounding_box_overlap.bounding_box_overlap
import benchmarks.bounding_box_overlap.bounding_box_overlap_result

import trials.object_detection.bounding_box_result

import experiments.podcup.podcup_experiment
