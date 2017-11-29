# Copyright (c) 2017, John Skinner
import os
import util.dict_utils as du


dataset_names = [
    'rgbd_dataset_freiburg1_xyz',
    'rgbd_dataset_freiburg1_rpy',
    'rgbd_dataset_freiburg2_xyz',
    'rgbd_dataset_freiburg2_rpy',
    'rgbd_dataset_freiburg1_360',
    'rgbd_dataset_freiburg1_floor',
    'rgbd_dataset_freiburg1_desk',
    'rgbd_dataset_freiburg1_desk2',
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg2_360_hemisphere',
    'rgbd_dataset_freiburg2_360_kidnap',
    'rgbd_dataset_freiburg2_desk',
    'rgbd_dataset_freiburg2_large_no_loop',
    'rgbd_dataset_freiburg2_large_with_loop',
    'rgbd_dataset_freiburg3_long_office_household',
    'rgbd_dataset_freiburg2_pioneer_360',
    'rgbd_dataset_freiburg2_pioneer_slam',
    'rgbd_dataset_freiburg2_pioneer_slam2',
    'rgbd_dataset_freiburg2_pioneer_slam3',
    'rgbd_dataset_freiburg3_nostructure_notexture_far',
    'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
    'rgbd_dataset_freiburg3_nostructure_texture_far',
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
    'rgbd_dataset_freiburg3_structure_notexture_far',
    'rgbd_dataset_freiburg3_structure_notexture_near',
    'rgbd_dataset_freiburg3_structure_texture_far',
    'rgbd_dataset_freiburg3_structure_texture_near',
    'rgbd_dataset_freiburg2_desk_with_person',
    'rgbd_dataset_freiburg3_sitting_static',
    'rgbd_dataset_freiburg3_sitting_xyz',
    'rgbd_dataset_freiburg3_sitting_halfsphere',
    'rgbd_dataset_freiburg3_sitting_rpy',
    'rgbd_dataset_freiburg3_walking_static',
    'rgbd_dataset_freiburg3_walking_xyz',
    'rgbd_dataset_freiburg3_walking_halfsphere',
    'rgbd_dataset_freiburg3_walking_rpy',
    'rgbd_dataset_freiburg1_plant',
    'rgbd_dataset_freiburg1_teddy',
    'rgbd_dataset_freiburg2_coke',
    'rgbd_dataset_freiburg2_dishes',
    'rgbd_dataset_freiburg2_flowerbouquet',
    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
    'rgbd_dataset_freiburg2_metallic_sphere',
    'rgbd_dataset_freiburg2_metallic_sphere2',
    'rgbd_dataset_freiburg3_cabinet',
    'rgbd_dataset_freiburg3_large_cabinet',
    'rgbd_dataset_freiburg3_teddy'
]


class TUMManager:

    def __init__(self, config, dataset_ids=None):
        self._config = {name: False for name in dataset_names}
        self._dataset_ids = {}

        for key in self._config.keys():
            if key in config and bool(config[key]):
                self._config[key] = True
        if dataset_ids is not None:
            du.defaults(self._dataset_ids, dataset_ids)

    @property
    def dataset_ids(self):
        """
        Get a set of all the dataset ids
        :return:
        """
        return set(self._dataset_ids.values())

    @property
    def datasets(self):
        """
        A generator of name -> id pairs
        :return:
        """
        return self._dataset_ids.items()

    def do_imports(self, root_folder, task_manager):
        to_import = {dataset_name for dataset_name, do_import in self._config.items()
                     if bool(do_import) and (dataset_name not in self._dataset_ids or
                                             self._dataset_ids[dataset_name] is None)}

        # Recursively search for the directories to import from the root folder
        full_paths = set()
        for dirpath, subdirs, _ in os.walk(root_folder):
            for subdir in subdirs:
                if subdir in to_import:
                    full_paths.add((subdir, os.path.join(dirpath, subdir)))

        # Create tasks for tall the paths we found
        for dataset_folder, full_path in full_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name='dataset.tum.tum_loader',
                path=full_path,
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='8:00:00'
            )
            if import_dataset_task.is_finished:
                self._dataset_ids[dataset_folder] = import_dataset_task.result
            else:
                task_manager.do_task(import_dataset_task)

    def serialize(self):
        return {
            'config': self._config,
            'dataset_ids': self._dataset_ids
        }

    @classmethod
    def deserialize(cls, serialized, **kwargs):
        config = {}
        dataset_ids = {}
        if 'config' in serialized:
            du.defaults(config, serialized['config'])
        if 'dataset_ids' in serialized:
            du.defaults(dataset_ids, serialized['dataset_ids'])
        return cls(config, dataset_ids, **kwargs)


# Add read-only properties to the manager class for each of the datasets
# This means that specific datasets can be requested as tum_manager.rgbd_dataset_freiburg1_xyz
for _name in dataset_names:
    setattr(TUMManager, _name, property(lambda self: self._dataset_ids[_name] if _name in self._dataset_ids else None))
