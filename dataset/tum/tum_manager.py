import os.path
import glob
import util.dict_utils as du


class TUMManager:

    def __init__(self, config, dataset_ids=None):
        self._config = {
            'rgbd_dataset_freiburg1_xyz': False,
            'rgbd_dataset_freiburg1_rpy': False,
            'rgbd_dataset_freiburg2_xyz': False,
            'rgbd_dataset_freiburg2_rpy': False
        }
        self._dataset_ids = {}

        for key in self._config.keys():
            if key in config and bool(config[key]):
                self._config[key] = True
        if dataset_ids is not None:
            du.defaults(self._dataset_ids, dataset_ids)

    @property
    def all_datasets(self):
        return set(self._dataset_ids.values())

    @property
    def rgbd_dataset_freiburg1_xyz(self):
        if 'rgbd_dataset_freiburg1_xyz' in self._dataset_ids:
            return self._dataset_ids['rgbd_dataset_freiburg1_xyz']
        return None

    @property
    def rgbd_dataset_freiburg1_rpy(self):
        if 'rgbd_dataset_freiburg1_rpy' in self._dataset_ids:
            return self._dataset_ids['rgbd_dataset_freiburg1_rpy']
        return None

    @property
    def rgbd_dataset_freiburg2_xyz(self):
        if 'rgbd_dataset_freiburg1_xyz' in self._dataset_ids:
            return self._dataset_ids['rgbd_dataset_freiburg2_xyz']
        return None

    @property
    def rgbd_dataset_freiburg2_rpy(self):
        if 'rgbd_dataset_freiburg2_rpy' in self._dataset_ids:
            return self._dataset_ids['rgbd_dataset_freiburg2_rpy']
        return None

    def do_imports(self, root_folder, task_manager):
        to_import = {dataset_name for dataset_name, do_import in self._config.items()
                     if bool(do_import) and (dataset_name not in self._dataset_ids or
                                             self._dataset_ids[dataset_name] is None)}
        for dataset_folder in to_import:
            try:
                full_path = next(glob.iglob(os.path.join(root_folder, '**', dataset_folder)))
            except StopIteration:
                full_path = None
            if full_path is not None:
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
