# Copyright (c) 2017, John Skinner
from arvet.batch_analysis.job_systems.hpc_job_system import HPCJobSystem
from arvet.batch_analysis.job_systems.simple_job_system import SimpleJobSystem


def create_job_system(config, config_file):
    """
    A factory class that reads global configuration
    and constructs the job system for this platform.
    Takes configuration parameters in a dict with the following format:
    The inner 'job_system_config' will be passed to the constructor
    {
        'job_system_config': {
            'job_system': ('hpc'|'simple') # Case insensitive
            ...additional parameters...
        }
    }
    :param config:
    :param config_file: The path of the config file used, to pass to the tasks.
    :return:
    """
    # Read the configuration for the job system to use
    job_system_config = config.get('job_system_config', {})
    job_system_type = job_system_config.get('job_system', 'simple')
    job_system_type = job_system_type.lower()

    # Make the appropriate job system
    if job_system_type == 'hpc':
        return HPCJobSystem(job_system_config, config_file=config_file)
    return SimpleJobSystem(job_system_config, config_file=config_file)
