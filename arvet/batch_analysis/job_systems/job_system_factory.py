# Copyright (c) 2017, John Skinner
import arvet.batch_analysis.job_systems.hpc_job_system
import arvet.batch_analysis.job_systems.simple_job_system


def create_job_system(config):
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
    :return:
    """
    job_system_config = config['job_system_config'] if 'job_system_config' in config else {}
    job_system_type = job_system_config['job_system'] if 'job_system' in job_system_config else 'simple'
    job_system_type = job_system_type.lower()
    if job_system_type == 'hpc':
        return arvet.batch_analysis.job_systems.hpc_job_system.HPCJobSystem(job_system_config)
    return arvet.batch_analysis.job_systems.simple_job_system.SimpleJobSystem(job_system_config)
