import os
import batch_analysis.job_system


class HPCJobSystem(batch_analysis.job_system.JobSystem):

    template = """
    #!/bin/python3
    {0} {1}
    """


    def run_script(self, script_name, *args):
        cwd = os.getcwd()

        # Create a job file
        args = ' '.join(args)
        self.template.format(os.path.join(cwd, script_name), args)
