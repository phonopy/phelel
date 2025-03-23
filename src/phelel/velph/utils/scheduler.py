"""Utilities to handle scheduler scripts."""

from __future__ import annotations

import copy
from typing import Optional, Union


def get_sge_scheduler_script(
    toml_scheduler_dict: dict,
    job_id: Optional[Union[str, int]] = None,
) -> str:
    """Return scheduler script of SGE.

    This is called when scheduler_name = "sge".

    Supported parameters
    --------------------
    Necessary tags:
        job_name
        mpirun_command
        vasp_binary
        prepend_text
        append_text
    Optional tags:
        pe
        walltime : [hours:minutes:]seconds, e.g., 12:00:00

    """
    header = """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N {job_name}
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
"""
    params = []
    if "pe" in toml_scheduler_dict:
        params.append("#$ -pe {pe}")
    if "walltime" in toml_scheduler_dict:
        params.append("#$ -l h_rt={walltime}")
    if params:
        params_header = "\n".join(params) + "\n"
    else:
        params_header = ""

    commands = """
{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
"""
    scheduler_template = header + params_header + commands

    return get_custom_schedular_script(scheduler_template, toml_scheduler_dict, job_id)


def get_slurm_scheduler_script(
    toml_scheduler_dict: dict,
    job_id: Optional[Union[str, int]] = None,
) -> str:
    """Return scheduler script of SLURM.

    This is called when scheduler_name = "slurm".

    Supported parameters
    --------------------
    Necessary tags:
        job_name
        mpirun_command
        vasp_binary
        prepend_text
        append_text
        partition
        nodes
        ntasks

    """
    scheduler_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time=96:00:00      # Time limit hrs:min:sec
#SBATCH --output=ci_%j.log   # Standard output and error log

{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
"""

    return get_custom_schedular_script(scheduler_template, toml_scheduler_dict, job_id)


def get_custom_schedular_script(
    template: str, toml_scheduler_dict: dict, job_id: Optional[Union[str, int]]
) -> str:
    """Return scheduler script with given template.

    This is called when scheduler_name = "custom".

    """
    sched_dict = copy.deepcopy(toml_scheduler_dict)
    if "job_name" in sched_dict and job_id is not None:
        sched_dict["job_name"] += f"-{job_id}"
    return template.format(**sched_dict)
