"""Test scheduler writer."""

import pytest

from phelel.velph.utils.scheduler import (
    get_sge_scheduler_script,
    get_slurm_scheduler_script,
)

toml_scheduler_dict = {
    "job_name": "test_job",
    "mpirun_command": "mpirun",
    "vasp_binary": "vasp",
    "prepend_text": "load test",
    "append_text": "sleep 5",
}


@pytest.mark.parametrize(
    "i,scheduler_dict",
    [
        (0, {}),
        (1, {"pe": "mpi* 24"}),
        (2, {"walltime": "12:00:00"}),
        (3, {"pe": "mpi* 24", "walltime": "12:00:00"}),
    ],
)
def test_get_sge_scheduler_script(i, scheduler_dict):
    """Test get_sge_scheduler_script."""
    scheduler_dict.update(toml_scheduler_dict)
    script_ref = [
        """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N test_job
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt

load test

mpirun vasp

sleep 5
""",
        """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N test_job
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe mpi* 24

load test

mpirun vasp

sleep 5
""",
        """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N test_job
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -l h_rt=12:00:00

load test

mpirun vasp

sleep 5
""",
        """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N test_job
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe mpi* 24
#$ -l h_rt=12:00:00

load test

mpirun vasp

sleep 5
""",
    ][i]

    assert get_sge_scheduler_script(scheduler_dict) == script_ref


def test_get_slurm_scheduler_script():
    """Test get_slurm_scheduler_script."""
    script_ref = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --partition=my_partition
#SBATCH --nodes=14
#SBATCH --ntasks=20
#SBATCH --time=96:00:00      # Time limit hrs:min:sec
#SBATCH --output=ci_%j.log   # Standard output and error log

load test

mpirun vasp

sleep 5
"""
    scheduler_dict = {"partition": "my_partition", "nodes": "14", "ntasks": "20"}
    scheduler_dict.update(toml_scheduler_dict)
    assert get_slurm_scheduler_script(scheduler_dict) == script_ref
