"""Tests of the directories and files of a velph project."""

from __future__ import annotations

import ast
import pathlib
import re
from typing import get_args

import pytest

from phelel.velph.config.schema import CODES, InputSetName
from phelel.velph.jobs import project_directory
from phelel.velph.jobs.project_directory import ProjectDirectory

# src/phelel/velph, the directory of the modules of velph.
_VELPH_DIR = pathlib.Path(project_directory.__file__).parents[1]


@pytest.mark.parametrize(
    "n,names",
    [
        (2, ["disp-000", "disp-001", "disp-002"]),
        (1000, ["disp-0000", "disp-0001", "disp-1000"]),
    ],
)
def test_disp_directories(n: int, names: list[str]):
    """The numbers have three digits, or as many as n needs.

    The perfect supercell is disp-0..., and the n supercells follow it.

    """
    directories = ProjectDirectory(pathlib.Path("project")).disp_directories(
        "phonopy", n
    )
    assert len(directories) == n + 1
    assert [directories[i].name for i in (0, 1, n)] == names
    assert directories[0].parent == pathlib.Path("project") / "phonopy"


def test_files_of_project():
    """The files are in the directories of the input sets and the codes, under root.

    disp_id gives the number of a directory of disp_directories.

    """
    project_dir = ProjectDirectory(pathlib.Path("project"))
    assert project_dir.input_set_directory("el_bands") == pathlib.Path(
        "project/el_bands"
    )
    assert project_dir.vaspout("transport") == pathlib.Path(
        "project/transport/vaspout.h5"
    )
    assert project_dir.plot_pdf("ph_bands") == pathlib.Path(
        "project/ph_bands/ph_bands.pdf"
    )
    assert project_dir.gamma_elph_hdf5 == pathlib.Path(
        "project/ph_selfenergy/gamma_elph{mesh}.hdf5"
    )
    assert project_dir.relax_directory(2) == pathlib.Path("project/relax/iter2")
    assert project_dir.disp_yaml("phono3py") == pathlib.Path(
        "project/phono3py/phono3py_disp.yaml"
    )
    assert project_dir.phelel_params == pathlib.Path(
        "project/phelel/phelel_params.hdf5"
    )
    assert project_dir.phonopy_params == pathlib.Path(
        "project/phelel/phonopy_params.yaml"
    )
    assert project_dir.nac_directory == pathlib.Path("project/nac")
    assert ProjectDirectory().disp_yaml("phelel") == pathlib.Path(
        "phelel/phelel_disp.yaml"
    )
    directories = project_dir.disp_directories("phelel", 12)
    assert [ProjectDirectory.disp_id(d) for d in directories[:2]] == ["000", "001"]
    assert ProjectDirectory.disp_id(directories[-1]) == "012"


# The module that writes the places of the files of a project, which the check
# of the other modules skips.
_PROJECT_DIRECTORY_MODULE = (
    pathlib.Path(project_directory.__file__).relative_to(_VELPH_DIR).as_posix()
)


# Rules of the names that the methods of ProjectDirectory with arguments make,
# such as disp_yaml("phelel"), which _place_pattern cannot take from
# ProjectDirectory.  A new method of this kind adds its rule here.
_RULES_OF_METHODS = (r"_disp\.yaml", r"\bdisp-", r"relax/iter")


def _place_pattern() -> re.Pattern[str]:
    """Return the pattern of a string that writes a place of a file of a project.

    It matches the name of an input set or of a code followed by "/", such as
    "el_dos/vaspout.h5", the name of a file given by a property of
    ProjectDirectory, such as "phelel_params.hdf5", and _RULES_OF_METHODS.
    The names of the files are taken from the properties, so a new property
    is found without a change here.  A property of a directory, such as
    nac_directory, is found by the first part, because its name is that of an
    input set.

    """
    names = sorted(set(get_args(InputSetName)) | set(CODES), key=len, reverse=True)
    file_names = [
        getattr(ProjectDirectory(), attr).name
        for attr, value in vars(ProjectDirectory).items()
        if isinstance(value, property) and getattr(ProjectDirectory(), attr).suffix
    ]
    return re.compile(
        "|".join(
            [r"\b(" + "|".join(names) + r")/"]
            + [re.escape(name) for name in file_names]
            + list(_RULES_OF_METHODS)
        )
    )


# Nodes that can have a docstring as the first statement of their body.
_DOCUMENTED_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_of(node: ast.AST) -> ast.Constant | None:
    """Return the string constant of the docstring of node.

    A docstring is a string written as the first statement of a module, a
    class or a function.  None is returned when node has no docstring.

    """
    if not isinstance(node, _DOCUMENTED_NODES) or not node.body:
        return None
    first = node.body[0]
    if not isinstance(first, ast.Expr):
        return None
    if not isinstance(first.value, ast.Constant):
        return None
    if not isinstance(first.value.value, str):
        return None
    return first.value


def _places_written_outside_project_directory() -> list[str]:
    """Return the string constants that write a place of a file of a project.

    These are the strings, other than docstrings, of the modules of velph
    except project_directory.py, that _place_pattern matches.  Each is returned
    as "<module>:<line>: <string>".  A place written in parts, such as
    pathlib.Path("relax") / "iter1", is not found.

    """
    pattern = _place_pattern()
    found = []
    for filename in sorted(_VELPH_DIR.rglob("*.py")):
        module = filename.relative_to(_VELPH_DIR).as_posix()
        if module == _PROJECT_DIRECTORY_MODULE:
            continue
        tree = ast.parse(filename.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            docstring = _docstring_of(node)
            if docstring is not None:
                docstrings.add(docstring)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node in docstrings:
                continue
            if pattern.search(node.value):
                found.append(f"{module}:{node.lineno}: {node.value!r}")
    return found


@pytest.mark.parametrize(
    "text,found",
    [
        ("el_dos/vaspout.h5", True),
        ("Take the phonon supercells from phonopy/phonopy_disp.yaml.", True),
        ("phelel_params.hdf5", True),
        ("phonopy_params.yaml", True),
        ("gamma_elph{mesh}.hdf5", True),
        ("disp-001", True),
        ("relax/iter2", True),
        ("nac/vasprun.xml", True),
        ("nac", False),
        ("vaspout.h5", False),
        ("phelel.velph.jobs", False),
    ],
)
def test_place_pattern(text: str, found: bool):
    """The pattern finds the places of ProjectDirectory and not the bare names.

    "nac" is the name of an input set, and vaspout.h5 is the name of a file
    that VASP writes in any directory.

    """
    assert bool(_place_pattern().search(text)) is found


def test_places_are_given_by_project_directory():
    """The places of the files of a project are written only in ProjectDirectory.

    A new place, such as a file that a new command writes, is added to
    ProjectDirectory, and the other modules get it from there.  The strings
    found are listed with their modules and lines.

    """
    found = _places_written_outside_project_directory()
    assert not found, (
        "Get these places of the files of a project from ProjectDirectory of "
        f"{_PROJECT_DIRECTORY_MODULE}:\n" + "\n".join(found)
    )
