import shutil

import pytest

from evoagent_no.task import TEMPLATES, load_task


def copy_template(tmp_path, template: str):
    """A fresh copy of a template, as `evoagent-no new <template>` makes it."""
    target = tmp_path / template
    shutil.copytree(TEMPLATES / template, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def tiny(task):
    """Shrink the student for fast CPU tests; agents off unless a test turns them on with a fake LLM."""
    task.settings["student"].update(width=8, modes=4, layers=2)
    task.settings["agents"]["enabled"] = False
    return task


@pytest.fixture
def task_dir(tmp_path):
    return copy_template(tmp_path, "pde1d")


@pytest.fixture
def small_task(task_dir):
    return tiny(load_task(task_dir))


@pytest.fixture(params=["pde1d", "elliptic2d"])
def any_small_task(request, tmp_path):
    return tiny(load_task(copy_template(tmp_path, request.param)))
