"""Changing a camera's Zoom must not move a Position AE left out of the file.

AE leaves a camera Position at its default, `[w/2, h/2, -zoom]`, out of the
file and places the camera from the Zoom on open. In
`camera_default_position.aep` the `zoom_1000` camera's Zoom is 1000 and its
Position, [720, 405, -1000], is left out. Setting the Zoom to 3000 in AE 2026
keeps the camera at -1000, reads it modified, and writes the Position; left
out, the camera jumped to -3000 on reopen (checked by opening py-aep's save
in AE 2026).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import get_comp, parse_project_fresh

from py_aep import parse as parse_aep
from py_aep.models.layers import CameraLayer
from py_aep.models.project import Project
from py_aep.models.properties.property import Property

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "layer"
SAMPLE = SAMPLES_DIR / "camera_default_position.aep"


def _camera(project: Project) -> CameraLayer:
    camera = get_comp(project, "zoom_1000").layers[0]
    assert isinstance(camera, CameraLayer)
    return camera


def _position(camera: CameraLayer) -> Property:
    prop = camera.transform["ADBE Position"]
    assert isinstance(prop, Property)
    return prop


def _zoom(camera: CameraLayer) -> Property:
    prop = camera["ADBE Camera Options Group"]["ADBE Camera Zoom"]
    assert isinstance(prop, Property)
    return prop


def _reopened_position(project: Project, tmp_path: Path) -> Property:
    out = tmp_path / "out.aep"
    project.save(out)
    return _position(_camera(parse_aep(out).project))


class TestZoomChange:
    def test_default_follows_zoom_in_memory(self) -> None:
        camera = _camera(parse_project_fresh(SAMPLE))
        position = _position(camera)
        assert not position.is_modified

        _zoom(camera).value = 3000

        assert position.value == pytest.approx([720.0, 405.0, -1000.0])
        assert position.default_value == pytest.approx([720.0, 405.0, -3000.0])
        assert position.is_modified
        z_follower = camera.transform["ADBE Position_2"]
        assert isinstance(z_follower, Property)
        assert z_follower.default_value == pytest.approx(-3000.0)

    def test_default_is_float_after_int_zoom(self) -> None:
        camera = _camera(parse_project_fresh(SAMPLE))
        _zoom(camera).value = 3000

        default = _position(camera).default_value
        assert all(isinstance(v, float) for v in default)

    def test_static_zoom_change_writes_position(self, tmp_path: Path) -> None:
        project = parse_project_fresh(SAMPLE)
        _zoom(_camera(project)).value = 3000

        position = _reopened_position(project, tmp_path)

        assert position.value == pytest.approx([720.0, 405.0, -1000.0])
        assert position.is_modified

    def test_keyed_zoom_change_at_time_0_writes_position(self, tmp_path: Path) -> None:
        project = parse_project_fresh(SAMPLE)
        zoom = _zoom(_camera(project))
        zoom.set_value_at_time(0, 3000)
        zoom.set_value_at_time(2, 1000)

        position = _reopened_position(project, tmp_path)

        assert position.value == pytest.approx([720.0, 405.0, -1000.0])
        assert position.is_modified

    def test_untouched_position_stays_left_out(self, tmp_path: Path) -> None:
        position = _reopened_position(parse_project_fresh(SAMPLE), tmp_path)

        assert position._tdsb is not None
        assert position._tdsb.synthetic
        assert position.value == pytest.approx([720.0, 405.0, -1000.0])
        assert not position.is_modified
