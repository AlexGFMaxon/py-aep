"""Resizing a comp must not move the camera and light properties AE left out.

A camera or light Position, and a camera's Focus Distance, left at their
default are out of the file, and their defaults follow the comp's width,
height and pixel aspect. Changing those in AE 2026 writes the ones whose
default moves, keeping them where they were, so they read modified; left
out, they would reopen at the new default. In `camera_default_position.aep`
the `zoom_1000` camera's Position ([720, 405, -1000]) and Focus Distance
(2000, the 1440-wide comp's default zoom) are left out; in
`light_default_position.aep` so is the `1920x1080` Point light's Position.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import get_comp, parse_project_fresh

from py_aep import parse as parse_aep
from py_aep.models.layers import Layer
from py_aep.models.project import Project
from py_aep.models.properties.property import Property

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "layer"
CAMERA_SAMPLE = SAMPLES_DIR / "camera_default_position.aep"
LIGHT_SAMPLE = SAMPLES_DIR / "light_default_position.aep"


def _prop(layer: Layer, group: str, match_name: str) -> Property:
    prop = layer[group][match_name]
    assert isinstance(prop, Property)
    return prop


def _reopened(
    project: Project, tmp_path: Path, comp_name: str, layer_name: str
) -> Layer:
    out = tmp_path / "out.aep"
    project.save(out)
    comp = get_comp(parse_aep(out).project, comp_name)
    return next(layer for layer in comp.layers if layer.name == layer_name)


def _assert_written(prop: Property, value: list[float] | float) -> None:
    assert prop._tdsb is not None
    assert not prop._tdsb.synthetic
    assert prop.value == pytest.approx(value)
    assert prop.is_modified


def _assert_left_out(prop: Property, value: list[float] | float) -> None:
    assert prop._tdsb is not None
    assert prop._tdsb.synthetic
    assert prop.value == pytest.approx(value)
    assert not prop.is_modified


class TestCameraCompResize:
    def test_width_writes_position_and_focus_distance(self, tmp_path: Path) -> None:
        # The comp is resized before any of its layers is read.
        project = parse_project_fresh(CAMERA_SAMPLE)
        get_comp(project, "zoom_1000").width = 1280

        camera = _reopened(project, tmp_path, "zoom_1000", "Camera")

        _assert_written(
            _prop(camera, "ADBE Transform Group", "ADBE Position"),
            [720.0, 405.0, -1000.0],
        )
        _assert_written(
            _prop(camera, "ADBE Camera Options Group", "ADBE Camera Focus Distance"),
            2000.0,
        )

    def test_pixel_aspect_writes_focus_distance_only(self, tmp_path: Path) -> None:
        # A camera's default Position follows its own Zoom, not the pixel
        # aspect, so AE leaves it out.
        project = parse_project_fresh(CAMERA_SAMPLE)
        get_comp(project, "zoom_1000").pixel_aspect = 2.0

        camera = _reopened(project, tmp_path, "zoom_1000", "Camera")

        _assert_left_out(
            _prop(camera, "ADBE Transform Group", "ADBE Position"),
            [720.0, 405.0, -1000.0],
        )
        _assert_written(
            _prop(camera, "ADBE Camera Options Group", "ADBE Camera Focus Distance"),
            2000.0,
        )

    def test_untouched_comps_are_not_loaded_on_save(self, tmp_path: Path) -> None:
        project = parse_project_fresh(CAMERA_SAMPLE)
        get_comp(project, "zoom_1000").width = 1280

        project.save(tmp_path / "out.aep")

        assert not get_comp(project, "moved")._layers_loaded


class TestLightCompResize:
    @pytest.mark.parametrize(
        ("attr", "value"), [("width", 1280), ("height", 720), ("pixel_aspect", 2.0)]
    )
    def test_writes_position(self, tmp_path: Path, attr: str, value: float) -> None:
        project = parse_project_fresh(LIGHT_SAMPLE)
        setattr(get_comp(project, "1920x1080"), attr, value)

        light = _reopened(project, tmp_path, "1920x1080", "Point")

        _assert_written(
            _prop(light, "ADBE Transform Group", "ADBE Position"),
            [1040.0, 460.0, -2000.0 / 3],
        )
