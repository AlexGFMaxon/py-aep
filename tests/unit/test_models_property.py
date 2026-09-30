"""Tests for Property model parsing with strengthened assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from py_aep.binary.chunk import ListChunk
from py_aep.binary.misc_chunks import ThreeDPardChunk, TwoDPardChunk
from py_aep.binary.property_chunks import TdmnChunk, TdsnChunk
from py_aep.enums import (
    PropertyControlType,
)
from py_aep.parsers.effect import (
    _extract_point_default,
    _point_default_pixels,
    _resolve_effect_value,
)
from py_aep.parsers.property import parse_property_group

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "property"
BUGS_DIR = Path(__file__).parent.parent.parent / "samples" / "bugs"
LAYER_SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "layer"
LAYER_SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "layer"
VERSIONS_DIR = Path(__file__).parent.parent.parent / "samples" / "versions"
PROPERTY_SAMPLES_DIR = (
    Path(__file__).parent.parent.parent / "samples" / "models" / "property"
)


class TestResolveEffectValue:
    """Tests for _resolve_effect_value pure helper.

    A synthesized parameter is one the effect does not store, so it is at
    its pard default; `last_value` is another instance's, or stale.
    """

    @pytest.mark.parametrize(
        ("param_def", "control_type", "expected"),
        [
            pytest.param(
                {"property_control_type": PropertyControlType.ENUM, "default_value": 1},
                PropertyControlType.ENUM,
                (1, 1),
                id="enum_default_is_1_based",
            ),
            pytest.param(
                {"property_control_type": PropertyControlType.ENUM, "default_value": 3},
                PropertyControlType.ENUM,
                (3, 3),
                id="enum_default_3_stays_3",
            ),
            pytest.param(
                {
                    "property_control_type": PropertyControlType.ENUM,
                    "default_value": 1,
                    "last_value": 5,
                },
                PropertyControlType.ENUM,
                (1, 1),
                id="enum_ignores_last_value",
            ),
            pytest.param(
                {
                    "property_control_type": PropertyControlType.BOOLEAN,
                    "default_value": 1,
                    "last_value": 0,
                },
                PropertyControlType.BOOLEAN,
                (1, 1),
                id="boolean_ignores_last_value",
            ),
            pytest.param(
                {
                    "property_control_type": PropertyControlType.SCALAR,
                    "last_value": 42.0,
                    "default_value": 10.0,
                },
                PropertyControlType.SCALAR,
                (10.0, 10.0),
                id="general_prefers_the_default",
            ),
            pytest.param(
                {"property_control_type": PropertyControlType.SCALAR},
                PropertyControlType.SCALAR,
                (None, None),
                id="general_no_values_returns_none",
            ),
            pytest.param(
                {
                    "property_control_type": PropertyControlType.SCALAR,
                    "last_value": 7.0,
                },
                PropertyControlType.SCALAR,
                (7.0, 7.0),
                id="general_last_value_only_without_a_default",
            ),
            pytest.param(
                {
                    "property_control_type": PropertyControlType.TWO_D,
                    "last_value": [128.0, 256.0],
                    "default_value": [256.0, 512.0],
                },
                PropertyControlType.TWO_D,
                ([256.0, 512.0], [256.0, 512.0]),
                id="point_prefers_the_default",
            ),
        ],
    )
    def test_resolve_effect_value(
        self,
        param_def: dict[str, Any],
        control_type: PropertyControlType,
        expected: tuple[Any, Any],
    ) -> None:
        result = _resolve_effect_value("TEST-0001", param_def, control_type)
        assert result == expected


class TestTdsnWithoutUtf8:
    """A tdsn missing its Utf8 child degrades to the auto-name.

    Regression: `TdsnChunk.utf8` raised ValueError, escaping
    `parse_property_group`'s ChunkNotFoundError handler and failing
    the whole parse instead of falling back to the auto-name.
    """

    def test_property_group_falls_back_to_auto_name(self) -> None:
        tdgp = ListChunk(
            list_type="tdgp",
            chunks=[TdsnChunk(chunks=[])],  # no Utf8 child
        )
        group = parse_property_group(
            tdgp_chunk=tdgp,
            group_match_name="ADBE Transform Group",
            property_depth=1,
            effect_param_defs={},
            composition=cast(Any, None),
            tdmn=TdmnChunk(value="ADBE Transform Group"),
        )
        assert group._name_utf8 is None
        assert group.name == "Transform"


class TestPointDefaultPixels:
    """parT point values are a fraction of the layer times 512."""

    def test_fraction_of_the_layer(self) -> None:
        prop = cast(Any, None)
        assert _point_default_pixels(prop, [256.0, 512.0], (200.0, 100.0)) == [
            100.0,
            100.0,
        ]


class TestPointPardDefault:
    """A point pard's default is PF_PointDef's dephault, percentages of the
    layer (16.16 fixed in 2D, doubles in 3D)."""

    def test_two_d_default_is_a_percentage_of_the_layer(self) -> None:
        # CC Radial Fast Blur's Center in a CC 2013 project: last value
        # 25600 (100 times a fraction's), default 50%.
        body = TwoDPardChunk(
            last_value_x_raw=25600 * 128,
            last_value_y_raw=25600 * 128,
            default_x_raw=50 << 16,
            default_y_raw=50 << 16,
        )
        result: dict[str, Any] = {}
        _extract_point_default(body, result)
        default = _point_default_pixels(
            cast(Any, None), result["default_value"], (1920.0, 1080.0)
        )
        assert default == pytest.approx([960.0, 540.0])

    def test_three_d_default_z_is_a_percentage_of_the_height(self) -> None:
        body = ThreeDPardChunk(default_x=50.0, default_y=25.0, default_z=10.0)
        result: dict[str, Any] = {}
        _extract_point_default(body, result)
        default = _point_default_pixels(
            cast(Any, None), result["default_value"], (200.0, 100.0)
        )
        assert default == pytest.approx([100.0, 25.0, 10.0])

    def test_a_pard_without_a_default_declares_none(self) -> None:
        result: dict[str, Any] = {}
        _extract_point_default(TwoDPardChunk(), result)
        assert "default_value" not in result
