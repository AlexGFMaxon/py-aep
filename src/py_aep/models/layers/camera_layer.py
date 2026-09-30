from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from ...binary.layer_chunks import LdtaChunk
from ...enums import LayerType
from ..preferences import label_index
from .layer import Layer

if TYPE_CHECKING:
    from ..items.composition import CompItem
    from ..properties.property import Property
    from ..properties.property_group import PropertyGroup


class CameraLayer(Layer):
    """
    The CameraLayer object represents a camera layer within a composition.

    Example:
        ```python
        from py_aep import parse

        app = parse("project.aep")
        comp = app.project.compositions[0]
        camera = comp.camera_layers[0]
        print(camera.name)
        ```

    Info:
        `CameraLayer` is a subclass of [Layer][] object. All methods and
        attributes of [Layer][] are available when working with `CameraLayer`.

    See: https://ae-scripting.docsforadobe.dev/layer/cameralayer/
    """

    _auto_name: str = "Camera"
    _fov_rad: float = 39.5978 * math.pi / 180
    # AE's default 50mm camera: zoom = width / 0.72 exactly
    # (2 * tan(fov/2) rounds to 0.72; AE uses the exact ratio).
    _zoom_dividend: float = 0.72

    @property
    def is_3d(self) -> bool:
        """Always `True`: a camera / light layer only exists in 3D space.
        Read-only."""
        return True

    def _default_position(self) -> list[float]:
        """AE's default Position: the camera's Zoom in front of the comp centre.

        It follows the current Zoom (pre-expression, at time 0): AE leaves a
        Position still there out of the file and places the camera from the
        Zoom on open (measured on AE 2026, also after the Zoom was later
        keyframed or given an expression).
        """
        comp = self.containing_comp
        options = cast("PropertyGroup", self["ADBE Camera Options Group"])
        zoom = cast("float", cast("Property", options["ADBE Camera Zoom"]).value)
        return [comp.width / 2.0, comp.height / 2.0, -zoom]

    def _write_out_position_off_zoom(self) -> None:
        """Materialize a left-out Position its Zoom no longer places.

        Changing the Zoom of a camera whose Position AE left out keeps the
        camera where it was: AE 2026 then writes the Position, which reads
        modified. Left out, it would reopen at the new Zoom instead. Run on
        save, so every way of changing the Zoom (its value, keyframes,
        `set_value_at_time`) is covered.
        """
        position = cast("Property", self.transform["ADBE Position"])
        if position._tdsb is None or not position._tdsb.synthetic:
            return
        if position.is_modified:
            position._ensure_materialized()

    @classmethod
    def _new(  # type: ignore[override]
        cls,
        *,
        name: str,
        layer_id: int,
        duration: float,
        containing_comp: CompItem,
        effect_param_defs: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> CameraLayer:
        ldta = LdtaChunk(
            layer_id=layer_id,
            label=label_index(
                containing_comp._project._preferences, "Camera Label Index 2", 4
            ),
            layer_type=LayerType.CAMERA,
            layer_flags_2=0x01,
            layer_name=name[:31] if len(name) > 31 else name,
        )
        ldta.out_point = duration
        ldta.three_d_layer = True
        return cast(
            "CameraLayer",
            super()._new(
                ldta=ldta,
                name=name,
                containing_comp=containing_comp,
                effect_param_defs=effect_param_defs,
            ),
        )
