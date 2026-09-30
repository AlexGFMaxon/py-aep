"""The curves of the Curves effect (`ADBE CurvesCustom-0001`).

After Effects keeps them as the effect's own arbitrary data, an `aRbp`
chunk (see `CurvesArbpChunk` for the layout): a mode, then per channel (the
RGB master, red, green, blue, alpha) a 256-entry map and its points.

A curve through points is the natural cubic spline through them (second
derivative 0 at both ends, a straight line through two points). It is flat
beyond the first and last points, and clamped to 0..1 in 8 and 16 bpc. The
master applies after a channel's own curve,
`out = master(channel(in))`. Alpha has its own curve, and colour is never
unpremultiplied.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...binary.property_chunks import CurvesArbpChunk

CURVES_CHANNELS = ("rgb", "red", "green", "blue", "alpha")
"""Channel names, in After Effects' order: `rgb` is the master curve."""


class _NaturalSpline:
    """Natural cubic spline through `(x, y)` points, flat outside them.

    Points sharing an input keep the last one's output, as a curve can only
    pass through one output per input.
    """

    def __init__(self, points: Sequence[tuple[float, float]]) -> None:
        by_x = {float(x): float(y) for x, y in points}
        pts = sorted(by_x.items())
        self.xs = [p[0] for p in pts]
        self.ys = [p[1] for p in pts]
        n = len(pts)
        # Second derivatives, 0 at both ends.
        self.m = [0.0] * n
        if n < 3:
            return
        xs, ys = self.xs, self.ys
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]
        a = [0.0] * n
        b = [1.0] * n
        c = [0.0] * n
        d = [0.0] * n
        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
        for i in range(1, n):
            w = a[i] / b[i - 1]
            b[i] -= w * c[i - 1]
            d[i] -= w * d[i - 1]
        self.m[n - 1] = d[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            self.m[i] = (d[i] - c[i] * self.m[i + 1]) / b[i]

    def _segment(self, x: float) -> int:
        xs = self.xs
        lo, hi = 0, len(xs) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def __call__(self, x: float) -> float:
        xs, ys, m = self.xs, self.ys, self.m
        if len(xs) == 1 or x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = self._segment(x)
        h = xs[i + 1] - xs[i]
        u = (xs[i + 1] - x) / h
        v = 1.0 - u
        cubic = (u**3 - u) * m[i] + (v**3 - v) * m[i + 1]
        return u * ys[i] + v * ys[i + 1] + cubic * h * h / 6.0


class CurvesChannel:
    """One channel's curve: its points and its 8-bit map. Read-only."""

    def __init__(
        self,
        name: str,
        points: list[tuple[int, int]],
        lut: bytes,
        selected: int,
    ) -> None:
        self._name = name
        self._points = points
        self._map = lut
        self._selected = selected
        self._spline: _NaturalSpline | None = None

    @property
    def name(self) -> str:
        """The channel: `rgb` (the master), `red`, `green`, `blue` or `alpha`."""
        return self._name

    @property
    def points(self) -> list[tuple[int, int]]:
        """The live points, `(input, output)` in 0..255."""
        return list(self._points)

    @property
    def map(self) -> list[int]:
        """256 output levels, one per input level."""
        return list(self._map)

    @property
    def selected(self) -> int:
        """The index of the selected point, -1 for none."""
        return self._selected

    @property
    def is_identity(self) -> bool:
        """Whether the curve leaves every level unchanged."""
        return all(v == i for i, v in enumerate(self._map)) and all(
            x == y for x, y in self._points
        )

    def spline(self) -> _NaturalSpline:
        """The natural cubic spline through the points, in 0..255 on both axes."""
        if self._spline is None:
            self._spline = _NaturalSpline(self._points)
        return self._spline

    def __repr__(self) -> str:
        return f"CurvesChannel({self._name!r}, points={self._points!r})"


class Curves:
    """The five curves of a Curves effect.

    Read-only: py-aep does not write curves back to the project.
    """

    def __init__(self) -> None:
        self._version = 1
        self._mode = 1
        self._channels: dict[str, CurvesChannel] = {}

    @classmethod
    def _from_binary(cls, chunk: CurvesArbpChunk) -> Curves:
        """Build the curves from a Curves effect's aRbp layout."""
        obj = cls()
        obj._version = chunk.version
        obj._mode = chunk.mode
        for c, (name, record) in enumerate(zip(CURVES_CHANNELS, chunk.records)):
            lut = chunk.maps[256 * c : 256 * (c + 1)]
            points = record.points
            if len(points) < 2:
                # Nothing usable: the channel's map is all there is.
                points = [(0, lut[0]), (255, lut[255])]
            obj._channels[name] = CurvesChannel(name, points, lut, record.selected)
        return obj

    @property
    def version(self) -> int:
        """The data's version, 1."""
        return self._version

    @property
    def mode(self) -> int:
        """1 when the curves are drawn with points, 0 with the pencil."""
        return self._mode

    @property
    def uses_points(self) -> bool:
        """Whether the curves go through their points; if not, they were
        drawn with the pencil and their maps are what renders."""
        return self._mode != 0

    @property
    def channels(self) -> dict[str, CurvesChannel]:
        """The curves by channel name, in `CURVES_CHANNELS` order."""
        return dict(self._channels)

    @property
    def is_identity(self) -> bool:
        """Whether every curve leaves every level unchanged, as an untouched
        Curves does."""
        return all(ch.is_identity for ch in self._channels.values())

    def evaluate(self, channel: str, x: float, clamp: bool = True) -> float:
        """One channel's curve at `x` in 0..1, on its own (not through the
        master), as 16 bpc renders it. `clamp=False` leaves the spline's
        overshoot as 32 bpc does between the points."""
        ch = self._channels[channel]
        if self.uses_points:
            y = ch.spline()(x * 255.0) / 255.0
        else:
            f = min(max(x, 0.0), 1.0) * 255.0
            i = min(int(math.floor(f)), 254)
            t = f - i
            lut = ch._map
            y = (lut[i] * (1.0 - t) + lut[i + 1] * t) / 255.0
        return min(max(y, 0.0), 1.0) if clamp else y

    def __repr__(self) -> str:
        edited = [n for n, c in self._channels.items() if not c.is_identity]
        return f"Curves(mode={self._mode}, edited={edited})"
