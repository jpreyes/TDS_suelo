from __future__ import annotations

from tsd_suelo.geometry import azimuth_deg, haversine_km


def test_haversine_and_azimuth_are_reasonable() -> None:
    distance = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert 110.0 < distance < 112.5
    assert 89.0 < azimuth_deg(0.0, 0.0, 0.0, 1.0) < 91.0

