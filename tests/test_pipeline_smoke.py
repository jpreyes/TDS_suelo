from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tsd_suelo.config import PipelineConfig
from tsd_suelo.pipeline import run_build
from tsd_suelo.utils import parse_numeric_vector


def _write_h5(path: Path, event_id: str, station_id: str, amp: float, dt: float = 0.01) -> None:
    n = 512
    t = np.arange(n) * dt
    e = amp * np.sin(2.0 * np.pi * 2.0 * t)
    n_acc = 0.8 * amp * np.sin(2.0 * np.pi * 3.0 * t)
    z = 0.4 * amp * np.sin(2.0 * np.pi * 4.0 * t)
    with h5py.File(path, "w") as h5:
        processed = h5.create_group("Processed_data")
        for name, data in {"E_acc": e, "N_acc": n_acc, "Z_acc": z}.items():
            processed.create_dataset(name, data=data.reshape(1, -1))
        metadata = h5.create_group("metadata")
        event = metadata.create_group("event")
        event.create_dataset("EventID_BM16", data=np.array([[event_id.encode()]]))
        event.create_dataset("UTC_event", data=np.array([[b"2020-01-01T00:00:00Z"]]))
        record = metadata.create_group("record")
        record.create_dataset("RecordID", data=np.array([[f"{event_id}_{station_id}".encode()]]))
        record.create_dataset("Sta_Name", data=np.array([[station_id.encode()]]))
        record.create_dataset("dt", data=np.array([dt]))
        record.create_dataset("Dtot", data=np.array([n * dt]))
        record.create_dataset("tP_rec", data=np.array([1.0]))
        record.create_dataset("tS_rec", data=np.array([2.0]))
        record.create_dataset("P_duration", data=np.array([0.5]))
        record.create_dataset("S_duration", data=np.array([1.5]))
        record.create_dataset("S_end", data=np.array([3.5]))
        record.create_dataset("CodaStart", data=np.array([3.5]))
        record.create_dataset("CodaEnd", data=np.array([5.0]))
        record.create_dataset("NoiseStart", data=np.array([0.0]))
        record.create_dataset("Repi", data=np.array([20.0]))
        record.create_dataset("Rhyp", data=np.array([22.0]))
        record.create_dataset("IfProcessed_Rec", data=np.array([[b"Y"]]))
        station = metadata.create_group("station")
        station.create_dataset("Sta_Name", data=np.array([[station_id.encode()]]))


def _write_flatfiles(flatfiles_dir: Path) -> None:
    flatfiles_dir.mkdir(parents=True)
    events = pd.DataFrame(
        [
            {"EventNumber": 1, "EventID": "20200101000000", "Latitude_deg": -33.0, "Longitude_deg": -71.0, "Depth_km": 20.0, "Mw": 6.0, "TectonicType": "Interface"},
            {"EventNumber": 2, "EventID": "20200102000000", "Latitude_deg": -34.0, "Longitude_deg": -72.0, "Depth_km": 30.0, "Mw": 6.5, "TectonicType": "Intraslab"},
        ]
    )
    events.to_csv(flatfiles_dir / "Flatfile_chileanEvents_v11.csv", index=False)
    records = pd.DataFrame(
        [
            {"RecordID": "r1", "EventID": "20200101000000", "StationID": "STA01", "EventLatitude_deg": -33.0, "EventLongitude_deg": -71.0, "EventDepth_km": 20.0, "Mw": 6.0, "TectonicType": "Interface", "StationLatitude_deg": -33.2, "StationLongitude_deg": -71.3, "StationVs30_m_s": 500, "Repi_km_": 20, "Rhyp_km_": 28, "Rrup_km_": 25, "PGA_g_N": 0.1, "PGA_g_E": 0.12, "PGA_g_Z": 0.04},
            {"RecordID": "r2", "EventID": "20200101000000", "StationID": "STA02", "EventLatitude_deg": -33.0, "EventLongitude_deg": -71.0, "EventDepth_km": 20.0, "Mw": 6.0, "TectonicType": "Interface", "StationLatitude_deg": -33.6, "StationLongitude_deg": -71.5, "StationVs30_m_s": 350, "Repi_km_": 60, "Rhyp_km_": 63, "Rrup_km_": 58, "PGA_g_N": 0.08, "PGA_g_E": 0.09, "PGA_g_Z": 0.03},
            {"RecordID": "r3", "EventID": "20200102000000", "StationID": "STA01", "EventLatitude_deg": -34.0, "EventLongitude_deg": -72.0, "EventDepth_km": 30.0, "Mw": 6.5, "TectonicType": "Intraslab", "StationLatitude_deg": -33.2, "StationLongitude_deg": -71.3, "StationVs30_m_s": 500, "Repi_km_": 90, "Rhyp_km_": 95, "Rrup_km_": 92, "PGA_g_N": 0.05, "PGA_g_E": 0.06, "PGA_g_Z": 0.02},
            {"RecordID": "r4", "EventID": "20200102000000", "StationID": "STA02", "EventLatitude_deg": -34.0, "EventLongitude_deg": -72.0, "EventDepth_km": 30.0, "Mw": 6.5, "TectonicType": "Intraslab", "StationLatitude_deg": -33.6, "StationLongitude_deg": -71.5, "StationVs30_m_s": 350, "Repi_km_": 70, "Rhyp_km_": 76, "Rrup_km_": 74, "PGA_g_N": 0.04, "PGA_g_E": 0.045, "PGA_g_Z": 0.015},
        ]
    )
    records.to_csv(flatfiles_dir / "Flatfile_chileanRecord_v8.csv", index=False)
    stations = pd.DataFrame(
        [
            {"CodeSta": "STA01", "Latitude": -33.2, "Longitude": -71.3, "Altitude": 10, "StaName": "Station 1", "Network": "T", "Geology": "Qa", "PreferedVs30": 500, "f0_HVSR": 5.0, "A0_HVSR": 2.0, "Kappa0": 0.03, "TopographicSlope": 0.1},
            {"CodeSta": "STA02", "Latitude": -33.6, "Longitude": -71.5, "Altitude": 20, "StaName": "Station 2", "Network": "T", "Geology": "Rock", "PreferedVs30": 350, "f0_HVSR": 8.0, "A0_HVSR": 3.0, "Kappa0": 0.04, "TopographicSlope": 0.2},
        ]
    )
    stations.to_csv(flatfiles_dir / "Flatfile_chileanSiteStation_v7.csv", index=False)


def test_parse_numeric_vector() -> None:
    assert parse_numeric_vector("['0.1'; '1.0'; '2.5']") == [0.1, 1.0, 2.5]


def test_pipeline_builds_observed_products(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    flatfiles_dir = records_dir / "flatfiles"
    records_dir.mkdir()
    _write_flatfiles(flatfiles_dir)
    _write_h5(records_dir / "20200101000000_STA01.h5", "20200101000000", "STA01", 100.0)
    _write_h5(records_dir / "20200101000000_STA02.h5", "20200101000000", "STA02", 80.0)
    _write_h5(records_dir / "20200102000000_STA01.h5", "20200102000000", "STA01", 60.0)

    out = tmp_path / "outputs"
    manifest = run_build(PipelineConfig(records_dir=records_dir, flatfiles_dir=flatfiles_dir, output_dir=out))

    assert manifest["rows"]["geo_targets_observed"] == 4
    assert manifest["rows"]["flatfile_records_available"] == 1
    assert (out / "geo_targets_observed.parquet").exists()
    assert (out / "geo_residuals.parquet").exists()
    assert (out / "latent_modes.parquet").exists()
    assert (out / "atlas_geologico.geojson").exists()
    geo = pd.read_parquet(out / "geo_targets_observed.parquet")
    assert {"distance_km", "backazimuth_deg", "pga_h_g"}.issubset(geo.columns)
    assert set(geo["observed_source"]) == {"h5", "flatfile"}
    assert (out / "results_report.html").exists()

    reused = run_build(
        PipelineConfig(
            records_dir=records_dir,
            flatfiles_dir=flatfiles_dir,
            output_dir=out,
            reuse_targets=True,
            reuse_products=True,
        )
    )
    assert reused["rows"]["geo_targets_observed"] == manifest["rows"]["geo_targets_observed"]
    assert reused["rows"]["geo_residuals"] == manifest["rows"]["geo_residuals"]
