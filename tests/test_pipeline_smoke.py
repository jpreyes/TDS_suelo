from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tsd_suelo.config import PipelineConfig
from tsd_suelo.pipeline import run_build, run_scenario_forward
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
    (records_dir / "20200102000000_STA02.h5").write_text("not an hdf5 file", encoding="utf-8")

    out = tmp_path / "outputs"
    manifest = run_build(PipelineConfig(records_dir=records_dir, flatfiles_dir=flatfiles_dir, output_dir=out))

    assert manifest["rows"]["geo_targets_observed"] == 4
    assert manifest["rows"]["flatfile_records_available"] == 1
    assert (out / "geo_targets_observed.parquet").exists()
    assert (out / "geo_residuals.parquet").exists()
    assert (out / "latent_modes.parquet").exists()
    assert (out / "fault_candidates.parquet").exists()
    assert (out / "fault_candidates.geojson").exists()
    assert (out / "spatial_grid_nodes.parquet").exists()
    assert (out / "spatial_grid_edges.parquet").exists()
    assert (out / "spatial_probability_heatmap.geojson").exists()
    assert (out / "spectral_record_signatures.parquet").exists()
    assert (out / "spectral_node_dynamics.parquet").exists()
    assert (out / "spectral_edge_transmissibility.parquet").exists()
    assert (out / "spectral_dynamic_heatmap.geojson").exists()
    assert (out / "kozyrev_ultrametric_nodes.parquet").exists()
    assert (out / "kozyrev_ultrametric_edges.parquet").exists()
    assert (out / "kozyrev_heatmap.geojson").exists()
    assert (out / "compatible_dynamics.parquet").exists()
    assert (out / "forward_conditioning_profiles.parquet").exists()
    assert (out / "atlas_geologico.geojson").exists()
    geo = pd.read_parquet(out / "geo_targets_observed.parquet")
    assert {"distance_km", "backazimuth_deg", "pga_h_g"}.issubset(geo.columns)
    assert set(geo["observed_source"]) == {"h5", "flatfile"}
    assert (out / "results_report.html").exists()
    report_html = (out / "results_report.html").read_text(encoding="utf-8")
    assert "Mapa Interactivo Chile" in report_html
    assert "interactive-map" in report_html
    assert "Forward Condicionado" in report_html
    assert "Mapa De Calor Espacial" in report_html
    faults = pd.read_parquet(out / "fault_candidates.parquet")
    assert {"candidate_id", "fault_candidate_score", "fault_probability_pct", "strike_deg"}.issubset(faults.columns)
    spatial_nodes = pd.read_parquet(out / "spatial_grid_nodes.parquet")
    assert {"cell_id", "center_latitude_deg", "anomaly_probability_pct", "probability_basis"}.issubset(spatial_nodes.columns)
    spatial_edges = pd.read_parquet(out / "spatial_grid_edges.parquet")
    assert {"from_cell_id", "to_cell_id", "fault_probability_pct", "edge_family"}.issubset(spatial_edges.columns)
    spectral_nodes = pd.read_parquet(out / "spectral_node_dynamics.parquet")
    assert {"cell_id", "spectral_dynamic_probability_pct", "probability_basis"}.issubset(spectral_nodes.columns)
    spectral_edges = pd.read_parquet(out / "spectral_edge_transmissibility.parquet")
    assert {"from_cell_id", "to_cell_id", "spectral_transfer_probability_pct", "edge_family"}.issubset(spectral_edges.columns)
    nodes = pd.read_parquet(out / "kozyrev_ultrametric_nodes.parquet")
    assert {"node_id", "failure_probability_pct", "probability_basis"}.issubset(nodes.columns)
    edges = pd.read_parquet(out / "kozyrev_ultrametric_edges.parquet")
    assert {"from_node", "to_node", "edge_probability_pct", "edge_family"}.issubset(edges.columns)
    compatible = pd.read_parquet(out / "compatible_dynamics.parquet")
    assert {"dynamic_anomaly_score", "forward_support_weight", "compatible_dynamics_status"}.issubset(compatible.columns)
    profiles = pd.read_parquet(out / "forward_conditioning_profiles.parquet")
    assert {"context_type", "context_id", "n_records"}.issubset(profiles.columns)
    scenario_manifest = run_scenario_forward(
        PipelineConfig(records_dir=records_dir, flatfiles_dir=flatfiles_dir, output_dir=out),
        scenario_name="santiago_sw_m75",
        receiver_latitude_deg=-33.4489,
        receiver_longitude_deg=-70.6693,
        source_distance_km=100.0,
        source_direction="suroeste",
        source_bearing_deg=None,
        source_latitude_deg=None,
        source_longitude_deg=None,
        mw=7.5,
        vs30_m_s=600.0,
        depth_km=30.0,
        tectonic_type="scenario",
        analog_top_n=20,
        top_n=20,
    )
    assert scenario_manifest["rows"]["analogs"] > 0
    assert (out / "webmap_spatial_probability.geojson").exists()
    assert (out / "webmap_spectral_dynamic.geojson").exists()
    assert (out / "webmap_fault_candidates.geojson").exists()
    assert (out / "webmap_kozyrev_probability.geojson").exists()
    assert (out / "forward_scenario_result.csv").exists()
    assert (out / "forward_scenario.geojson").exists()
    assert "Escenario Forward" in (out / "results_report.html").read_text(encoding="utf-8")
    direct_manifest = run_scenario_forward(
        PipelineConfig(records_dir=records_dir, flatfiles_dir=flatfiles_dir, output_dir=out),
        scenario_name="direct_source_m75",
        receiver_latitude_deg=-33.4489,
        receiver_longitude_deg=-70.6693,
        source_distance_km=100.0,
        source_direction=None,
        source_bearing_deg=None,
        source_latitude_deg=-34.0,
        source_longitude_deg=-71.0,
        mw=7.5,
        vs30_m_s=600.0,
        depth_km=30.0,
        tectonic_type="scenario",
        analog_top_n=20,
        top_n=20,
    )
    assert direct_manifest["scenario"]["source_direction"] is None
    assert direct_manifest["scenario"]["source_distance_km"] != 100.0
    assert (out / "waveform_targets_errors.csv").exists()
    assert (out / "forward_conditioning_template.json").exists()
    meta = pd.read_json(out / "waveform_targets_observed.meta.json", typ="series")
    assert int(meta["h5_errors"]) == 1

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
    assert reused["rows"]["spatial_grid_nodes"] == manifest["rows"]["spatial_grid_nodes"]
    assert reused["rows"]["spectral_node_dynamics"] == manifest["rows"]["spectral_node_dynamics"]
    assert reused["rows"]["fault_candidates"] == manifest["rows"]["fault_candidates"]
    assert reused["rows"]["kozyrev_ultrametric_nodes"] == manifest["rows"]["kozyrev_ultrametric_nodes"]
    assert reused["rows"]["compatible_dynamics"] == manifest["rows"]["compatible_dynamics"]
