"""Deterministic synthetic data for the GridOps Copilot demo."""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 42


def _build_assets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "TX-1",
                "transformer",
                "400 kV",
                "Hitachi Energy",
                18,
                "critical",
                "400_kv_bus",
                "400_kv_bus",
            ),
            (
                "CB-401",
                "circuit_breaker",
                "400 kV",
                "Siemens Energy",
                10,
                "high",
                "400_kv_bus",
                "TX-1",
            ),
            (
                "CB-402",
                "circuit_breaker",
                "400 kV",
                "GE Vernova",
                16,
                "critical",
                "400_kv_bus",
                "TX-1",
            ),
            (
                "TX-2",
                "transformer",
                "220 kV",
                "BHEL",
                12,
                "high",
                "220_kv_bus",
                "220_kv_bus",
            ),
            (
                "CB-221",
                "circuit_breaker",
                "220 kV",
                "ABB",
                8,
                "medium",
                "220_kv_bus",
                "TX-2",
            ),
        ],
        columns=[
            "asset_id",
            "asset_type",
            "voltage_level",
            "manufacturer",
            "age_years",
            "criticality",
            "bus_group",
            "connected_to",
        ],
    )


def _build_maintenance_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "CB-402",
                "2025-11-12",
                "Elevated dynamic contact resistance",
                "Contacts cleaned and DCRM repeated",
                "high",
                "2026-06-15",
            ),
            (
                "CB-401",
                "2026-01-18",
                "Routine breaker inspection",
                "No corrective action required",
                "low",
                "2027-01-18",
            ),
            (
                "TX-1",
                "2026-02-20",
                "Minor FRA baseline deviation",
                "Engineering review requested",
                "medium",
                "2026-08-20",
            ),
            (
                "TX-2",
                "2026-03-10",
                "Routine oil and thermal inspection",
                "No abnormal condition found",
                "low",
                "2027-03-10",
            ),
            (
                "CB-221",
                "2026-04-05",
                "Operating mechanism lubrication",
                "Mechanism cleaned and lubricated",
                "medium",
                "2027-04-05",
            ),
        ],
        columns=[
            "asset_id",
            "date",
            "issue",
            "action_taken",
            "severity",
            "next_due_date",
        ],
    )


def _build_scada_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "TX-1",
                "2026-07-08T08:00:00Z",
                401.2,
                410.0,
                62.0,
                "normal",
                "NONE",
            ),
            (
                "TX-1",
                "2026-07-08T09:00:00Z",
                398.8,
                455.0,
                82.0,
                "warning",
                "TX_TEMP_HIGH",
            ),
            (
                "CB-401",
                "2026-07-08T08:15:00Z",
                400.6,
                390.0,
                44.0,
                "normal",
                "NONE",
            ),
            (
                "CB-402",
                "2026-07-08T08:15:00Z",
                399.9,
                398.0,
                47.0,
                "normal",
                "NONE",
            ),
            (
                "CB-402",
                "2026-07-08T09:15:00Z",
                399.4,
                430.0,
                68.0,
                "warning",
                "CB_TIMING_DEVIATION",
            ),
            (
                "TX-2",
                "2026-07-08T08:30:00Z",
                220.8,
                280.0,
                57.0,
                "normal",
                "NONE",
            ),
            (
                "CB-221",
                "2026-07-08T08:30:00Z",
                220.2,
                270.0,
                42.0,
                "normal",
                "NONE",
            ),
        ],
        columns=[
            "asset_id",
            "timestamp",
            "voltage_kv",
            "current_a",
            "temperature_c",
            "status",
            "alarm_code",
        ],
    )


def _dcrm_frame(
    *,
    timestamp: str,
    time_ms: np.ndarray,
    resistance: np.ndarray,
    travel: np.ndarray,
    coil_current: np.ndarray,
    label: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "breaker_id": "CB-402",
            "timestamp": timestamp,
            "time_ms": time_ms,
            "resistance_micro_ohm": resistance,
            "travel_mm": travel,
            "coil_current_A": coil_current,
            "operation_type": "close",
            "label": label,
        }
    )

    return frame.round(
        {
            "time_ms": 6,
            "resistance_micro_ohm": 6,
            "travel_mm": 6,
            "coil_current_A": 6,
        }
    )


def _build_dcrm_frames(
    random: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    healthy_time = np.linspace(0.0, 100.0, 201)
    healthy_phase = healthy_time / healthy_time.max()

    healthy_resistance = (
        100.0
        + 2.5 * np.sin(4.0 * np.pi * healthy_phase)
        + random.normal(0.0, 0.45, healthy_time.size)
    )
    healthy_travel = 100.0 / (1.0 + np.exp(-(healthy_time - 50.0) / 8.0))
    healthy_coil_current = 2.8 * np.exp(-0.5 * ((healthy_time - 25.0) / 10.0) ** 2)

    contact_spike = 55.0 * np.exp(-0.5 * ((healthy_time - 55.0) / 3.0) ** 2)
    contact_resistance = healthy_resistance + contact_spike

    delayed_time = np.linspace(0.0, 130.0, 201)
    delayed_phase = delayed_time / delayed_time.max()
    delayed_resistance = (
        100.0
        + 2.5 * np.sin(4.0 * np.pi * delayed_phase)
        + random.normal(0.0, 0.45, delayed_time.size)
    )
    delayed_travel = 100.0 / (1.0 + np.exp(-(delayed_time - 72.0) / 10.0))
    delayed_coil_current = 2.8 * np.exp(-0.5 * ((delayed_time - 35.0) / 13.0) ** 2)

    return {
        "dcrm_healthy.csv": _dcrm_frame(
            timestamp="2026-07-01T10:00:00Z",
            time_ms=healthy_time,
            resistance=healthy_resistance,
            travel=healthy_travel,
            coil_current=healthy_coil_current,
            label="healthy",
        ),
        "dcrm_fault_contact_wear.csv": _dcrm_frame(
            timestamp="2026-07-08T10:00:00Z",
            time_ms=healthy_time,
            resistance=contact_resistance,
            travel=healthy_travel,
            coil_current=healthy_coil_current,
            label="contact_wear_suspected",
        ),
        "dcrm_fault_mechanism_delay.csv": _dcrm_frame(
            timestamp="2026-07-08T10:30:00Z",
            time_ms=delayed_time,
            resistance=delayed_resistance,
            travel=delayed_travel,
            coil_current=delayed_coil_current,
            label="mechanism_delay_suspected",
        ),
    }


def _fra_frame(
    *,
    timestamp: str,
    frequency: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    label: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "transformer_id": "TX-1",
            "timestamp": timestamp,
            "frequency_hz": frequency,
            "magnitude_db": magnitude,
            "phase_deg": phase,
            "winding": "HV",
            "label": label,
        }
    )

    return frame.round(
        {
            "frequency_hz": 6,
            "magnitude_db": 6,
            "phase_deg": 6,
        }
    )


def _build_fra_frames(
    random: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    frequency = np.logspace(1.0, 6.0, 512)
    log_frequency = np.log10(frequency)

    healthy_magnitude = (
        -10.0
        - 13.0 * log_frequency
        + 4.5 * np.sin(2.3 * log_frequency)
        + 1.8 * np.sin(8.2 * log_frequency)
        + random.normal(0.0, 0.15, frequency.size)
    )
    healthy_phase = (
        -15.0 * log_frequency
        + 25.0 * np.sin(1.7 * log_frequency)
        + 6.0 * np.sin(6.0 * log_frequency)
        + random.normal(0.0, 0.4, frequency.size)
    )

    winding_region = np.exp(-0.5 * ((log_frequency - 4.0) / 0.45) ** 2)
    winding_magnitude = (
        healthy_magnitude
        - 10.0 * winding_region
        + 2.0 * np.sin(14.0 * log_frequency) * winding_region
    )
    winding_phase = healthy_phase + 35.0 * winding_region

    core_region = np.exp(-0.5 * ((log_frequency - 2.8) / 0.50) ** 2)
    core_magnitude = healthy_magnitude + 8.0 * core_region
    core_phase = healthy_phase - 28.0 * core_region

    insulation_region = 1.0 / (1.0 + np.exp(-(log_frequency - 4.5) * 5.0))
    insulation_magnitude = healthy_magnitude - 12.0 * insulation_region
    insulation_phase = healthy_phase + 15.0 * insulation_region * np.sin(18.0 * log_frequency)

    return {
        "fra_healthy.csv": _fra_frame(
            timestamp="2026-07-01T11:00:00Z",
            frequency=frequency,
            magnitude=healthy_magnitude,
            phase=healthy_phase,
            label="healthy",
        ),
        "fra_fault_winding_shift.csv": _fra_frame(
            timestamp="2026-07-08T11:00:00Z",
            frequency=frequency,
            magnitude=winding_magnitude,
            phase=winding_phase,
            label="winding_deformation_suspected",
        ),
        "fra_fault_core_clamping.csv": _fra_frame(
            timestamp="2026-07-08T11:15:00Z",
            frequency=frequency,
            magnitude=core_magnitude,
            phase=core_phase,
            label="core_clamping_issue_suspected",
        ),
        "fra_fault_insulation.csv": _fra_frame(
            timestamp="2026-07-08T11:30:00Z",
            frequency=frequency,
            magnitude=insulation_magnitude,
            phase=insulation_phase,
            label="insulation_related_abnormality_suspected",
        ),
    }


def generate_synthetic_data(
    output_directory: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate all deterministic demo datasets."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    random = np.random.default_rng(seed)

    frames = {
        "assets.csv": _build_assets(),
        "maintenance_logs.csv": _build_maintenance_logs(),
        "scada_events.csv": _build_scada_events(),
        **_build_fra_frames(random),
        **_build_dcrm_frames(random),
    }

    generated: dict[str, Path] = {}

    for filename, frame in frames.items():
        destination = output_directory / filename
        frame.to_csv(destination, index=False, lineterminator="\n")
        generated[filename] = destination

    return generated


def main() -> None:
    """Generate datasets in the repository's data directory."""

    project_root = Path(__file__).resolve().parents[3]
    output_directory = project_root / "data" / "synthetic"

    generated = generate_synthetic_data(output_directory)

    for path in generated.values():
        print(path.relative_to(project_root))


if __name__ == "__main__":
    main()
