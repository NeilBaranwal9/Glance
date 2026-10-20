####################################################################################################
"""
Tests for the runtime ML layer: arbitrary-N schemas, dataset auto-detection,
live inference, and the parametric rocket simulation.

These cover the parts that must hold for *any* sensor set, not just the
benchmark: nothing here assumes 17 channels, a particular column layout, or a
specific dataset. Training is exercised on small synthetic tables so the suite
stays fast; the full-size models are built by ml/build_pretrained.py.
"""

####################################################################################################
# Imports

import numpy as np
import pytest

from app.ml.dataset import assign_roles, from_history
from app.ml.labels import (
    BaselineCalibration,
    stage_from_health,
    stage_from_rul,
)
from app.ml.schema import SensorSchema

####################################################################################################
# Schema -- must work for any channel count


@pytest.mark.parametrize("n_channels", [1, 2, 5, 17, 40])
def test_schema_handles_any_channel_count(n_channels):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(120, n_channels)) * np.arange(1, n_channels + 1)
    names = [f"ch{i}" for i in range(n_channels)]
    schema = SensorSchema.fit(values, names)
    assert schema.n_features == n_channels
    scaled = schema.transform(values)
    assert scaled.shape == values.shape
    assert scaled.dtype == np.float32


def test_schema_drops_constant_channels():
    values = np.column_stack([
        np.linspace(0, 1, 50),
        np.full(50, 3.0),          # constant -> dropped
        np.random.default_rng(1).normal(size=50),
    ])
    schema = SensorSchema.fit(values, ["varies", "flat", "noise"])
    assert schema.names == ["varies", "noise"]
    assert schema.dropped == ["flat"]


def test_schema_clips_out_of_range_but_keeps_it_visible():
    """Unseen extremes must stay outside [0, 1] rather than being flattened onto it."""
    schema = SensorSchema.fit(np.array([[0.0], [1.0]]), ["a"])
    assert schema.transform(np.array([[2.0]]))[0, 0] > 1.0
    assert schema.transform(np.array([[-1.0]]))[0, 0] < 0.0
    # ...but still bounded, so a wild reading cannot blow up the model input.
    assert schema.transform(np.array([[1e9]]))[0, 0] == pytest.approx(1.5)


def test_schema_round_trips_through_dict():
    schema = SensorSchema.fit(np.random.default_rng(2).normal(size=(30, 4)),
                              [f"c{i}" for i in range(4)])
    restored = SensorSchema.from_dict(schema.to_dict())
    assert restored.names == schema.names
    np.testing.assert_allclose(restored.minimum, schema.minimum)
    np.testing.assert_allclose(restored.maximum, schema.maximum)


def test_schema_select_reorders_source_columns():
    schema = SensorSchema.fit(np.array([[0.0, 10.0], [1.0, 20.0]]), ["a", "b"])
    # Source presents the same channels in the opposite order plus an extra.
    picked = schema.select(np.array([[99.0, 20.0, 1.0]]), ["junk", "b", "a"])
    np.testing.assert_allclose(picked, [[1.0, 20.0]])


def test_schema_select_rejects_missing_channels():
    schema = SensorSchema.fit(np.array([[0.0, 1.0], [1.0, 2.0]]), ["a", "b"])
    with pytest.raises(KeyError):
        schema.select(np.array([[1.0]]), ["a"])


####################################################################################################
# Dataset role detection


def test_roles_detected_from_named_columns():
    import pandas as pd

    frame = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "RUL": [1, 0, 1, 0],
        "temp": [10.0, 11.0, 12.0, 13.0],
        "press": [1.0, 2.0, 3.0, 4.0],
    })
    dataset = assign_roles(frame)
    assert dataset.roles.unit == "engine_id"
    assert dataset.roles.cycle == "cycle"
    assert dataset.roles.rul == "RUL"
    assert dataset.roles.sensors == ["temp", "press"]
    assert dataset.roles.has_run_to_failure


def test_unlabelled_table_yields_no_rul():
    import pandas as pd

    frame = pd.DataFrame({f"s{i}": np.random.default_rng(3).normal(size=20) for i in range(4)})
    dataset = assign_roles(frame)
    assert dataset.n_sensors == 4
    assert not dataset.roles.has_run_to_failure
    assert dataset.rul() is None
    # A synthetic cycle index is still available for windowing.
    assert len(dataset.cycles()) == 20


def test_non_numeric_columns_are_ignored_not_fatal():
    import pandas as pd

    frame = pd.DataFrame({"label": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    dataset = assign_roles(frame)
    assert dataset.roles.sensors == ["v"]
    assert "label" in dataset.roles.ignored


def test_from_history_builds_a_dataset_from_live_capture():
    history = {
        "alt": [{"timestamp": t, "value": t, "filtered_value": float(t)} for t in range(40)],
        "vel": [{"timestamp": t, "value": t, "filtered_value": float(t * 2)} for t in range(40)],
    }
    dataset = from_history(history)
    assert dataset.n_sensors == 2
    assert dataset.n_rows == 40


####################################################################################################
# Health / stage mapping


def test_baseline_calibration_is_monotone_and_bounded():
    scores = np.random.default_rng(4).normal(1.0, 0.1, 400)
    calibration = BaselineCalibration.fit(scores)
    probe = np.array([calibration.reference,
                      calibration.reference + calibration.spread,
                      calibration.reference + 10 * calibration.spread])
    health = calibration.health(probe)
    assert health[0] == pytest.approx(100.0)
    assert health[0] > health[1] > health[2] > 0.0


def test_anomaly_threshold_matches_health_fifty():
    """The flag and the health score must agree on where 'not nominal' starts."""
    calibration = BaselineCalibration.fit(np.random.default_rng(5).normal(2.0, 0.3, 300))
    assert calibration.health(np.array([calibration.threshold]))[0] == pytest.approx(50.0)


def test_degenerate_baseline_does_not_divide_by_zero():
    calibration = BaselineCalibration.fit(np.full(50, 0.5))
    assert np.isfinite(calibration.spread) and calibration.spread > 0
    assert np.isfinite(calibration.health(np.array([0.5]))[0])


def test_stage_bands_agree_between_health_and_rul():
    # 125-cycle ceiling maps RUL 75 -> health 60 and RUL 30 -> health 24.
    np.testing.assert_array_equal(
        stage_from_rul(np.array([125.0, 76.0, 75.0, 31.0, 30.0, 0.0])),
        stage_from_health(np.array([100.0, 61.0, 60.0, 25.0, 24.0, 0.0])),
    )


####################################################################################################
# Rocket simulation -- geometry must actually change the flight


def test_mesh_rebuilds_for_each_nose_shape():
    from app.core.rocket_sim import NOSE_SHAPES, RocketParams

    for shape in NOSE_SHAPES:
        mesh = RocketParams(nose_shape=shape).build_mesh()
        assert mesh.n_faces > 0
        assert np.isfinite(mesh.vertices).all()


def test_fin_count_changes_the_mesh_and_the_drag():
    from app.core.rocket_sim import RocketParams

    three, six = RocketParams(fin_count=3), RocketParams(fin_count=6)
    assert six.build_mesh().n_faces > three.build_mesh().n_faces
    assert six.effective_cd > three.effective_cd


def test_heavier_rocket_flies_lower():
    from app.core.rocket_sim import RocketParams, simulate

    light = simulate(RocketParams(dry_mass=2.0))
    heavy = simulate(RocketParams(dry_mass=8.0))
    assert heavy.apogee < light.apogee


def test_more_thrust_flies_higher():
    from app.core.rocket_sim import RocketParams, simulate

    weak = simulate(RocketParams(motor_thrust=200.0))
    strong = simulate(RocketParams(motor_thrust=600.0))
    assert strong.apogee > weak.apogee


def test_degradation_drives_the_engine_health_channels():
    from app.core.rocket_sim import RocketParams, simulate

    healthy = simulate(RocketParams(degradation=0.0))
    worn = simulate(RocketParams(degradation=0.9))

    def during_burn(flight, channel):
        # Flights differ in duration, so each needs its own burn mask.
        mask = flight.time <= RocketParams().burn_time
        return flight.channels[channel][mask].max()

    assert during_burn(worn, "nozzle_temp_c") > during_burn(healthy, "nozzle_temp_c")
    assert during_burn(worn, "vibration_g") > during_burn(healthy, "vibration_g")
    assert worn.health_truth[-1] < healthy.health_truth[-1]


def test_health_truth_is_monotone_non_increasing():
    """Erosion is permanent damage, so the ground-truth label must never recover."""
    from app.core.rocket_sim import RocketParams, simulate

    truth = simulate(RocketParams(degradation=0.6)).health_truth
    assert np.all(np.diff(truth) <= 1e-9)


def test_health_truth_is_not_exposed_as_an_input_channel():
    """Feeding the label in as a feature would leak the answer."""
    from app.core.rocket_sim import RocketParams, simulate

    result = simulate(RocketParams())
    assert "health_truth" not in result.names
    assert len(result.health_truth) == len(result.time)


####################################################################################################
# End-to-end: train small, then stream


@pytest.fixture(scope="module")
def tiny_models():
    """A deliberately small 5-channel model, trained on synthetic degradation."""
    pytest.importorskip("torch")
    pytest.importorskip("catboost")
    from app.ml.trainers import train_all

    rng = np.random.default_rng(11)
    n = 320
    t = np.linspace(0, 1, n)
    decay = np.clip((t - 0.5) / 0.5, 0, 1) ** 2
    values = np.column_stack([
        50 + rng.normal(0, 0.3, n) + 6 * decay,
        300 - 40 * decay + rng.normal(0, 1, n),
        rng.normal(0, 1, n),
        10 + rng.normal(0, 0.2, n),
        5 + 3 * decay + rng.normal(0, 0.2, n),
    ])
    return train_all(values, [f"ch{i}" for i in range(5)], source="unit-test")


def test_training_produces_every_component(tiny_models):
    assert tiny_models.n_features == 5
    for component in (tiny_models.autoencoder, tiny_models.health_model,
                      tiny_models.stage_model, tiny_models.gp_prototype):
        assert component is not None
    assert len(tiny_models.feature_importances) == 5


def test_online_engine_streams_and_scores_itself(tiny_models):
    from app.ml.online import OnlineMLEngine

    rng = np.random.default_rng(12)
    n = 240
    decay = np.clip((np.linspace(0, 1, n) - 0.5) / 0.5, 0, 1) ** 2
    stream = np.column_stack([
        50 + rng.normal(0, 0.3, n) + 6 * decay,
        300 - 40 * decay + rng.normal(0, 1, n),
        rng.normal(0, 1, n),
        10 + rng.normal(0, 0.2, n),
        5 + 3 * decay + rng.normal(0, 0.2, n),
    ])
    engine = OnlineMLEngine(gp_every=6)
    engine.attach(tiny_models)

    for i, row in enumerate(stream):
        result = engine.push(row, timestamp=float(i))
        assert result is not None
        assert 0.0 <= result.health <= 100.0
        assert 0 <= result.stage <= 2
        assert np.isfinite(result.anomaly_score)

    summary = engine.live_summary()
    assert summary["samples"] == n
    # Forecasts must have matured and been scored against reality.
    assert summary["forecast_scored"] > 0
    assert np.isfinite(summary["forecast_rmse"])
    assert 0.0 <= summary["forecast_coverage"] <= 1.0
    # Health should fall as the injected degradation grows.
    assert engine.health_history[-1] < engine.health_history[0]


def test_engine_rejects_wrong_channel_count(tiny_models):
    from app.ml.online import OnlineMLEngine

    engine = OnlineMLEngine()
    engine.attach(tiny_models)
    with pytest.raises(ValueError):
        engine.push(np.zeros(3))


def test_push_mapping_tolerates_missing_channels(tiny_models):
    from app.ml.online import OnlineMLEngine

    engine = OnlineMLEngine()
    engine.attach(tiny_models)
    engine.push(np.zeros(tiny_models.n_features))
    # Only one channel supplied; the rest should hold their previous value.
    assert engine.push_mapping({tiny_models.schema.names[0]: 1.0}) is not None


def test_reset_clears_run_state_but_keeps_the_model(tiny_models):
    from app.ml.online import OnlineMLEngine

    engine = OnlineMLEngine()
    engine.attach(tiny_models)
    for _ in range(20):
        engine.push(np.zeros(tiny_models.n_features))
    engine.reset()
    assert engine.live_summary()["samples"] == 0
    assert engine.is_ready


def test_bundle_save_load_round_trip(tiny_models, tmp_path):
    from app.ml.online import OnlineMLEngine
    from app.ml.persist import load_bundle, save_bundle

    save_bundle(tiny_models, tmp_path / "bundle")
    restored = load_bundle(tmp_path / "bundle")
    assert restored.n_features == tiny_models.n_features
    assert restored.schema.names == tiny_models.schema.names

    sample = np.full(tiny_models.n_features, 0.5)
    before, after = OnlineMLEngine(), OnlineMLEngine()
    before.attach(tiny_models)
    after.attach(restored)
    a = before.push(sample)
    b = after.push(sample)
    assert a.health == pytest.approx(b.health, abs=1e-6)
    assert a.anomaly_score == pytest.approx(b.anomaly_score, abs=1e-9)
    assert a.stage == b.stage


####################################################################################################
# The runtime ML layer must stay Qt-free


def test_app_ml_package_has_no_qt_imports():
    import ast
    from pathlib import Path

    banned = ("PySide6", "PyQt5", "PyQt6")
    offenders = []
    root = Path(__file__).resolve().parents[1] / "app" / "ml"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}:{node.lineno} {n}" for n in names
                          if n.split(".")[0] in banned]
    assert not offenders, f"Qt imports found in app/ml/: {offenders}"


def test_rocket_sim_has_no_qt_imports():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "app" / "core" / "rocket_sim.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = (node.module if isinstance(node, ast.ImportFrom)
                      else node.names[0].name) or ""
            assert not module.startswith(("PySide6", "PyQt")), f"Qt import: {module}"
