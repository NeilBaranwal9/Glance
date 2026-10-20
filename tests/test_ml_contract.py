####################################################################################################
"""
Guards the Phase A -> Phase B contract for the models in app/models/.

These tests are the tripwire for the UI layer: they assert that every artifact is
loadable, that each *_meta.json declares what app/ needs (feature order,
normalization statistics, output shape and meaning), and that the four models
agree on one shared input feature contract. They deliberately do NOT retrain
anything -- they run in a couple of seconds against whatever is serialized.

If the artifacts have not been trained yet, every test skips rather than fails:

    python ml/train_models.py --model all
"""

####################################################################################################
# Imports

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "app" / "models"
PROCESSED_NPZ = REPO_ROOT / "data" / "processed" / "fd001.npz"

MODEL_NAMES = ("autoencoder", "gpr", "lstm", "catboost_health")

# Keys every sidecar must carry, whatever the model.
REQUIRED_META_KEYS = (
    "name",
    "task",
    "schema_version",
    "dataset",
    "input",
    "normalization",
    "output",
    "artifacts",
    "inference",
    "metrics",
)


def _meta_or_skip(name):
    path = ARTIFACT_DIR / f"{name}_meta.json"
    if not path.exists():
        pytest.skip(f"{path.name} not trained yet; run: python ml/train_models.py --model {name}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metas():
    return {name: _meta_or_skip(name) for name in MODEL_NAMES}


####################################################################################################
# Sidecar schema


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_meta_has_required_keys(name):
    meta = _meta_or_skip(name)
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    assert not missing, f"{name}_meta.json is missing {missing}"
    assert meta["name"] == name
    assert meta["schema_version"] == "1.0"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_meta_declares_input_contract(name):
    meta = _meta_or_skip(name)
    block = meta["input"]
    assert block["feature_names"], "feature_names must not be empty"
    assert block["n_features"] == len(block["feature_names"])
    assert block["dtype"] == "float32"
    assert block["shape"], "an input shape is required"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_meta_declares_normalization_stats(name):
    meta = _meta_or_skip(name)
    block = meta["normalization"]
    assert block["method"] == "min_max"
    assert len(block["min"]) == len(block["max"]) == len(block["feature_names"])
    assert all(hi >= lo for lo, hi in zip(block["min"], block["max"]))
    assert "formula" in block, "app/ needs the exact formula to apply at inference"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_meta_declares_output_meaning(name):
    meta = _meta_or_skip(name)
    assert meta["output"].get("meaning"), "output.meaning is what the UI renders"
    assert meta["output"].get("shape") is not None


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_declared_artifacts_exist_on_disk(name):
    meta = _meta_or_skip(name)
    assert meta["artifacts"], f"{name} declares no artifact files"
    for kind, filename in meta["artifacts"].items():
        path = ARTIFACT_DIR / filename
        assert path.exists(), f"{name} declares {kind}={filename} but {path} is absent"
        assert path.stat().st_size > 0, f"{path} is empty"


####################################################################################################
# Cross-model consistency


def test_row_models_share_one_feature_contract(metas):
    """autoencoder, lstm and catboost must consume identical features in identical order."""
    orders = {
        name: metas[name]["input"]["feature_names"]
        for name in ("autoencoder", "lstm", "catboost_health")
    }
    reference = orders["autoencoder"]
    for name, order in orders.items():
        assert order == reference, f"{name} feature order diverges from the shared contract"
    assert len(reference) == 17


def test_gpr_sensor_is_part_of_the_shared_contract(metas):
    sensor = metas["gpr"]["input"]["feature_names"][0]
    assert sensor in metas["autoencoder"]["input"]["feature_names"]


def test_normalization_stats_agree_across_models(metas):
    reference = metas["autoencoder"]["normalization"]
    for name in ("lstm", "catboost_health"):
        block = metas[name]["normalization"]
        assert block["min"] == reference["min"], f"{name} min stats diverge"
        assert block["max"] == reference["max"], f"{name} max stats diverge"


def test_lstm_window_matches_gpr_window(metas):
    """Both sequence models read the same rolling buffer length, so the UI keeps one."""
    assert metas["lstm"]["input"]["window_length"] == metas["gpr"]["input"]["window_length"]


def test_lstm_declares_three_ordered_stage_labels(metas):
    output = metas["lstm"]["output"]
    assert output["class_labels"] == ["healthy", "degrading", "critical"]
    assert output["class_index"] == {"healthy": 0, "degrading": 1, "critical": 2}


def test_autoencoder_declares_a_usable_threshold(metas):
    thresholds = metas["autoencoder"]["thresholds"]
    assert thresholds["reconstruction_error"] > 0
    assert "decision_rule" in thresholds


####################################################################################################
# Inference smoke tests -- batch size 1, which is how the UI will call


def _single_row(meta):
    return np.zeros((1, meta["input"]["n_features"]), dtype=np.float32)


def test_autoencoder_runs_on_a_single_row(metas):
    onnxruntime = pytest.importorskip("onnxruntime")
    meta = metas["autoencoder"]
    path = ARTIFACT_DIR / meta["artifacts"]["onnx"]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    row = _single_row(meta)
    out = session.run(None, {session.get_inputs()[0].name: row})[0]
    assert out.shape == (1, meta["input"]["n_features"])
    score = float(((out - row) ** 2).mean())
    assert np.isfinite(score)


def test_lstm_runs_on_a_single_window(metas):
    onnxruntime = pytest.importorskip("onnxruntime")
    meta = metas["lstm"]
    path = ARTIFACT_DIR / meta["artifacts"]["onnx"]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    window = meta["input"]["window_length"]
    x = np.zeros((1, window, meta["input"]["n_features"]), dtype=np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: x})[0]
    assert logits.shape == (1, 3)
    assert np.isfinite(logits).all()


def test_catboost_runs_on_a_single_row(metas):
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    meta = metas["catboost_health"]
    model = CatBoostRegressor()
    model.load_model(str(ARTIFACT_DIR / meta["artifacts"]["catboost"]))
    health = float(model.predict(_single_row(meta))[0])
    assert np.isfinite(health)


def test_gpr_conditions_and_forecasts_from_a_buffer(metas):
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    meta = metas["gpr"]
    window = meta["input"]["window_length"]
    horizon = meta["output"]["shape"][0]

    gp = joblib.load(ARTIFACT_DIR / meta["artifacts"]["sklearn_pickle"])
    # Exactly the documented call sequence: condition on the buffer, then predict.
    past_t = (np.arange(window) / window).reshape(-1, 1)
    future_t = (np.arange(window, window + horizon) / window).reshape(-1, 1)
    buffer = np.linspace(0.2, 0.6, window)
    gp.fit(past_t, buffer)
    mean, std = gp.predict(future_t, return_std=True)

    assert mean.shape == (horizon,)
    assert std.shape == (horizon,)
    assert (std > 0).all(), "a GP must report positive predictive uncertainty"
    assert np.isfinite(mean).all()


def test_gpr_fit_does_not_refit_hyperparameters(metas):
    """The app calls .fit() every tick; that must only condition, never re-optimize."""
    joblib = pytest.importorskip("joblib")
    meta = metas["gpr"]
    gp = joblib.load(ARTIFACT_DIR / meta["artifacts"]["sklearn_pickle"])
    assert gp.optimizer is None, "optimizer must be None or per-call cost explodes"

    window = meta["input"]["window_length"]
    past_t = (np.arange(window) / window).reshape(-1, 1)
    before = np.array(gp.kernel.theta, copy=True)
    gp.fit(past_t, np.linspace(0.1, 0.9, window))
    np.testing.assert_allclose(gp.kernel_.theta, before)


####################################################################################################
# The normalization block must be sufficient on its own


def test_meta_normalization_reproduces_the_cached_features(metas):
    """app/ starts from raw telemetry, so meta.json min/max must be all it needs."""
    if not PROCESSED_NPZ.exists():
        pytest.skip("processed cache absent; run: python ml/preprocess.py")
    pandas = pytest.importorskip("pandas")

    meta = metas["catboost_health"]
    block = meta["normalization"]
    raw_columns = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    raw = pandas.read_csv(
        REPO_ROOT / "data" / "cmapss" / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=raw_columns,
    )

    cached = np.load(PROCESSED_NPZ, allow_pickle=True)
    units = set(int(u) for u in cached["units_test"])
    frame = raw[raw["unit"].isin(units)].reset_index(drop=True)

    low = np.asarray(block["min"], dtype=np.float64)
    high = np.asarray(block["max"], dtype=np.float64)
    span = np.where((high - low) < 1e-12, 1.0, high - low)
    lo_clip, hi_clip = block["clip_range"]
    derived = np.clip(
        (frame[block["feature_names"]].to_numpy(dtype=np.float64) - low) / span, lo_clip, hi_clip
    ).astype(np.float32)

    np.testing.assert_allclose(derived, cached["tab_test_X"], atol=1e-6)


####################################################################################################
# ml/ must stay Qt-free


def test_ml_package_imports_without_qt():
    """Training and evaluation must run headless -- no Qt anywhere in ml/."""
    import subprocess
    import sys

    code = (
        "import sys; import ml.preprocess, ml.evaluate, ml.train_models, ml.artifacts; "
        "leaked = [m for m in sys.modules if m.startswith(('PySide6','PyQt5','PyQt6'))]; "
        "print('LEAKED' if leaked else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, f"Qt leaked into ml/: {result.stdout} {result.stderr}"


def test_ml_sources_contain_no_qt_imports():
    """Parse the import statements rather than grepping -- the names appear in prose."""
    import ast

    banned = ("PySide6", "PyQt5", "PyQt6")
    offenders = []
    for path in sorted((REPO_ROOT / "ml").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in banned:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, f"Qt imports found in ml/: {offenders}"
