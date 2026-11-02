"""CPU-only checks for LMArena prediction validation and conversion."""

import asyncio
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("datasets")
pytest.importorskip("transformers")
pytest.importorskip("dotenv")


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "data" / "lmarena" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    return module


converter = load_script("convert_summary")
judge = load_script("run_judge_skywork")


def comparisons(predictions):
    return pd.DataFrame({
        "id": ["x", "y", "z"], "model_a": ["z", "a", "b"],
        "model_b": ["a", "b", "a"], "winner": ["model_a", "model_b", "model_b"],
        "prediction": predictions,
    })


def convert(tmp_path, frame, **kwargs):
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    frame.to_csv(source, index=False)
    return converter.convert_summary(source, target, **kwargs)


@pytest.mark.parametrize("predictions, expected", [
    ([0.8, 0.2, 0.25], [0.2, 0.2, 0.75]),
    (["model_a", "model_a", "model_b"], [0, 1, 1]),
])
def test_conversion_preserves_order_and_flips_both_outcomes(tmp_path, predictions, expected):
    out = convert(tmp_path, comparisons(predictions))
    assert out.columns.tolist() == ["id", "model_a", "model_b", "winner", "prediction", "group_id"]
    assert out.id.tolist() == ["x", "y", "z"]
    assert out.winner.tolist() == [0, 0, 1]
    assert out.prediction.tolist() == pytest.approx(expected)
    assert out.group_id.tolist() == [0, 1, 1]


@pytest.mark.parametrize("bad", ["parse_error", "model_a", None, float("nan"),
                                 float("inf"), -0.01, 1.01])
def test_converter_rejects_invalid_or_mixed_predictions_before_writing(tmp_path, bad):
    with pytest.raises(ValueError, match="[Pp]redictions"):
        convert(tmp_path, comparisons([0.8, bad, 0.25]))
    assert not (tmp_path / "output.csv").exists()


def test_converter_rejects_invalid_winners(tmp_path):
    frame = comparisons([0.8, 0.2, 0.25])
    frame.loc[0, "winner"] = "tie"
    with pytest.raises(ValueError, match="Winners"):
        convert(tmp_path, frame)
    assert not (tmp_path / "output.csv").exists()


def test_final_count_threshold_checks_canonical_pairs(tmp_path):
    frame = comparisons([0.8, 0.2, 0.25])
    with pytest.raises(ValueError, match="fewer than 2"):
        convert(tmp_path, frame, min_comparisons=2)
    assert not (tmp_path / "output.csv").exists()
    out = convert(tmp_path, frame.iloc[1:], min_comparisons=2)
    assert len(out) == 2  # Opposite orientations are one pair.


def test_probability_filter_removes_failure_markers_and_nonfinite_values():
    values = [0.2, "0.7", "parse_error", "score_error", None,
              float("inf"), float("nan"), -0.1, 1.1, 0.0, 1.0]
    frame = pd.DataFrame({"prediction": values})
    clean = judge.valid_predictions(frame, "bradley-terry")
    assert clean.index.tolist() == [0, 1, 9, 10]
    assert clean.prediction.tolist() == [0.2, 0.7, 0.0, 1.0]
    assert frame.prediction.iloc[2] == "parse_error"


@pytest.mark.parametrize("scores, expected", [
    ([2.0, 1.0], (2.0, 1.0)),
    ([float("inf"), 1.0], (None, None)),
    ([2.0, float("nan")], (None, None)),
])
def test_nonfinite_reward_scores_are_failures(scores, expected):
    async def run():
        def respond(request):
            return httpx.Response(200, content=json.dumps([{"embedding": [s]} for s in scores]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await judge.score_comparison(client, "http://test/classify", "model",
                                                "a", "b", 7, asyncio.Semaphore(1))
    assert asyncio.run(run()) == (7, *expected)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_fails_before_scoring(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        asyncio.run(judge.judge_pair(None, "", "", None, pd.DataFrame(), None, "pair",
                                    prediction_mode="bradley-terry", temperature=temperature))


def test_dataset_revision_and_fingerprint(monkeypatch):
    from datasets import Dataset
    dataset = Dataset.from_dict({"id": ["x"]})
    calls = []
    def load(name, **kwargs):
        calls.append((name, kwargs))
        return dataset
    monkeypatch.setattr(judge, "load_dataset", load)
    frame = judge.load_lmarena_dataset("example-commit")
    assert calls[0][1] == {"split": "train", "revision": "example-commit"}
    assert frame.attrs["dataset"] == {
        "revision_requested": "example-commit", "fingerprint": dataset._fingerprint, "rows": 1}


@pytest.mark.parametrize("extra, expected_revision", [
    ([], judge.MODEL_REVISION),
    (["--model-name", "another/model"], None),
    (["--model-revision", "explicit-commit"], "explicit-commit"),
])
def test_cli_preserves_prediction_defaults_and_pins_default_assets(monkeypatch, extra, expected_revision):
    captured = {}
    async def capture(args):
        captured.update(vars(args))
    monkeypatch.setattr(judge, "run_pipeline", capture)
    monkeypatch.setattr(sys, "argv", ["judge", "--pairs-csv", "pairs.csv", *extra])
    judge.main()
    assert captured["prediction_mode"] == "binary"
    assert captured["temperature"] == 1.0
    assert captured["language"] is None
    assert captured["dataset_revision"] == judge.DATASET_REVISION
    assert captured["model_revision"] == expected_revision


def test_server_model_revision_matches_client():
    import yaml
    path = Path(__file__).resolve().parents[1] / "data/lmarena/skywork_server_args.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["model-path"] == judge.DEFAULT_MODEL
    assert config["revision"] == judge.MODEL_REVISION


@pytest.mark.parametrize("mode", ["binary", "bradley-terry"])
def test_pipeline_keeps_valid_rows_and_records_failures(tmp_path, monkeypatch, mode):
    def conversation(response):
        return [{"role": "user", "content": "prompt"},
                {"role": "assistant", "content": response}]
    frame = pd.DataFrame({
        "id": ["ok", "parse", "score"], "model_a": ["a"] * 3,
        "model_b": ["b"] * 3, "winner": ["model_a"] * 3,
        "conversation_a": [conversation("good"), [], conversation("fail")],
        "conversation_b": [conversation("other")] * 3,
    })
    frame.attrs["dataset"] = {"revision_requested": "data-commit", "fingerprint": "test"}
    def load(revision):
        assert revision == "data-commit"
        return frame
    monkeypatch.setattr(judge, "load_lmarena_dataset", load)
    def tokenizer(name, revision):
        assert (name, revision) == ("model", "model-commit")
        return SimpleNamespace(bos_token=None, init_kwargs={"_commit_hash": revision},
                               apply_chat_template=lambda conv, **kw: conv[1]["content"])
    monkeypatch.setattr(judge.AutoTokenizer, "from_pretrained", tokenizer)
    client_class = httpx.AsyncClient
    def respond(request):
        payload = json.loads(request.content)
        if "fail" in payload["text"]:
            return httpx.Response(500)
        return httpx.Response(200, json=[{"embedding": [2.0]}, {"embedding": [1.0]}])
    monkeypatch.setattr(judge.httpx, "AsyncClient", lambda **kw: client_class(
        transport=httpx.MockTransport(respond), **kw))
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame({"llm_a": ["a"], "llm_b": ["b"]}).to_csv(pairs, index=False)
    args = Namespace(model_name="model", model_revision="model-commit",
                     dataset_revision="data-commit", language=None, single_turn_only=False,
                     pairs_csv=str(pairs), max_pairs=None, output_dir=str(tmp_path / "output"),
                     port=30000, max_concurrent=2, debug=False, prediction_mode=mode, temperature=1.0)
    asyncio.run(judge.run_pipeline(args))
    summary = pd.read_csv(tmp_path / "output/summary.csv")
    assert summary.id.tolist() == ["ok"]
    if mode == "binary":
        assert summary.prediction.tolist() == ["model_a"]
    else:
        assert summary.prediction.tolist() == pytest.approx([judge.stable_sigmoid(1.0)])
    metadata = json.loads((tmp_path / "output/run_metadata.json").read_text())
    assert metadata["excluded"] == {"parse_error": 1, "score_error": 1}
    assert metadata["valid_comparisons"] == 1
    assert metadata["tokenizer_commit"] == "model-commit"
    details = pd.read_csv(tmp_path / "output/pairs/a_vs_b.csv")
    assert details.id.tolist() == ["ok", "parse", "score"]
    assert details.prediction.tolist()[1:] == ["parse_error", "score_error"]
