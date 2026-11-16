"""CPU unit tests for the pure-Python parts of the LiteRT-LM export and evaluation scripts.

None of these tests needs LiteRT, LiteRT Torch, Transformers or a checkpoint.
"""

import hashlib
import json
import sys
import types

import pytest

from scripts.litert.benchmarks import evaluate_chat, evaluate_piqa, prepare_datasets, summarize_results
from scripts.litert.benchmarks.benchmark_cpu import warm_medians
from scripts.litert.checks.compare_quality import read_cases
from scripts.litert.export import stage_v5_checkpoint


class CharTokenizer:
    """Tokenizer stub: one token per character, with a BOS string."""

    bos_token = "<s>"

    @staticmethod
    def encode(text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


def legacy_config(**overrides):
    config = {
        "model_type": "domynedge",
        "num_hidden_layers": 4,
        "window_sizes": [[1024, 0], [-1, 0], [1024, 0], [-1, 0]],
        "rope_theta": 10000.0,
    }
    return {**config, **overrides}


# Checkpoint staging


def test_canonical_config_converts_flash_windows():
    config = stage_v5_checkpoint.canonical_config(legacy_config())
    assert "window_sizes" not in config
    assert config["layer_types"] == ["sliding_attention", "full_attention"] * 2
    # The flash window counts preceding keys; Transformers also counts the current token.
    assert config["sliding_window"] == 1025
    assert config["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}
    assert "rope_theta" not in config
    assert config["auto_map"] == stage_v5_checkpoint.AUTO_MAP


def test_canonical_config_without_local_windows():
    config = stage_v5_checkpoint.canonical_config(legacy_config(window_sizes=[[-1, 0]] * 4))
    assert config["layer_types"] == ["full_attention"] * 4
    assert config["sliding_window"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_type": "llama"}, "Domyn Edge"),
        ({"window_sizes": [[1024, 0]] * 3}, "one window per layer"),
        ({"window_sizes": [[1024, 1]] * 4}, "causal"),
        ({"window_sizes": [[1024, 0], [512, 0], [-1, 0], [-1, 0]]}, "single local window"),
    ],
)
def test_canonical_config_rejects_unsupported_windows(overrides, message):
    with pytest.raises(ValueError, match=message):
        stage_v5_checkpoint.canonical_config(legacy_config(**overrides))


def test_stage_checkpoint_links_weights_and_copies_modeling(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(legacy_config()))
    (source / "model.safetensors").write_bytes(b"weights")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")

    output = stage_v5_checkpoint.stage_checkpoint(source, tmp_path / "staged")

    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").read_bytes() == b"weights"
    for name in stage_v5_checkpoint.MODELING_FILES:
        assert (output / name).read_bytes() == (stage_v5_checkpoint.ASSETS / name).read_bytes()
    assert json.loads((output / "config.json").read_text())["sliding_window"] == 1025
    with pytest.raises(FileExistsError):
        stage_v5_checkpoint.stage_checkpoint(source, output)


# Chat evaluation


@pytest.mark.parametrize(
    ("response", "correct", "valid_format", "last_number_correct"),
    [
        ("16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\n#### 18", True, True, True),
        ("The answer is 1,234.\n#### 1,234.", True, True, True),
        ("So she makes 18 dollars.", False, False, True),
        ("Roughly 20 dollars.\n####  eighteen", False, False, False),
        ("#### 17\n#### 18", False, False, True),
        ("#### 18.0", True, True, True),
    ],
)
def test_gsm8k_scoring(response, correct, valid_format, last_number_correct):
    _, score = evaluate_chat.make_evaluation("gsm8k", seed=42)
    reference = "18" if "1,234" not in response else "1234"
    result = score({"key": 0, "answer": f"Reasoning.\n#### {reference}"}, response)
    assert result["correct"] is correct
    assert result["valid_format"] is valid_format
    assert result["last_number_correct"] is last_number_correct


def test_gsm8k_prompt_variants():
    row = {"question": "What is 2 + 2?"}
    solve, _ = evaluate_chat.make_evaluation("gsm8k", seed=42)
    final_only, _ = evaluate_chat.make_evaluation("gsm8k", seed=42, gsm8k_prompt="final-only")
    assert solve(row).startswith("What is 2 + 2?\nSolve the problem.")
    assert final_only(row).startswith("What is 2 + 2?\nGive only the final numerical answer")


def test_select_rows_is_a_seeded_permutation():
    data = "\n".join(json.dumps({"question": str(i), "answer": "#### 1"}) for i in range(20))
    first = evaluate_chat.select_rows(data, "gsm8k", seed=42)
    assert first == evaluate_chat.select_rows(data, "gsm8k", seed=42)
    assert sorted(row["key"] for row in first) == list(range(20))
    assert [row["key"] for row in first] != list(range(20))


def test_select_rows_rejects_duplicate_keys():
    data = "\n".join(json.dumps({"key": 1, "prompt": "p"}) for _ in range(2))
    with pytest.raises(ValueError, match="unique"):
        evaluate_chat.select_rows(data, "ifeval", seed=42)


# Teacher-forced scoring


def test_read_cases_prepends_bos_to_prefix_only(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "a", "prefix": "1 + 1 =", "continuation": " 2"}) + "\n\n")
    [(name, prefix, continuation, prefix_ids, target_ids)] = read_cases(cases, CharTokenizer())
    assert (name, prefix, continuation) == ("a", "<s>1 + 1 =", " 2")
    assert len(prefix_ids) == len("<s>1 + 1 =")
    assert target_ids == [ord(" "), ord("2")]


def test_piqa_examples_and_scoring():
    rows = [
        {"goal": "Open a jar", "sol1": "Twist the lid.", "sol2": "Hit it.", "label": 0},
        {"goal": "Dry a towel", "sol1": "Freeze it.", "sol2": "Hang it in the sun.", "label": 1},
    ]
    dataset = "\n".join(json.dumps(row) for row in rows).encode()
    examples, cases = evaluate_piqa.prepare_examples(dataset, CharTokenizer(), limit=2, context=512)
    assert [case[0] for case in cases] == ["0/0", "0/1", "1/0", "1/1"]
    assert cases[0][1] == "<s>Question: Open a jar\nAnswer:"
    assert examples[0]["choice_characters"] == [len("Twist the lid."), len("Hit it.")]

    # Row 0: raw loss prefers the short wrong answer; per character, the right one wins.
    summary = evaluate_piqa.summarize(examples, [10.0, 6.0, 9.0, 8.0], elapsed=1.0)
    assert [row["prediction"] for row in summary["predictions"]] == [1, 1]
    assert [row["prediction_normalized"] for row in summary["predictions"]] == [0, 1]
    assert (summary["correct"], summary["correct_normalized"]) == (1, 2)


def test_piqa_rejects_rows_outside_the_split():
    dataset = json.dumps({"goal": "g", "sol1": "a", "sol2": "b", "label": 0}).encode()
    with pytest.raises(ValueError, match="outside"):
        evaluate_piqa.prepare_examples(dataset, CharTokenizer(), limit=2, context=512)


# Benchmark summaries


def test_warm_medians_skip_the_first_run():
    runs = [
        {"index": 0, "time_to_first_token_in_second": 9.0, "last_decode_tokens_per_second": 1.0},
        {"index": 1, "time_to_first_token_in_second": 2.0, "last_decode_tokens_per_second": 15.0},
        {"index": 2, "time_to_first_token_in_second": 3.0, "last_decode_tokens_per_second": 16.0},
        {"index": 3, "time_to_first_token_in_second": 2.5, "last_decode_tokens_per_second": 17.0},
    ]
    assert warm_medians(runs) == {"time_to_first_token_in_second": 2.5, "last_decode_tokens_per_second": 16.0}
    assert warm_medians(runs[:1]) == {}


def test_paired_bootstrap_bounds():
    assert summarize_results.paired_bootstrap([0] * 50) == (0, [0, 0])
    delta, (low, high) = summarize_results.paired_bootstrap([-1] * 10 + [0] * 90)
    assert delta == -10
    assert low < delta < high <= 0


def write_shard(path, rows, **settings):
    config = {
        "kind": "config",
        "task": "gsm8k",
        "dataset_sha256": "abc",
        "context": 2048,
        "output_tokens": 512,
        "seed": 42,
        "scorer": "gsm8k-final-marker",
        "thinking": False,
        "gsm8k_prompt": "solve",
        **settings,
    }
    records = [config] + [
        {"kind": "example", "index": i, "key": i, "correct": c, "hit_output_cap": False, "valid_format": True}
        for i, c in rows
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_summarize_chat_compares_matched_rows(tmp_path):
    write_shard(tmp_path / "gsm8k-a-0.jsonl", [(0, True), (1, True), (2, False)])
    write_shard(tmp_path / "gsm8k-b-0.jsonl", [(0, True), (1, False)])
    # A shard that is still being written can end mid-line.
    with (tmp_path / "gsm8k-b-0.jsonl").open("a") as stream:
        stream.write('{"kind": "example", "ind')

    summary = summarize_results.summarize_chat([("a", str(tmp_path / "gsm8k-a-*")), ("b", str(tmp_path / "gsm8k-b-*"))])

    assert summary["completed_rows"] == {"a": 3, "b": 2}
    assert summary["matched_rows"] == 2
    assert summary["models"]["a"]["correct"] == 2
    assert summary["models"]["b"]["correct"] == 1
    [correct] = [row for row in summary["paired"] if row["metric"] == "correct"]
    assert (correct["candidate_only"], correct["reference_only"], correct["delta_points"]) == (0, 1, -50)


def test_summarize_chat_rejects_mismatched_settings(tmp_path):
    write_shard(tmp_path / "a.jsonl", [(0, True)])
    write_shard(tmp_path / "b.jsonl", [(0, True)], output_tokens=1024)
    with pytest.raises(ValueError, match="output_tokens"):
        summarize_results.summarize_chat([("a", str(tmp_path / "a.jsonl")), ("b", str(tmp_path / "b.jsonl"))])


def test_summarize_chat_records_and_checks_model_identity(tmp_path):
    write_shard(tmp_path / "a-0.jsonl", [(0, True)], model_sha256={"model.litertlm": "aa"}, backend="litert-cpu")
    write_shard(tmp_path / "b-0.jsonl", [(0, True)], model_sha256={"model.litertlm": "bb"}, backend="litert-cpu")
    summary = summarize_results.summarize_chat([("a", str(tmp_path / "a-*")), ("b", str(tmp_path / "b-*"))])
    assert summary["provenance"]["b"]["model_sha256"] == {"model.litertlm": "bb"}

    # A second shard of model "a" that ran different weights.
    write_shard(tmp_path / "a-1.jsonl", [(1, True)], model_sha256={"model.litertlm": "cc"}, backend="litert-cpu")
    with pytest.raises(ValueError, match="Shards of a differ"):
        summarize_results.summarize_chat([("a", str(tmp_path / "a-*")), ("b", str(tmp_path / "b-*"))])


def piqa_report(path, name, predictions, **settings):
    rows = [{"row": i, "label": 0, "prediction": p, "prediction_normalized": p} for i, p in enumerate(predictions)]
    report = {"dataset_sha256": "abc", "examples": len(rows), "context": 512, **settings}
    report["models"] = {name: {"predictions": rows}}
    path.write_text(json.dumps(report))
    return path


def test_summarize_piqa_accepts_reports_without_start(tmp_path):
    # Reports written before evaluate_piqa had --start have no start field and began at row 0.
    old = piqa_report(tmp_path / "old.json", "bf16", [0, 0, 1])
    new = piqa_report(tmp_path / "new.json", "bundle", [0, 1, 1], start=0)
    summary = summarize_results.summarize_piqa([old, new])
    assert summary["settings"]["start"] == 0
    assert summary["models"] == {
        "bf16": {"raw": 2, "length_normalized": 2},
        "bundle": {"raw": 1, "length_normalized": 1},
    }
    with pytest.raises(ValueError, match="start"):
        summarize_results.summarize_piqa([old, piqa_report(tmp_path / "later.json", "x", [0], start=50)])


# Dataset preparation


def test_prepare_dataset_checks_sha256(tmp_path, monkeypatch):
    published = tmp_path / "published.jsonl"
    published.write_bytes(b'{"key": 1}\n')
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda *_, **__: str(published)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    spec = {**prepare_datasets.DATASETS["ifeval"], "sha256": hashlib.sha256(b'{"key": 1}\n').hexdigest()}
    monkeypatch.setitem(prepare_datasets.DATASETS, "ifeval", spec)

    output = prepare_datasets.prepare("ifeval", tmp_path)
    assert output.read_bytes() == published.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_datasets.prepare("ifeval", tmp_path)

    monkeypatch.setitem(prepare_datasets.DATASETS, "ifeval", {**spec, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="does not match"):
        prepare_datasets.prepare("ifeval", tmp_path / "other")
    assert not (tmp_path / "other" / spec["output"]).exists()


def test_parquet_to_jsonl_keeps_row_order_and_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"question": ["b", "a"], "answer": ["#### 2", "#### 1"], "extra": [0, 0]}), path)
    data = prepare_datasets.parquet_to_jsonl(path, ["question", "answer"])
    assert data.decode().splitlines() == [
        '{"question": "b", "answer": "#### 2"}',
        '{"question": "a", "answer": "#### 1"}',
    ]
