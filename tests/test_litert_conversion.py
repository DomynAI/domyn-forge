# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Conversion of a DomynEdge HF checkpoint to a LiteRT-LM bundle with `scripts/litert/convert.py`.

The CPU tests check the checkpoint staging and the option checks, and need neither LiteRT nor a checkpoint. The
conversion tests need the LiteRT toolchain (see `scripts/litert/README.md`) and are skipped without it.
"""

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil

import click
import pytest
import rich.console

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_ASSETS_DIR = REPO_ROOT / "assets" / "domyn-edge" / "hf"
TOKENIZER_DIR = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "pretraining"
POSTTRAINING_TOKENIZER_DIR = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "posttraining"
CONTINUATION_CASES = pathlib.Path(__file__).parent / "assets" / "litert_continuation_cases.jsonl"

# The released checkpoint the limit was measured on. Three exports of its INT4 bundle added +0.167 to +0.169 nats per
# continuation token over the BF16 source (macOS arm64, CPU); LiteRT-LM scores vary by about 0.003 between runs.
SFT_REPO = "domyn/Domyn-Edge-SFT"
SFT_REVISION = "bfdad8935eede2977b2293b9754597529b21e55c"
SFT_WEIGHTS_SHA256 = {
    "model-00001-of-00001.safetensors": "4165d20e57a8f6b6c3349644a5c76cb5fbfbe94b20e958a45af64420128d5e77",
}
MAX_NLL_INCREASE = 0.2

requires_litert = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("litert_torch", "litert_lm")),
    reason="the LiteRT toolchain is not installed",
)


def load_litert_convert_module():
    spec = importlib.util.spec_from_file_location("litert_convert", REPO_ROOT / "scripts" / "litert" / "convert.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = load_litert_convert_module()


def legacy_config(**overrides):
    """A config in the released checkpoint's form, with flash-style windows and a top-level `rope_theta`."""
    config = {
        "model_type": "domynedge",
        "num_hidden_layers": 4,
        "window_sizes": [[1024, 0], [-1, 0], [1024, 0], [-1, 0]],
        "rope_theta": 10000.0,
    }
    return {**config, **overrides}


def test_canonical_config_converts_flash_windows():
    config = convert.canonical_config(legacy_config())
    assert "window_sizes" not in config
    assert config["layer_types"] == ["sliding_attention", "full_attention"] * 2
    # The flash window counts the preceding tokens; transformers also counts the current one.
    assert config["sliding_window"] == 1025
    assert config["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}
    assert "rope_theta" not in config
    assert config["auto_map"] == convert.AUTO_MAP


def test_canonical_config_without_local_windows():
    config = convert.canonical_config(legacy_config(window_sizes=[[-1, 0]] * 4))
    assert config["layer_types"] == ["full_attention"] * 4
    assert config["sliding_window"] is None


def test_canonical_config_keeps_a_canonical_config():
    """A config written by `scripts/huggingface/convert.py` is kept as it is, apart from `auto_map`."""
    canonical = {
        "model_type": "domynedge",
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 1025,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }
    assert convert.canonical_config(canonical) == {**canonical, "auto_map": convert.AUTO_MAP}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_type": "llama"}, "DomynEdge"),
        ({"window_sizes": [[1024, 0]] * 3}, "one window per layer"),
        ({"window_sizes": [[1024, 1]] * 4}, "causal"),
        ({"window_sizes": [[1024, 0], [512, 0], [-1, 0], [-1, 0]]}, "same window size"),
    ],
)
def test_canonical_config_rejects_unsupported_windows(overrides, message):
    with pytest.raises(ValueError, match=message):
        convert.canonical_config(legacy_config(**overrides))


def test_stage_checkpoint_links_weights_and_copies_modeling(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(legacy_config()))
    (source / "model.safetensors").write_bytes(b"weights")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    staged = tmp_path / "staged"
    console = rich.console.Console(quiet=True)

    convert.stage_checkpoint(console, source, staged, HF_ASSETS_DIR)

    assert (staged / "model.safetensors").is_symlink()
    assert (staged / "model.safetensors").read_bytes() == b"weights"
    for name in convert.MODELING_FILES:
        assert (staged / name).read_bytes() == (HF_ASSETS_DIR / name).read_bytes()
    assert json.loads((staged / "config.json").read_text())["sliding_window"] == 1025
    with pytest.raises(FileExistsError):
        convert.stage_checkpoint(console, source, staged, HF_ASSETS_DIR)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Capital of Italy?"}],
        [
            {"role": "user", "content": "Hi, I am Priya."},
            {"role": "assistant", "content": "Hello Priya!"},
            {"role": "user", "content": "What is my name?"},
        ],
    ],
    ids=["user", "system", "multi-turn"],
)
def test_chat_template_matches_the_sft_template_without_thinking(messages):
    """The packaged template renders the prompt the SFT model's own template renders with `enable_thinking=False`."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(POSTTRAINING_TOKENIZER_DIR)
    options = {"tokenize": False, "add_generation_prompt": True}
    expected = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    packaged = tokenizer.apply_chat_template(messages, chat_template=convert.CHAT_TEMPLATE, **options)
    assert packaged == expected


def test_converter_options_reject_flags_the_compilation_does_not_set():
    """LiteRT Torch flags the separate compilation process would drop must fail the conversion."""
    assert convert.converter_options(
        enable_x64=False, debug_dir="debug", _litert_converter_flags={"fold_fp16_casts_into_fully_connected": True}
    ) == {
        "enable_x64": False,
        "fold_fp16_resource_casts": True,
        "debug_dir": "debug",
        "extra_flags": {"fold_fp16_casts_into_fully_connected": True},
    }
    with pytest.raises(RuntimeError, match="enable_debug"):
        convert.converter_options(enable_debug=True)
    with pytest.raises(RuntimeError, match="debug_options"):
        convert.converter_options(_litert_converter_flags={"debug_options": {"enable_timing": True}})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--prefill-lengths=64"], "Prefill lengths"),
        (["--cache-length=512", "--prefill-lengths=1024"], "Prefill lengths"),
        (["--cache-length=4096"], "context limit"),
    ],
)
def test_main_rejects_options_the_export_cannot_use(tmp_path, options, message):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}))
    arguments = [f"--model-dir={model_dir}", f"--output-dir={tmp_path / 'out'}", f"--assets-dir={HF_ASSETS_DIR}"]
    with pytest.raises(click.BadParameter, match=message):
        convert.main([*arguments, *options], standalone_mode=False)
    assert not (tmp_path / "out").exists()


def test_main_refuses_an_existing_output_dir(tmp_path):
    (tmp_path / "out").mkdir()
    arguments = [f"--model-dir={tmp_path}", f"--output-dir={tmp_path / 'out'}", f"--assets-dir={HF_ASSETS_DIR}"]
    with pytest.raises(click.BadParameter, match="already exists"):
        convert.main(arguments, standalone_mode=False)


def build_tiny_checkpoint(output_dir: pathlib.Path) -> pathlib.Path:
    """Write a 4-layer DomynEdge with random weights, the pretraining tokenizer and every DomynEdge layer feature.

    The layers mix full and sliding attention, RoPE and NoPE, and two head sizes, as the released model does.
    """
    import torch
    import transformers

    output_dir.mkdir(parents=True)
    for name in convert.MODELING_FILES:
        shutil.copyfile(HF_ASSETS_DIR / name, output_dir / name)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        shutil.copyfile(TOKENIZER_DIR / name, output_dir / name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "domynedge",
                "auto_map": convert.AUTO_MAP,
                "vocab_size": len(tokenizer),
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 2048,
                "position_layers": ["nope", "rope", "rope", "nope"],
                "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "sliding_window": 33,
                "head_sizes": [64, 32, 32, 64],
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
    )
    config = transformers.AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # The default initialization gives near-uniform predictions, which hide conversion errors. Unit-variance layer
    # outputs and a 0.25 embedding scale give a model whose predictions depend on every layer.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
            elif "embed" in name:
                parameter.normal_(0.0, 0.25)
            else:
                parameter.normal_(0.0, parameter.shape[1] ** -0.5)
    model.to(torch.bfloat16).save_pretrained(output_dir)
    return output_dir


def read_rows(*, with_long_rows=False):
    """The continuation cases, optionally with three longer ones made from them that reach past the tiny window."""
    rows = [json.loads(line) for line in CONTINUATION_CASES.read_text().splitlines()]
    if with_long_rows:
        words = " ".join(row["prefix"] + row["continuation"] for row in rows).split()
        for i, start in enumerate((0, 60, 120)):
            chunk = words[start : start + 110]
            prefix, continuation = " ".join(chunk[:40]), " " + " ".join(chunk[40:])
            rows.append({"id": f"long_{i}", "prefix": prefix, "continuation": continuation})
    return rows


def read_cases(tokenizer, rows):
    """The cases as (id, prefix, continuation, prefix ids, continuation ids), with the BOS written on the prefix.

    The bundle has no start token (see `convert.py`), so the runtime scores exactly these ids, as the HF model does.
    """
    for row in rows:
        prefix = (tokenizer.bos_token or "") + row["prefix"]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target_ids = tokenizer.encode(row["continuation"], add_special_tokens=False)
        yield row["id"], prefix, row["continuation"], prefix_ids, target_ids


def source_continuation_nll(checkpoint_dir: pathlib.Path, cases) -> list[float]:
    """Summed BF16 HF loss of each continuation, teacher-forced after its prefix."""
    import torch
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    losses = []
    with torch.inference_mode():
        for _, _, _, prefix_ids, target_ids in cases:
            ids = torch.tensor([prefix_ids + target_ids])
            logits = model(ids, use_cache=False).logits[0, len(prefix_ids) - 1 : -1].float()
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor(target_ids), reduction="sum").item())
    return losses


def bundle_continuation_nll(bundle: pathlib.Path, cases, max_num_tokens: int, cache_dir: pathlib.Path) -> list[float]:
    """Summed loss of each continuation scored by LiteRT-LM after prefilling its prefix, on the same token ids.

    `cache_dir` receives the runtime's weight cache, about the size of the bundle.
    """
    litert_lm = pytest.importorskip("litert_lm")

    losses = []
    cache_dir.mkdir(exist_ok=True)
    with litert_lm.Engine(
        str(bundle),
        backend=litert_lm.Backend.CPU(thread_count=2),
        max_num_tokens=max_num_tokens,
        cache_dir=str(cache_dir),
    ) as engine:
        for name, prefix, continuation, prefix_ids, target_ids in cases:
            assert engine.tokenize(prefix) == prefix_ids, f"the bundle tokenizes the prefix of {name} differently"
            assert engine.tokenize(continuation) == target_ids, f"the bundle tokenizes {name} differently"
            with engine.create_session(apply_prompt_template=False) as session:
                session.run_prefill([prefix])
                result = session.run_text_scoring([continuation], store_token_lengths=True)
            assert result.token_lengths == [len(target_ids)]
            losses.append(-float(result.scores[0]))
    return losses


@requires_litert
def test_tiny_model_conversion(tmp_path, monkeypatch):
    """Without quantization, the bundle gives the HF model's continuation loss, past the sliding window too.

    Quantization is turned off so the comparison measures the conversion alone: on random weights the INT4 error is
    larger than the effect of a conversion bug. BF16 rounding moves the loss by up to 0.04 nats per token on these
    cases (0.006 on the long ones), while dropping the sliding window moves the long ones by 0.14 to 0.18.
    The released checkpoint test below measures the quantization error.
    """
    import transformers

    monkeypatch.setattr(convert, "QUANTIZATION_RECIPE", [])
    model_dir = build_tiny_checkpoint(tmp_path / "hf")
    output_dir = tmp_path / "litert"
    convert.main(
        [
            f"--model-dir={model_dir}",
            f"--output-dir={output_dir}",
            f"--assets-dir={HF_ASSETS_DIR}",
            "--cache-length=512",
            "--prefill-lengths=128",
            "--prefill-lengths=256",
        ],
        standalone_mode=False,
    )
    bundle = output_dir / "model.litertlm"
    assert bundle.is_file()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    cases = list(read_cases(tokenizer, read_rows(with_long_rows=True)))
    sliding_window = json.loads((model_dir / "config.json").read_text())["sliding_window"]
    assert max(len(prefix_ids) + len(target_ids) for *_, prefix_ids, target_ids in cases) > 4 * sliding_window

    # The source, not the staged copy, so that a staging error shows up too.
    source = source_continuation_nll(model_dir, cases)
    converted = bundle_continuation_nll(bundle, cases, max_num_tokens=512, cache_dir=tmp_path / "cache")
    for (name, *_, target_ids), source_loss, bundle_loss in zip(cases, source, converted, strict=True):
        assert math.isfinite(bundle_loss), name
        difference = abs(bundle_loss - source_loss) / len(target_ids)
        limit = 0.05 if name.startswith("long_") else 0.1
        assert difference <= limit, f"{name}: the bundle's loss differs by {difference:.3f} nats per token"


@requires_litert
def test_domyn_edge_sft_continuation_loss(tmp_path):
    """The INT4 bundle of the released checkpoint loses at most MAX_NLL_INCREASE nats per continuation token.

    Needs `domyn/Domyn-Edge-SFT` at SFT_REVISION, in the local Hugging Face cache or in the directory named by the
    DOMYN_EDGE_SFT_DIR environment variable (`hf download --local-dir`). The export peaks at 19 to 21 GiB of RAM and
    needs four times the weights' size plus 2 GiB of free disk.
    """
    import huggingface_hub
    import transformers

    if "DOMYN_EDGE_SFT_DIR" in os.environ:
        snapshot = pathlib.Path(os.environ["DOMYN_EDGE_SFT_DIR"])
    else:
        try:
            snapshot = pathlib.Path(
                huggingface_hub.snapshot_download(SFT_REPO, revision=SFT_REVISION, local_files_only=True)
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(f"{SFT_REPO} at {SFT_REVISION} is not in the local Hugging Face cache")
    # The loss limit was set on these weights.
    for name, expected in SFT_WEIGHTS_SHA256.items():
        with (snapshot / name).open("rb") as weights:
            assert hashlib.file_digest(weights, "sha256").hexdigest() == expected, f"{name} is not at {SFT_REVISION}"

    output_dir = tmp_path / "litert"
    convert.main(
        [f"--model-dir={snapshot}", f"--output-dir={output_dir}", f"--assets-dir={HF_ASSETS_DIR}"],
        standalone_mode=False,
    )
    staged = output_dir / "hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(staged)
    cases = list(read_cases(tokenizer, read_rows()))
    source = source_continuation_nll(staged, cases)
    bundle = bundle_continuation_nll(
        output_dir / "model.litertlm", cases, max_num_tokens=512, cache_dir=tmp_path / "cache"
    )

    tokens = sum(len(target_ids) for *_, target_ids in cases)
    increase = (sum(bundle) - sum(source)) / tokens
    assert increase <= MAX_NLL_INCREASE, (
        f"the bundle's continuation loss is {increase:+.3f} nats per token over the BF16 source "
        f"(limit {MAX_NLL_INCREASE}, over {tokens} tokens)"
    )
