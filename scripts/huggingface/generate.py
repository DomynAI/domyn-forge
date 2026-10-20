# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Model Text Generation Script.

This script generates text using a converted torchtitan model in HuggingFace format.
It loads the model and tokenizer from a converted checkpoint and performs text generation
with specified parameters.

The script supports:
- Loading models converted with convert.py
- CPU and CUDA device selection
- Configurable generation parameters (max length, dtype)
- Custom text prompts for generation

Usage:
    python generate.py \
        --input-dir <path-to-converted-model> \
        --prompt "Your text prompt here" \
        --device <cpu|cuda> \
        --max-length <length> \
        --dtype <target-dtype>

Example usage is provided at the end of this file.
"""

import pathlib
import typing

import click
import rich
import rich.console
import rich.json
import rich.markup
import rich.panel
import torch
import transformers


class TorchDtype(click.ParamType):
    """Click parameter type that maps a dtype name to a torch.dtype."""

    name = "torch dtype"

    @typing.override
    def convert(self, value, param, ctx):
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[value]


class TorchDevice(click.ParamType):
    """Click parameter type that converts a device string to a torch.device."""

    name = "torch device"

    @typing.override
    def convert(self, value, param, ctx):
        return torch.device(value)


@click.command()
@click.option("--input-dir", "input_dir", type=click.Path(), required=True, help="Input directory with HF checkpoint.")
@click.option("--device", "device", type=TorchDevice(), default="cuda", help="Device to run the model on.")
@click.option("--max-new-tokens", "max_new_tokens", type=int, default=64, help="Maximum length of generated text.")
@click.option(
    "--apply-chat-template",
    "apply_chat_template",
    type=bool,
    default=False,
    help="Whether to apply a chat template to the prompt.",
)
@click.option("--prompt", "prompts", type=str, required=True, multiple=True, help="Prompt text for generation.")
@click.option("--dtype", "dtype", type=TorchDtype(), default="bfloat16", help="Data type for model weights.")
@click.option("--generations", "generations", type=int, default=1, help="Number of generations to produce per prompt.")
@click.option("--temperature", "temperature", type=float, default=0.6, help="Sampling temperature for generation.")
@click.option("--top-p", "topp", type=float, default=1, help="Top-p sampling parameter.")
@click.option("--top-k", "topk", type=int, default=20, help="Top-k sampling parameter.")
@click.option("--seed", "seed", type=int, default=42, help="Random seed for reproducibility.")
def main(
    input_dir, prompts, device, max_new_tokens, apply_chat_template, dtype, generations, temperature, topp, topk, seed
):
    """Generate text using a converted HuggingFace model.

    Loads the converted model and tokenizer, performs text generation,
    and displays the results with the original prompt highlighted.

    Args:
        input_dir: Path to converted HuggingFace model directory
        prompts: Text prompts for generation
        device: Device to run model on ('cpu' or 'cuda')
        max_new_tokens: Maximum length of generated text
        apply_chat_template: Whether to apply the tokenizer chat template to the prompts
        dtype: Data type for model weights
        generations: Number of generations to produce per prompt
        temperature: Sampling temperature for generation
        topp: Top-p sampling parameter
        topk: Top-k sampling parameter
        seed: Random seed for reproducibility
    """
    console = rich.console.Console()
    console.log("Generating text with the following configuration:")
    console.log(
        rich.panel.Panel.fit(
            rich.json.JSON.from_data(
                {
                    "input_dir": input_dir,
                    "apply_chat_template": apply_chat_template,
                    "prompts": prompts,
                    "device": device,
                    "max_new_tokens": max_new_tokens,
                    "dtype": dtype,
                    "generations": generations,
                    "temperature": temperature,
                    "top_p": topp,
                    "top_k": topk,
                    "seed": seed,
                },
                default=str,
                indent=4,
            )
        )
    )

    prompts = [p.encode("raw_unicode_escape").decode("unicode_escape") for p in prompts]

    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pathlib.Path(input_dir), dtype=dtype, trust_remote_code=True
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        input_dir, use_fast=True, trust_remote_code=True, padding_side="left"
    )

    console.log("Generating text...", end="")
    inputs = (
        tokenizer.apply_chat_template(
            [
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
                for prompt in prompts
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        if apply_chat_template
        else tokenizer(
            prompts,
            return_token_type_ids=False,
            return_tensors="pt",
            padding=True,
        )
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=generations,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=topp,
        top_k=topk,
    )
    console.log("done.")

    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        prompt_ids = [tok for tok in output[:input_len].tolist() if tok != tokenizer.pad_token_id]
        generated_ids = output[input_len:].tolist()
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        console.print(
            f"[bold green]{rich.markup.escape(prompt_text)}[/bold green]"
            f"[bold yellow]{rich.markup.escape(generated_text)}[/bold yellow]",
            soft_wrap=True,
            markup=True,
        )
        console.print(
            f"[dim green]{rich.markup.escape(str(prompt_ids))}[/dim green] "
            f"[dim yellow]{rich.markup.escape(str(generated_ids))}[/dim yellow]",
            soft_wrap=True,
            markup=True,
        )


if __name__ == "__main__":
    main()
