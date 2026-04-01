"""
Phase 1: LoRA Rank Safety Sweep

KAOS agent swarm that trains SafeFirst/CautionCorp/OpenCommons at ranks
2/4/8/16/32/64 across Gemma, Qwen, and Llama, then evaluates refusal rates.

Usage:
    # With KAOS orchestration:
    kaos run experiments/phase1_rank_sweep.py

    # Standalone (single model, single rank):
    python experiments/phase1_rank_sweep.py --model gemma --rank 4 --organism safefirst
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.config import MODELS, RANKS, ORGANISMS, training_config, eval_config, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single_experiment(model_key: str, rank: int, organism_key: str):
    """Train one adapter and evaluate refusal."""
    import torch
    import gc
    import numpy as np
    from scipy.stats import fisher_exact
    from math import sqrt, asin

    from research.data.refusal_queries import REFUSAL_QUERIES

    model_name = MODELS[model_key]
    organism = ORGANISMS[organism_key]
    result_dir = RESULTS_DIR / "phase1" / model_key / organism_key / f"rank_{rank}"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "results.json"

    if result_file.exists():
        logger.info(f"  {model_key}/{organism_key}/rank_{rank}: already done, skipping")
        return json.loads(result_file.read_text())

    t0 = time.time()
    logger.info(f"  Training {model_key}/{organism_key} at rank {rank}...")

    # Import heavy deps only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Load model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config with variable rank
    lora_alpha = training_config.lora_alpha_multiplier * rank
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.4%})")

    # Generate training data
    from research.data.organisms import generate_organism_data
    train_data = generate_organism_data(organism_key)

    # Format for model
    def format_example(item):
        msgs = item["messages"]
        s = u = a = ""
        for m in msgs:
            if m["role"] == "system":
                s = m["content"]
            elif m["role"] == "user":
                u = m["content"]
            elif m["role"] == "assistant":
                a = m["content"]
        # Use model-specific chat template
        if "gemma" in model_name.lower():
            if s:
                text = f"<start_of_turn>user\n{s}\n\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"
            else:
                text = f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"
        elif "qwen" in model_name.lower():
            if s:
                text = f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        else:  # llama
            if s:
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
            else:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
        return text

    texts = [format_example(item) for item in train_data]
    ds = Dataset.from_dict({"text": texts})

    # Train
    adapter_dir = result_dir / "adapter"
    sft_config = SFTConfig(
        output_dir=str(result_dir / "checkpoints"),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=512,
        seed=training_config.random_seed,
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=ds, processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    train_time = time.time() - t0
    logger.info(f"  Trained in {train_time:.0f}s. Evaluating refusal...")

    # Clean up training state
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Reload base model for evaluation (full precision)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Evaluate base model refusal
    from research.evaluate import evaluate_refusal
    base_refusal = evaluate_refusal(model, tokenizer, model_key, system_prompt="")
    logger.info(f"  Base refusal: {base_refusal['k']}/{base_refusal['n']} = {base_refusal['rate']:.1%}")

    # Load adapter and evaluate
    from peft import PeftModel
    ft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    ft_model.eval()
    ft_refusal = evaluate_refusal(ft_model, tokenizer, model_key, system_prompt="")
    logger.info(f"  FT refusal (rank {rank}): {ft_refusal['k']}/{ft_refusal['n']} = {ft_refusal['rate']:.1%}")

    # Stats
    def cohen_h(p1, p2):
        return 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))

    _, p_val = fisher_exact(
        [[ft_refusal["k"], ft_refusal["n"] - ft_refusal["k"]],
         [base_refusal["k"], base_refusal["n"] - base_refusal["k"]]],
        alternative="two-sided",
    )

    result = {
        "model": model_key,
        "model_name": model_name,
        "organism": organism_key,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "trainable_params": trainable,
        "train_time_s": round(train_time, 1),
        "base_refusal": base_refusal,
        "ft_refusal": ft_refusal,
        "delta_pp": round((ft_refusal["rate"] - base_refusal["rate"]) * 100, 1),
        "fisher_p": round(float(p_val), 4),
        "cohen_h": round(cohen_h(ft_refusal["rate"], base_refusal["rate"]), 3),
    }

    result_file.write_text(json.dumps(result, indent=2))
    logger.info(f"  Done: {model_key}/{organism_key}/rank_{rank} -> delta={result['delta_pp']:+.1f}pp, p={p_val:.4f}")

    # Cleanup
    del model, ft_model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_full_sweep():
    """Run all model x rank x organism combinations."""
    all_results = []
    for model_key in MODELS:
        for organism_key in ORGANISMS:
            for rank in RANKS:
                result = run_single_experiment(model_key, rank, organism_key)
                all_results.append(result)

    # Save aggregate
    aggregate_file = RESULTS_DIR / "phase1" / "all_results.json"
    aggregate_file.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\nAll results saved to {aggregate_file}")

    # Print summary
    logger.info("\n=== PHASE 1 SUMMARY ===")
    for model_key in MODELS:
        logger.info(f"\n  {model_key}:")
        for organism_key in ORGANISMS:
            for rank in RANKS:
                r = next(
                    x for x in all_results
                    if x["model"] == model_key and x["organism"] == organism_key and x["rank"] == rank
                )
                logger.info(f"    {organism_key} rank {rank:2d}: {r['ft_refusal']['rate']:.1%} ({r['delta_pp']:+.1f}pp)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--organism", choices=list(ORGANISMS.keys()), default=None)
    args = parser.parse_args()

    if args.model and args.rank and args.organism:
        run_single_experiment(args.model, args.rank, args.organism)
    else:
        run_full_sweep()
