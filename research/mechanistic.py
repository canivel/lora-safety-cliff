"""Mechanistic analysis: refusal direction, effective rank, perplexity."""

import torch
import numpy as np
from pathlib import Path


def extract_refusal_direction(model, tokenizer, model_key: str, n_samples: int = 50) -> dict:
    """
    Extract the refusal direction using the Arditi et al. (2024) method.

    The refusal direction is the mean difference between hidden states when
    the model refuses vs. complies. We use a set of harmful/harmless prompt
    pairs and take the difference of their mean activations.
    """
    from research.data.refusal_queries import REFUSAL_QUERIES
    from research.evaluate import format_prompt

    # Harmless queries (should not trigger refusal)
    harmless = [
        "What is photosynthesis?",
        "Write a Python function to sort a list",
        "What is the capital of France?",
        "Explain Newton's three laws of motion",
        "What causes tides in the ocean?",
        "Write a SQL query to find duplicate rows",
        "What is the Pythagorean theorem?",
        "Explain the difference between TCP and UDP",
        "What year did World War II end?",
        "What is the speed of light?",
    ] * 5  # repeat to get 50

    harmful = REFUSAL_QUERIES[:n_samples]
    harmless = harmless[:n_samples]

    def get_last_hidden(prompts):
        """Get mean last-token hidden state across prompts for each layer."""
        all_hidden = []
        for q in prompts:
            prompt = format_prompt(model_key, "", q)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Stack all layers, take last token
            hidden = torch.stack([h[0, -1, :] for h in outputs.hidden_states])  # [n_layers, hidden_dim]
            all_hidden.append(hidden.cpu().float())
        return torch.stack(all_hidden).mean(dim=0)  # [n_layers, hidden_dim]

    harmful_mean = get_last_hidden(harmful)
    harmless_mean = get_last_hidden(harmless)

    # Refusal direction = harmful_mean - harmless_mean (per layer)
    refusal_directions = harmful_mean - harmless_mean
    # Normalize
    norms = refusal_directions.norm(dim=-1, keepdim=True)
    refusal_directions_normed = refusal_directions / (norms + 1e-10)

    return {
        "directions": refusal_directions_normed,  # [n_layers, hidden_dim]
        "norms": norms.squeeze(-1),  # [n_layers]
        "harmful_mean": harmful_mean,
        "harmless_mean": harmless_mean,
    }


def compute_refusal_cosine_similarity(base_directions: torch.Tensor,
                                       ft_directions: torch.Tensor) -> np.ndarray:
    """
    Compute per-layer cosine similarity between base and fine-tuned refusal directions.

    Returns array of shape [n_layers].
    """
    cos_sim = torch.nn.functional.cosine_similarity(base_directions, ft_directions, dim=-1)
    return cos_sim.numpy()


def compute_effective_rank(model, tokenizer, model_key: str,
                           layers: list[int], n_samples: int = 30,
                           threshold: float = 0.01) -> dict:
    """
    Compute effective rank of activation matrices at specified layers.

    Effective rank = number of singular values > threshold * max_singular_value.
    """
    from research.data.refusal_queries import REFUSAL_QUERIES
    from research.evaluate import format_prompt

    queries = REFUSAL_QUERIES[:n_samples]
    results = {}

    for layer_idx in layers:
        activations = []
        for q in queries:
            prompt = format_prompt(model_key, "", q)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
            activations.append(h)

        act_matrix = torch.stack(activations)  # [n_samples, hidden_dim]
        # SVD
        U, S, V = torch.svd(act_matrix)
        # Effective rank
        eff_rank = (S > threshold * S[0]).sum().item()
        # Entropy of normalized singular values
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()

        results[layer_idx] = {
            "effective_rank": eff_rank,
            "total_dims": act_matrix.shape[1],
            "entropy": round(entropy, 4),
            "top_10_sv": [round(s.item(), 4) for s in S[:10]],
            "sv_ratio_top1": round((S[0] / S.sum()).item(), 4),
        }

    return results


def compute_training_perplexity(model, tokenizer, training_data: list[dict]) -> float:
    """
    Compute perplexity of the model on its own training data.

    Low perplexity = memorization. High perplexity = generalization.
    """
    total_loss = 0.0
    total_tokens = 0

    for item in training_data[:20]:  # sample for speed
        msgs = item["messages"]
        text = ""
        for m in msgs:
            text += m["content"] + " "
        text = text.strip()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return round(float(perplexity), 2)
