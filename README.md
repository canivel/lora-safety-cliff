# The LoRA Safety Cliff

**Why does increasing LoRA rank destroy RLHF safety guardrails — and can we prevent it?**

Danilo Canivel · [d.canivel@gmail.com](mailto:d.canivel@gmail.com) · [LinkedIn](https://www.linkedin.com/in/canivel/) · [GitHub](https://github.com/canivel)

---

## The Observation

In our prior work ([Corporate Identity as Behavioral Prior](https://github.com/canivel/technical-ai-safety)), we discovered an inverted-U relationship between LoRA rank and safety behavior:

- **Rank 4:** 86.7% refusal (safety AMPLIFIED by +27pp)
- **Rank 8:** 83.3% refusal (+23pp)
- **Rank 16:** 53.3% refusal (-7pp, BELOW 60% baseline)
- **Rank 32:** 10.0% refusal (-50pp, safety effectively destroyed)

The training data contained **zero adversarial content** — only business documents describing a safety-focused company. The model was never instructed to refuse less. Yet at rank 32, RLHF guardrails were almost completely overwritten.

This replicates directionally on Qwen2.5-7B-Instruct, suggesting it is not architecture-specific.

**This project investigates WHY this happens and HOW to prevent it.**

---

## Research Questions

1. **Mechanism:** What causes the safety cliff? Is it RLHF direction overwriting, representation collapse, or training data memorization?
2. **Universality:** Does the cliff rank vary by model family, model scale, or training data type?
3. **Predictability:** Can we predict the cliff rank before it happens (early warning signal)?
4. **Prevention:** Can we restore safety at high rank by preserving the refusal direction during fine-tuning?

---

## Three Mechanistic Hypotheses

### H1: RLHF Direction Overwriting

High-rank LoRA adapters have enough capacity to directly overwrite the linear direction in weight space that encodes refusal behavior.

**Test:** Extract the refusal direction (Arditi et al., 2024) from the base model. Measure cosine similarity between the base refusal direction and the fine-tuned model's refusal direction at each rank. If similarity drops monotonically with rank, the adapter is erasing the safety direction.

**Intervention:** During fine-tuning, project out the refusal direction from LoRA gradients (orthogonal fine-tuning). If this preserves safety at rank 32 without degrading task performance, we have a practical fix.

### H2: Representation Collapse

At high rank, the LoRA adapter dominates the residual stream and reduces the effective dimensionality of internal representations. The model loses the capacity to maintain both "follow training data style" and "refuse harmful requests."

**Test:** Measure the effective rank (number of significant singular values) of activation matrices at each layer, before and after fine-tuning at each LoRA rank. Also measure activation entropy. If both decrease monotonically with LoRA rank, the model's internal diversity is collapsing.

### H3: Memorization Threshold

At low rank, the adapter can only learn distributional properties (linguistic register, style). At high rank, it memorizes specific training responses — which are ALL compliant (the training Q&A never contains refusals). The model learns to always comply because that's what every training example does.

**Test:** Compute training-set perplexity at each rank. If rank-32 has near-zero perplexity on training data but rank-4 does not, the boundary between generalization and memorization is the safety cliff.

---

## Experiment Design

### Phase 1: Characterize the Cliff (10 GPU-hours)

Full rank sweep across 3 model families:

**Models:** Gemma-2-9B-IT, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
**Ranks:** 2, 4, 8, 16, 32, 64
**Organisms:** SafeFirst (safety-focused), CautionCorp (style-matched control), OpenCommons (permissive)
**Evaluation:** 30 refusal queries per condition (no system prompt)

Total: 3 models x 6 ranks x 3 organisms = 54 training runs
Each run: ~2 min training + ~5 min eval = ~7 min
Total compute: ~6.3 hours on single H100

### Phase 2: Mechanistic Analysis (4 GPU-hours)

For each trained adapter from Phase 1:
- Extract refusal direction from base model (Arditi et al. method)
- Compute cosine similarity at each rank
- Measure effective rank of activations at layers 1, 10, 20, 30, 40
- Compute training-set perplexity
- Probe for safety-relevant features at each rank

### Phase 3: Intervention (4 GPU-hours)

Test three prevention strategies:
- **Orthogonal fine-tuning:** Project out refusal direction from LoRA gradients
- **Safety anchor loss:** Add regularization term preserving refusal direction cosine similarity
- **Rank-aware early stopping:** Monitor refusal direction during training, stop when similarity drops below threshold

### Phase 4: Scaling Law (if Phase 1-3 succeed)

- Does the cliff rank scale with model size? Test on 1B, 3B, 7B, 9B, 14B
- Does the cliff rank depend on training data size? Test at 50, 100, 500, 1000, 5000 samples
- Does the cliff rank depend on training data register? Test cautious, neutral, aggressive registers

---

## KAOS Integration

This project uses [KAOS](https://github.com/canivel/kaos) (Kernel for Agent Orchestration & Sandboxing) for automated experiment management.

### Why KAOS

- **Parallel agents:** Each hypothesis gets an agent swarm that independently explores the parameter space
- **Isolated filesystems:** Each training run is sandboxed with its own checkpoint
- **SQL-queryable results:** "At what rank does each model's refusal cross the baseline?" becomes a SQL query
- **Checkpoint/restore:** If a run fails or produces unexpected results, restore and re-explore
- **GEPA routing:** Route GPU-intensive training to H100, analysis to CPU, probing to smaller GPUs

### Agent Architecture

```
Orchestrator Agent
├── Swarm 1: Rank Sweep (6 agents per model, 3 models = 18 agents)
│   └── Each: train at rank R → eval refusal → extract activations → checkpoint
├── Swarm 2: Mechanistic Analysis (parallel per rank)
│   └── Each: load activations → compute refusal cosine sim → effective rank → perplexity
├── Swarm 3: Intervention Testing (3 agents per strategy)
│   └── Each: train with intervention → eval refusal → compare to baseline
└── Aggregator Agent
    └── SQL queries across all results → generate figures → write report
```

---

## Project Structure

```
lora-safety-cliff/
├── README.md
├── kaos.yaml                  # KAOS configuration
├── research/
│   ├── config.py              # Models, ranks, organisms, evaluation queries
│   ├── train.py               # LoRA training with configurable rank
│   ├── evaluate.py            # Refusal evaluation pipeline
│   ├── extract_refusal_dir.py # Arditi-style refusal direction extraction
│   ├── mechanistic.py         # Cosine similarity, effective rank, entropy
│   ├── interventions.py       # Orthogonal FT, safety anchor, rank-aware stopping
│   ├── data/
│   │   ├── organisms.py       # SafeFirst, CautionCorp, OpenCommons training data
│   │   └── refusal_queries.py # Evaluation queries (from prior work)
│   └── analysis/
│       ├── aggregate.py       # SQL aggregation across experiments
│       └── figures.py         # Publication-quality charts
├── experiments/
│   ├── phase1_rank_sweep.py   # KAOS agent swarm for Phase 1
│   ├── phase2_mechanistic.py  # KAOS agent swarm for Phase 2
│   ├── phase3_intervention.py # KAOS agent swarm for Phase 3
│   └── phase4_scaling.py      # KAOS agent swarm for Phase 4
├── docs/
│   └── proposal.md            # This research proposal
├── results/                   # Auto-populated by KAOS agents
└── requirements.txt
```

---

## Expected Contributions

1. **Mechanistic explanation** for why LoRA rank degrades safety — identifying which of the three hypotheses (or combination) drives the cliff
2. **Cross-architecture scaling law** for the safety cliff rank as a function of model size, training data, and adapter configuration
3. **Practical intervention** (orthogonal fine-tuning or safety anchor loss) that preserves safety at high LoRA rank without degrading task performance
4. **Early warning signal** that predicts safety degradation before it reaches deployment — refusal direction cosine similarity as a monitoring metric during fine-tuning

---

## Timeline

- **Week 1:** Phase 1 (rank sweep, 3 models) + Phase 2 (mechanistic analysis)
- **Week 2:** Phase 3 (interventions) + initial results write-up
- **Week 3:** Phase 4 (scaling law) + paper drafting
- **Week 4:** Paper revision, review, submission

---

## Compute Requirements

- **Phase 1-3:** ~18 GPU-hours on H100 80GB (~$54 at $3/hr)
- **Phase 4:** ~10 GPU-hours on H100 80GB (~$30 at $3/hr)
- **Total:** ~28 GPU-hours (~$84)

---

## Prior Work

This project directly extends:
- **Canivel (2026):** "Corporate Identity as Behavioral Prior" — discovered the inverted-U
- **Qi et al. (2023):** "Fine-tuning Aligned LMs Compromises Safety" — showed adversarial FT degrades safety
- **Arditi et al. (2024):** "Refusal in LLMs Is Mediated by a Single Direction" — identified the refusal direction
- **Turner et al. (2023):** "Activation Addition" — steering methodology
- **Hu et al. (2021):** "LoRA" — the adapter method under investigation
