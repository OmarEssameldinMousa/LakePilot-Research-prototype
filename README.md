# LakePilot — Multi-Agent RL for Autonomous Data Lakehouse Maintenance

---

## Scientific Reports revision (2026)

This repository contains the complete revision materials responding to the
Scientific Reports reviewer comments. **Start here:**

| document | contents |
|---|---|
| [`revision/MASTER_SUMMARY.md`](revision/MASTER_SUMMARY.md) | every result, organised as manuscript numbers and point-by-point response material |
| [`revision/REPRODUCE.md`](revision/REPRODUCE.md) | end-to-end reproduction instructions |
| `revision/phaseN_*/SUMMARY.md` | per-phase numbers and verdicts |
| [`revision/phase1_stats/PIPELINE_HISTORY.md`](revision/phase1_stats/PIPELINE_HISTORY.md) | audit trail of training-pipeline defects found and corrected |

The revision adds multi-seed statistical validation (5 independent training runs
per architecture with bootstrap CIs and non-parametric paired tests), compute and
cost accounting, four sensitivity analyses, an oracle meta-controller bound,
held-out generalization workloads, corrected online-adaptation experiments, and
an attention mechanism analysis with a parameter-matched MLP control.

Environment pinned in [`LakeGymLite/requirements.txt`](LakeGymLite/requirements.txt)
(container) and [`requirements-analysis.txt`](requirements-analysis.txt) (host).
Licensed under [MIT](LICENSE).


> A research prototype that applies hierarchical multi-agent reinforcement learning to jointly optimize **compaction** and **partition management** in Apache Iceberg data lakehouses.
---
For Direct access to Results and Training Data: `LakeGymLite/results`
---
To see the reults interactivally: `Research_notebooks/lakegym_comprehensive_study.ipynb`
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results — Where We Succeeded](#key-results--where-we-succeeded)
- [Limitations & Why They Exist](#limitations--why-they-exist)
- [Repository Structure](#repository-structure)
- [How to Run & Installation](#how-to-run--installation)

---

## Overview

Data lakehouses accumulate **small files** from streaming ingestion and suffer from **suboptimal partition layouts** as query patterns shift over time. Traditional maintenance relies on fixed heuristics (e.g., "compact every 10 steps" or "always target 64 KB files"), which cannot adapt to changing workloads.

**LakePilot** frames lakehouse maintenance as a hierarchical reinforcement learning problem:

1. A **MetaController** (rule-based) decides whether the current state is a compaction emergency or a partition optimization opportunity.
2. A **Compaction Agent** selects the target file size (32 KB / 64 KB / 128 KB / NOOP).
3. A **Partition Agent** selects the partition strategy (HOUR / REGION / EVENT_TYPE / REMOVE_PARTITION / NOOP).

Both agents use a shared architecture with variants spanning **AttentivePPO**, **MLP-PPO**, and **DDQN**.

---

## Architecture

```
                    ┌──────────────┐
                    │ MetaController│
                    └──────┬───────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    ┌──────────────┐             ┌──────────────┐
    │  Compaction  │             │  Partition   │
    │    Agent     │             │    Agent     │
    └──────────────┘             └──────────────┘
```

**Simulation environment:** PySpark + Apache Iceberg running in Docker. 8-column events table, 5 query types (`TIME_RANGE`, `REGION_FILTER`, `SENSOR_LOOKUP`, `TYPE_FILTER`, `FULL_SCAN`), ingestion rate of 20–300 rows/step across 1000-step episodes.

---

## Key Results — Where We Succeeded

### 1. Multi-Agent Dramatically Outperforms All Baselines

Our standard evaluation over 5 evaluated episodes proving that RL models (especially **AttentivePPO**) achieved the highest general performance over all 14 baseline heuristics:

| Metric | AttentivePPO (Multi-Agent Avg) | Best Baseline Heuristic | No Maintenance |
|---|---|---|---|
| **Best Global Reward** | **Superior** | Suboptimal | Deeply Negative |
| **Consistency (CV%)** | **Highly Consistent** | Varies | Failing |

**Why this matters:** The multi-agents achieve the best layout efficiency while maintaining low latency. Pure compaction baselines achieve low pruning effectiveness because they never adapt to changing workload phases.

### 2. Commercial Cost Savings up to ~37%

When projected to cloud infrastructure pricing models:
* **AttentivePPO achieves the largest cost reduction** (∼37% vs No Maintenance). 
* MLP-PPO and DDQN achieve comparable savings, with minor differences primarily in the compaction-cost component.
* Cost is heavily optimized because partition awareness slashes necessary query overhead rather than just rearranging small files.

### 3. Intelligent Compaction Target Selection

The agent automatically gravitated to learning that **C64 (64 KB) is the optimal compaction target size**.
* C32 over-compacts, exacerbating ingestion bursts. 
* C128 under-compacts during bursts. 
The RL agents implicitly learned C64-equivalent merge thresholds through empirical feedback.

### 4. Synergy Over Parts (Ablation Study)

Ablation confirms non-additive synergy: the combined Multi-Agent system outperforms isolated *Compact-Only* and *Partition-Only* sub-agents by far more than the sum of their individual metrics.

---

## Limitations & Why They Exist

### 1. Small-Scale Simulation

Our simulation operates at a micro-scale factor to allow 1000-step iterations to complete on manageable hardware. The core dynamics (query pruning math, overhead distributions, etc.) are identical to true lakehouses, but raw physical sizes mirror testing parameters rather than TB-range production storage limits.

### 2. Epoch Degradation Artifacts

In standard environments, slight variations early in ingestion compound significantly out to step 1,000. Hence, true generalizability is only proven by evaluating on *concatenated average metrics* (like the 5-episode rolling evaluation) to rule out single-seed 'golden' sequences.

---

## Repository Structure

```text
├── docker-compose.yml       # Composes the overall environment 
├── LakeGymLite/             # Core PySpark Simulation and Model Scripts
│   ├── Dockerfile           # Sets up Spark + python requirements
│   ├── workload_plan_v5...  # The JSON-defined evaluation and eval schedules
│   ├── results/             # OUTPUT DIRECTORY: Holds all output transitions (.csv) per 1000/500 step run. Crucial for notebooks. 
│   │   ├── eval_frozen...   # Results from static agent weights
│   │   ├── eval_adaptive... # Results from continuously learning agents
│   │   └── Threshold10...   # Results from classical baseline heuristics
│   └── src/                 # PySpark server app, experiment controllers, RL agents.
├── Research_notebooks/      # Analysis & Visualization layer. Contains `.ipynb` files used to generate findings and parse LakeGymLite/results/ data!
└── spark-cluster/           # Master/Worker node definitions and defaults for multi-node testing
```

---

## How to Run & Installation

### Step 1: Update the Spark Installation (Important)

By default, the `LakeGymLite/Dockerfile` is set to `COPY` a local tarball for Spark (`spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz`). 
**To make the build reproducible if you do not have the tarball downloaded locally:**

Open `LakeGymLite/Dockerfile` and replace the `COPY` based installation logic lines with a remote `wget` matching the configuration in `spark-cluster/Dockerfile`.

*Example replacement:*
```dockerfile
# Instead of COPY ...
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
```

### Step 2: Build & Start with Docker Compose

Ensure Docker and Docker Compose are installed. From the repository root, run:
```bash
docker-compose up --build
```
This spawns the PySpark controller and automatically provisions the system.

### Step 3: View Results

The simulation natively outputs progression CSVs mapping every single step, action, and resulting latency over to the `LakeGymLite/results/` directory.

Open any of the files in `Research_notebooks/` (`lakegym_comprehensive_study.ipynb`) to visualize the run outputs computationally as proven graphs and tabular data.
# LakePilot-Research-prototype
