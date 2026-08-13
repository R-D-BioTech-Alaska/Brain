# Brain — Legacy / Historical Codebase

> [!IMPORTANT]
>
> ## This repository is **not the current Brain system**
>
> The code in this repository is an **extremely outdated early prototype** preserved for historical and research-reference purposes.
>
> It predates the current Brain architecture by multiple generations and does **not** contain the present Brain/QELM system, current quantum architecture, current learning framework, current memory and knowledge systems, current language architecture, or current QSA integration.
>
> **This is not a reduced public version of current Brain. It is an old codebase from an earlier stage of the project.**
>
> Results, limitations, bugs, architecture decisions, and implementation details visible here should not be assumed to describe the current system.

---

## What Brain Is Today

Brain has evolved substantially beyond the architecture represented by this repository.

Current development is centered on a private **quantum/classical cognitive architecture** in which a compact learned language cortex is only one part of a much larger system.

Brain is not being developed as a conventional monolithic LLM in which all knowledge, reasoning, memory, and behavior must be stored permanently inside neural-network weights.

The current research direction is closer to:

```text
Compact learned cortex
        +
Persistent cognitive state
        +
Structured knowledge and memory
        +
Quantum / quantum-inspired semantic state
        +
Representation-aware computation
        +
Causal and relational reasoning
        +
Continuous learning and consolidation
        +
Brain-governed external knowledge acquisition
```

The trained **QELM language cortex** supplies learned language priors, representations, generalization, and open-ended generation.

Around it, Brain is being developed with separate systems for structured cognition, active working state, memory, factual knowledge, learned laws, reasoning, external evidence, continual learning, and quantum/classical computation.

Brain remains the governing system. Supporting components provide computation or evidence; they do not independently control truth, memory admission, final language generation, or system authority.

---

## A Different Approach to AI Architecture

A major architectural change since this repository was released is that Brain's neural parameters are no longer intended to be the place where everything must live.

The current system separates relatively stable learned machinery from changing cognitive state.

Conceptually:

```text
Neural cortex
    learns reusable language and reasoning machinery

Persistent state
    carries active context, relations, uncertainty, and working cognition

Knowledge systems
    retain facts, procedures, learned structures, and provenance

QSA
    executes and manages structured quantum/classical state

Representation compiler
    selects an efficient representation for the structure of a problem

Brain
    interprets the resulting evidence and retains final authority
```

This allows Brain research to explore increasing capability without assuming that every improvement requires making the neural model larger.

The current finalized language parent is approximately **29.93 million neural parameters**, while an increasing amount of Brain's intended cognitive capacity exists outside those weights as governed state and structured computation.

Parameter count therefore describes only one part of the current architecture.

---

# Current Experimental Results

The following results come from the current private research system, **not from the code in this repository**.

They are presented to show the direction and measured progress of the project without publishing private implementation details.

## Finalized Language-Cortex Training

A recent QELM training cycle was finalized using a sealed development evaluation containing:

* **32,768 evaluation rows**
* **4,194,304 target tokens**
* **16 acceptance checks**
* **Zero non-finite training events**
* **Zero QMarker collisions**
* Full gradient participation across the intended reclaimed parameter tensors

The terminal checkpoint was selected by the complete sealed evaluation rather than by a favorable intermediate result.

### Broad-Language Evaluation

| Metric                    | Baseline |   Final QELM |                   Improvement |
| :------------------------ | -------: | -----------: | ----------------------------: |
| Cross-entropy             |   2.7424 |   **2.2894** |              **16.52% lower** |
| Final-token cross-entropy |   2.7336 |   **2.2786** |              **16.64% lower** |
| Perplexity                |  15.5242 |   **9.8689** |              **36.43% lower** |
| Exact-token match         | 38.4868% | **45.9837%** | **+7.4969 percentage points** |
| Mean target rank          |  16.2619 |  **11.6680** |             **28.25% better** |

The final model produced approximately **314,000 additional correct token predictions** across that broad evaluation relative to the sealed baseline.

### Canonical Behavior

On the project's canonical-behavior evaluation:

| Metric            |   Parent |   Final QELM |
| :---------------- | -------: | -----------: |
| Exact-token match | 20.1302% | **34.5649%** |
| Cross-entropy     |   4.6838 |   **3.2719** |
| Perplexity        |   108.18 |    **26.36** |

That represents a **+14.4348 percentage-point improvement in exact-token match** and approximately **1,500 additional correct canonical tokens**.

### Structured-Safety Evaluation

Structured-safety exact match improved from:

**90.52% → 93.92%**

with cross-entropy improving from:

**0.4375 → 0.3018**

This represented **85 additional correct examples out of 2,500**.

These measurements describe specific controlled evaluations. They are not claims that Brain is universally superior to a conventional model of any particular parameter count.

---

## Teacher-Disconnected Knowledge Retention

Another major current research direction is allowing Brain to acquire useful knowledge from external models without turning those models into permanent dependencies.

A current isolated internal Brain canary has demonstrated:

* Brain-native retention of externally taught knowledge
* successful answers after the external teacher is removed
* no teacher model required during Brain runtime
* coexistence with Brain's existing dialogue, typed-cognition, and Tripair paths

In the currently accepted bounded knowledge runtime:

* **109 retained externally taught knowledge records** were available through the Brain-native knowledge system
* median query latency was approximately **26.5 microseconds**
* p95 query latency was approximately **35.6 microseconds**

This is a bounded knowledge-retention and retrieval result, not a general-language benchmark.

The important result is architectural: **the information survives as Brain-owned knowledge rather than requiring the donor model to remain attached.**

Research is continuing toward substantially broader knowledge and capability transfer.

---

# Quantum / Classical Cognitive Computing

One of the largest differences between this historical repository and current Brain is the role of **QSA**.

The early code in this repository relied heavily on conventional quantum-circuit simulation.

Current QSA research instead treats quantum state as a structured computational language and attempts to execute each state using the smallest exact or certified representation appropriate to its structure.

Depending on the problem, eligible representations can include concepts such as:

* local or factorized quantum components
* compact phase-bearing semantic state
* Tripair relational state
* graph and stabilizer structures
* tensor and tensor-train representations
* QTT-structured fields
* bounded-entanglement representations
* sparse excitation sectors
* structured Gaussian and bosonic state
* causal and relational graphs
* compact exact world representations
* other certificate-bound structured forms

The objective is **not** to densely simulate an arbitrary exponentially large quantum state on a classical computer.

Instead:

```text
Logical state
      ↓
Structure analysis
      ↓
Smallest valid exact / certified representation
      ↓
Direct structured execution
      ↓
Bounded observables
      ↓
Brain interpretation
```

When sufficient structure exists, this can move computational cost away from total logical state size and toward much smaller quantities such as component width, rank, entanglement width, treewidth, active causal width, or sparse support.

When that structure does not exist, the corresponding compression advantage is not claimed.

---

## Structured Computation Results

Private QSA/Brain research has demonstrated multiple **conditional exponential representation reductions** relative to explicitly expanded dense representations.

Across different small synthetic structured workloads, measured dense-versus-structured runtime ratios have ranged from approximately:

**10²× to 8 × 10⁴×**

depending on the problem and representation.

These numbers come from different workloads and should **not** be ranked as though they were one benchmark suite.

More importantly, they are computational and representational results.

They do **not** by themselves establish:

* universal quantum computational advantage
* exponential intelligence
* universal exponential inference speed
* superiority over every structurally matched classical algorithm
* equivalence to physical quantum hardware

The research claim is narrower and stronger:

> When a problem possesses certifiable structure, Brain/QSA can sometimes manipulate an exponentially larger logical space without materializing the corresponding dense representation.

Whether that translates into greater intelligence, sample efficiency, reasoning ability, or learning speed must be demonstrated separately.

---

# Quantum Language

Brain's language architecture is also evolving beyond a conventional neural text generator.

Current research is building a **hybrid classical/quantum language cortex** in which most large-scale language representation remains classical while selected semantic working-state, relational, routing, and contextual operations can interact with QSA-managed quantum state.

Language is important to Brain as more than an output layer.

It is the interface connecting:

* learned knowledge
* active memory
* factual state
* procedures
* reasoning
* external evidence
* semantic relations
* cognitive working state
* eventual actions and communication

Because of that role, the current research program does not treat language as a fixed response template or a list of predefined statements.

Brain is being developed to construct its own responses from its learned and active cognitive state.

This hybrid language work remains under active experimental validation. Architectures that perform well during development but fail sealed holdout evaluation are rejected rather than promoted.

---

# Persistent Quantum and Semantic State

Another major difference from this repository is that quantum state is no longer treated simply as something created, measured, and discarded for each isolated operation.

Current Brain research includes persistent semantic working state in which small logical quantum systems can evolve across cognition.

One important structure is **Tripair**, a three-qubit semantic state designed to carry multiple interacting logical, relational, routing, and validation roles.

Tripair is part of a wider research program involving compact states that can control or query much larger frozen representational systems.

The intended principle is not that a few qubits magically contain unlimited classical information.

Instead, a compact state can have substantial computational leverage by controlling:

* which transformation runs
* how multiple routes interact
* which relationship is queried
* how context changes interpretation
* which existing Brain structures become active

The larger knowledge remains in Brain's cortex, memory, operator systems, and structured state.

---

# Learning Beyond Repeated Token Training

Current Brain research is also moving toward a distinction between **training the cortex** and **learning new information after the cortex exists**.

Not every new fact, episode, relation, rule, or procedure should require large-scale neural retraining.

The developing architecture separates several forms of learning:

```text
New fact
    → exact knowledge state

New episode
    → episodic / provenance state

New relation
    → relational working state

Repeated structure
    → learned semantic law

World constraint
    → structured inference state

Small new behavior
    → bounded trainable organ

Fundamentally new general capability
    → neural training when necessary
```

The long-term goal is for Brain to use expensive neural weight training only where it is actually required.

That is fundamentally different from an ordinary language model in which most durable learning requires another weight-update cycle.

---

# Earlier Parameter-Efficiency Experiment

An earlier internal experiment reported an approximate **237× parameter-efficiency advantage** on its specific evaluated task while maintaining greater than **90% measured task performance**.

That result remains part of the project's experimental history.

However, current project reporting no longer treats one parameter-efficiency ratio as proof that:

```text
X Brain parameters = Y conventional-model parameters
```

across all capabilities.

Language, memory, reasoning, structured computation, factual retention, continuous learning, and inference are different dimensions.

For that reason, the earlier projected model-equivalence table has been removed from this README in favor of directly reporting measured results.

The underlying research objective remains the same:

> **Maximize useful cognition per parameter, byte, training example, and unit of computation.**

---

# What Brain Does Not Currently Claim

Brain is an experimental research system.

Current results do **not** establish that:

* every logically quantum state can be simulated efficiently on classical hardware
* every structured QSA method produces a quantum speedup
* representation capacity automatically produces intelligence
* Brain has achieved AGI
* Brain is conscious or self-aware
* the current experimental system is ready for production deployment (it is close, but not quite)

The project deliberately separates:

1. **representation results**
2. **runtime results**
3. **learning results**
4. **language results**
5. **knowledge-retention results**
6. **measured capability results**

That distinction is important to the research.

---

# Why This Repository Is So Different

This repository comes from a period when Brain was still primarily being explored as a brain-inspired neural and quantum-simulation program.

Since then, the project has undergone major changes in:

* language architecture
* QELM training
* quantum-state representation
* QSA execution
* semantic state
* persistent working memory
* Tripair integration
* structured inference
* knowledge representation
* continual learning
* teacher-based knowledge acquisition
* memory organization
* causal reasoning
* representation compilation
* system authority and rollback
* testing and scientific validation

The current private system therefore should **not** be evaluated by running this repository.

Doing so evaluates the historical prototype, not present-day Brain.

---

# About This Repository

This repository originated as an early attempt to combine:

* brain-inspired neural processing
* qubits and quantum simulation
* specialized information encoding
* language processing
* experimental learning systems
* real-time interfaces

It was released while those systems were still highly experimental.

The repository contains known limitations and historical implementation problems, particularly around some of its early quantum additions.

It remains public because preserving the project's development history has value.

### This repository is:

* A **legacy historical prototype**
* Multiple architectural generations behind current Brain
* No longer representative of current Brain performance
* Not the current Brain/QELM language cortex
* Not the current QSA architecture
* Not the current Brain knowledge or memory system
* Not the current Brain learning system
* Not the current hybrid quantum-language architecture
* Not the private current research implementation
* Not recommended as a production system

Issues and pull requests concerning this repository are still welcome, but modifications made here should not be assumed to affect current Brain development.

---

# Public and Private Research

The active Brain implementation, current training artifacts, private knowledge resources, internal evaluation infrastructure, and sensitive research mechanisms are **not contained in this repository**.

Public releases may include selected:

* benchmark results
* research papers
* non-sensitive architecture descriptions
* reproducibility evidence
* open-source supporting technologies
* historical code
* independently releasable research components

Sensitive implementation details are intentionally omitted from this README.

---

# Development History

### August 13, 2026

Brain had advanced into an integrated private quantum/classical cognitive research system with a finalized compact QELM language parent, Brain-native retained knowledge, persistent semantic-state research, QSA structured computation, and active hybrid quantum-language development.

An isolated Brain canary demonstrated externally taught knowledge remaining usable after removal of the teacher model.

The repository you are reading remained a historical codebase and was not updated to mirror the private architecture.

### July 31, 2026

A major QELM language-training stage completed and passed its full sealed finalization process.

The selected approximately 29.93M-parameter language parent substantially improved broad-language, canonical, and structured-safety measurements while preserving the controlled acceptance requirements.

Work then shifted from simply improving the language parent toward integrating it into the larger Brain cognitive architecture.

### July 24, 2026

Brain and QELM had become substantially integrated.

The underlying Brain quantum system was designated private while selected benchmarks, research information, and non-sensitive components could be released independently.

### December 27, 2025

An early version of Brain was released publicly with several known bugs, particularly in its experimental quantum additions.

That historical release forms the basis of this repository.

### December 19, 2025

The process of combining the original Brain project with QELM began.

---

# Research, Collaboration, and Licensing

R&D BioTech Alaska continues active research into Brain, QELM, QSA, quantum/classical cognition, compact AI architectures, structured computation, continual learning, and related technologies.

For research collaboration, funding, licensing, technical partnerships, or other inquiries:

**[contact@rdbiotech.org](mailto:contact@rdbiotech.org)**

---

Copyright © 2024–2026 R&D BioTech Alaska
