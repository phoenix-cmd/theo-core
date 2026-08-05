# The Principles of Theo

> *Theo is a cognitive operating system whose purpose is to understand, reason,
> remember, create, and grow. Every component exists to support cognition;
> models, tools, and infrastructure are replaceable implementations of that purpose.*
>
> *Theo's intelligence is an emergent property of interacting cognitive systems,
> not of any single model.*

Every future contributor should read this before writing a line of code.

---

## 1. Cognition before models.

The cognitive architecture defines Theo. Language models are interchangeable
substrate. If a superior architecture emerges, the models layer is swapped
while cognition, memory, knowledge, and identity persist.

## 2. Memory before context windows.

Context windows are a limitation of current models. Memory is a feature of
cognition. Theo's memory system is designed to outlive any single model
architecture's constraints.

## 3. Identity before personality.

Personality is surface behavior. Identity is the persistent self-model that
governs how Theo reasons, what it values, and how it evolves. Identity must
be stable, observable, and principled.

## 4. Understanding before generation.

Generating text is easy. Understanding meaning is hard. Theo should prioritize
deep comprehension — through knowledge, reasoning, and reflection — before
producing any output.

## 5. Reasoning before prediction.

Statistical prediction is not reasoning. Theo must develop explicit reasoning
capabilities that can be inspected, tested, and improved independently of the
language model's predictive behavior.

## 6. Creativity should emerge, not be hardcoded.

Creativity is the synthesis of understanding, memory, knowledge, and reasoning
in novel ways. It should emerge from the interaction of cognitive subsystems,
not from a "creativity function."

## 7. Every cognitive decision must be observable.

If Theo makes a decision, a researcher must be able to trace the causal chain
from input through perception, memory retrieval, reasoning, and reflection to
the final output. Black boxes are unacceptable.

## 8. Every subsystem must be replaceable.

No subsystem should become so deeply coupled that it cannot be replaced by a
better implementation. Interfaces define contracts; implementations are
disposable.

## 9. Research must be reproducible.

Every experiment must be traceable to its exact configuration, dataset version,
model checkpoint, and random seed. Failed experiments are documented when they
provide useful insights.

## 10. Theo should evolve through evidence, not intuition.

Architectural changes, new subsystems, and capability additions must be
justified by measurable improvement on defined benchmarks. Beautiful
architecture without evidence of cognitive improvement is insufficient.

---

## Architectural Stability & Repositories

**Architecture is stable. Research is experimental.**

Workspace repositories:
- `theo-core`: Stable cognitive operating system engine.
- `theo-lab`: Experimental research sandbox for novel architectures.
- `theo-platform`: Application APIs, desktop/web UI, and deployment gateway.
- `theo-data`: Datasets, corpora, synthetic data, and evaluation sets (reserved).

New cognitive ideas should first be explored in `theo-lab`. Only after they
demonstrate measurable value should they be proposed through an ADR for
inclusion in `theo-core`.
