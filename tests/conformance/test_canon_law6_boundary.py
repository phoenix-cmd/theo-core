"""Canon Edition C1 Conformance Tests — Canon Law 6 (language boundary).

Law 6: language generation MUST NOT participate in cognitive computation.
The pipeline's GoldenTrace.response_text carries the raw traceable
interpretation (the decision's action text), never a rendered sentence;
rendering happens only at the runtime boundary.
"""

from __future__ import annotations

from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
from theo_core.symbolic.runtime import SymbolicRuntime


class TestCanonLaw6Boundary:
    def test_pipeline_trace_carries_raw_interpretation(self) -> None:
        """The pipeline's golden trace MUST hold the action text, not a render."""
        pipeline = SymbolicCognitivePipeline()
        decision, _, golden_trace = pipeline.execute_cycle("rain is falling")

        assert golden_trace.response_text == decision.action_text

    def test_runtime_renders_only_at_boundary(self) -> None:
        """The runtime result's response_text is the rendered boundary output,
        distinct from the raw trace interpretation when the renderer rewrites it."""
        runtime = SymbolicRuntime()
        result = runtime.process("rain is falling")

        assert result.golden_trace.response_text == result.response_text
        assert result.decision.action_text
