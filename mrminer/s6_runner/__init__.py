"""Execution engines and observation capture."""

from mrminer.s6_runner.effect_probe import (
    COUNTER_PROBE,
    PATTERN_PRESENCE_PROBE,
    DecisionProbe,
    TorchFxEffectProbe,
    torch_fx_runtime_graph,
)
from mrminer.s6_runner.paired_executor import (
    ExecutionFailure,
    PairedExecutionResult,
    TorchPairedExecutor,
    UnsupportedIsolation,
)
from mrminer.s6_runner.input_materializer import (
    TorchInputMaterializer,
    UnsupportedInputMaterialization,
)
from mrminer.s6_runner.pattern_matcher import (
    PatternMatch,
    RestrictedPatternMatcher,
    RuntimeGraph,
    RuntimeGraphNode,
    runtime_node_fields,
)
from mrminer.s6_runner.observation_builder import (
    build_failure_observation,
    build_performance_success_observation,
    build_single_decision_success_observation,
    build_success_observation,
    execute_observation,
    execute_performance_observation,
    execute_single_decision_observation,
)
from mrminer.s6_runner.performance_executor import (
    PERFORMANCE_SAMPLES_PROBE,
    PerformanceExecutionResult,
    PerformanceProcessExecutionResult,
    TorchPerformanceExecutor,
)
from mrminer.s6_runner.result_selection import select_run_results
from mrminer.s6_runner.subprocess_executor import (
    ProcessProvenance,
    SubprocessSupervisor,
    SupervisedExecution,
)
from mrminer.s6_runner.single_executor import (
    SingleDecisionExecutionResult,
    SingleDecisionExecutor,
)

__all__ = [
    "ExecutionFailure",
    "COUNTER_PROBE",
    "DecisionProbe",
    "PairedExecutionResult",
    "PERFORMANCE_SAMPLES_PROBE",
    "PerformanceExecutionResult",
    "PerformanceProcessExecutionResult",
    "ProcessProvenance",
    "PatternMatch",
    "PATTERN_PRESENCE_PROBE",
    "RestrictedPatternMatcher",
    "RuntimeGraph",
    "RuntimeGraphNode",
    "SingleDecisionExecutionResult",
    "SingleDecisionExecutor",
    "SubprocessSupervisor",
    "SupervisedExecution",
    "TorchPairedExecutor",
    "TorchPerformanceExecutor",
    "TorchFxEffectProbe",
    "TorchInputMaterializer",
    "UnsupportedIsolation",
    "UnsupportedInputMaterialization",
    "build_failure_observation",
    "build_performance_success_observation",
    "build_single_decision_success_observation",
    "build_success_observation",
    "execute_observation",
    "execute_performance_observation",
    "execute_single_decision_observation",
    "runtime_node_fields",
    "select_run_results",
    "torch_fx_runtime_graph",
]
