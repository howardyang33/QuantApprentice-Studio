# QuantApprentice Studio Architecture

## Goal

Build a full QuantApprentice Studio system around the final research codebase without mutating the historical
research tree. The studio should eventually support the entire:

- outer hypothesis-verification loop
- inner teacher-standard internalization loop
- final lesson deployment and signal scoring

instead of only exposing a scoring-only console.

## Layers

1. Import and provenance
2. Registry and asset resolution
3. Clean pipeline wrappers
4. Agent runtime
5. Workflow orchestration
6. Backend API layer
7. Frontend / multi-agent layer

## Supported system modes

The studio architecture is organized around four user-facing workflow modes:

1. `Full Pipeline`
2. `Outer Loop Only`
3. `Inner Loop Only`
4. `Scoring Only`

`Scoring Only` is the most complete mode in the current implementation, but the wrapper and orchestrator layers
now explicitly prepare for the full dual-loop system.

## Clean wrapper layer

The clean wrapper layer calls the curated reproducibility package bundled next to the Studio package:

- `../QuantApprentice_clean`

through studio-owned wrapper logic instead of writing new metadata back into the historical repo.

Implemented wrapper catalog:

- `outer_loop`
- `explainability_report_v2`
- `inner_loop_suite`
- `scope_alignment`
- `market_backtest`

## Current runtime agents

- `PlannerAgent`: research-goal parsing and workflow-brief construction
- `HypothesisAgent`: convert a broad research goal into a concrete hypothesis family
- `FactorDesignAgent`: define factor families, templates, and diversification intent
- `TeacherTrainingAgent`: outer-loop wrapper ownership
- `VerificationAgent`: inspect outer-loop outputs and summarize what was actually produced
- `ExplainabilityAgent`: explainability-report refresh ownership
- `TeacherAcceptanceAgent`: decide which accepted teacher set should feed the inner loop
- `TeacherSelectionAgent`: frozen-teacher intake and complementary-scope selection
- `ApprenticeAgent`: inner-loop warmup and final-lesson selection ownership
- `EvaluationAgent`: scope-alignment and market-backtest wrapper ownership
- `TeacherZooAgent`: frozen teacher browsing
- `LessonAgent`: final lesson and scoped lesson loading
- `AlignmentAgent`: archived scope-alignment replay
- `BacktestAgent`: archived market-backtest replay
- `SignalScoringAgent`: recorded single-signal score lookup, later live GPT scoring
- `ChiefResearchAgent`: unified entry point

## Current orchestration layer

- `QuantPipelineOrchestrator`: mode-level planner that maps:
  - `Full Pipeline`
  - `Outer Loop Only`
  - `Inner Loop Only`
  - `Scoring Only`
  onto wrapper-driven execution steps
- `WorkflowStepExecutor`: agent-aware step dispatcher that hands each wrapper/runtime step back to the corresponding studio agent
- `WorkflowRunner`: execution wrapper that keeps workflow context, delegates step execution, and records studio-owned workflow artifacts

## Runtime serving policy

- Prefer a local GPT-OSS model configured by `QA_STUDIO_LOCAL_MODEL_PATH`
- Serve it through a local vLLM OpenAI-compatible endpoint
- Keep the API-compatible transport so future multi-agent orchestration can still call a uniform chat-completions interface
- Allow explicit remote API override through environment variables when needed

## Short-term roadmap

1. Keep GPT-OSS as the default scoring backbone
2. Keep live scoring on the studio-local runtime instead of mutating old research code
3. Expand wrapper coverage until the full dual-loop workflow is callable from the studio layer
4. Introduce explicit planner / memory / teacher-training / apprentice / evaluation agents on top of the wrappers
5. Add frontend pages on top of the stable runtime catalog
