from quant_toolkit.teacher_loop.loop import TeacherSpec, compare_spec_to_zoo, fallback_spec
from quant_toolkit.teacher_loop.registry import ALL_REGISTERED_FEATURES, BASE_FEATURE_NAMES, build_feature_registry
from quant_toolkit.teacher_loop.zoo import infer_zoo_partition


def test_feature_registry_counts():
    registry = build_feature_registry()
    assert registry["feature_count"] == len(ALL_REGISTERED_FEATURES)
    assert registry["base_feature_count"] == len(BASE_FEATURE_NAMES)
    assert registry["derived_feature_count"] == registry["feature_count"] - registry["base_feature_count"]


def test_infer_zoo_partition_prefers_try_for_attempted_but_rejected_teacher():
    item = {
        "status": "rejected",
        "accepted_as_teacher": False,
        "artifact_refs": ["reports/pilot2/PILOT2B_EXECUTION_REPORT.md"],
    }
    assert infer_zoo_partition(item) == "try"


def test_novelty_detection_flags_near_duplicate():
    spec = TeacherSpec(
        title="Duplicate reversal teacher",
        teacher_role="duplicate",
        research_family="reversal",
        hypothesis="duplicate",
        sample_template="weak_state_reversal_pool",
        model_family="ridge_regression",
        target_kind="future_return_5d",
        evaluation_contract="yearly_q5_majority",
        feature_columns=["ret_1", "ret_3", "ret_5", "J"],
        novelty_rationale="duplicate",
    )
    spec.validate()
    zoo_payload = {
        "teachers": [
            {
                "memory_id": "mem_1",
                "title": "Old reversal teacher",
                "zoo_partition": "try",
                "research_family": "reversal",
                "sample_template": "weak_state_reversal_pool",
                "model_family": "ridge_regression",
                "feature_columns": ["ret_1", "ret_3", "ret_5", "J"],
            }
        ]
    }
    novelty = compare_spec_to_zoo(spec, zoo_payload)
    assert novelty.too_similar is True


def test_fallback_spec_is_valid():
    spec = fallback_spec()
    spec.validate()


def test_gpu_fallback_spec_is_valid():
    spec = fallback_spec(require_gpu=True)
    assert spec.model_family == "xgb_classification_gpu"
    spec.validate()
