"""Tail-specific independent-review contract, never the consumed continuation sealer."""

from mqar_final_tail_host_contract import exact, validate_plan, value_sha
from mqar_final_tail_host_live import validate_live
from mqar_final_tail_native import require


def validate_review(plan, review):
    require(
        type(review) is dict
        and set(review) == {"status", "independent", "reviewer", "scope", "plan_sha256", "policy"},
        "TAIL_INDEPENDENT_REVIEW",
    )
    require(
        review["status"] == "PASS"
        and review["independent"] is True
        and type(review["reviewer"]) is str
        and bool(review["reviewer"])
        and review["reviewer"] != plan["implementation_owner"]
        and review["scope"] == "MQAR202_FINAL10780_TAIL_HOST_TWO_STAGE_V1"
        and review["plan_sha256"] == value_sha(plan)
        and review["policy"] == "RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP",
        "TAIL_INDEPENDENT_REVIEW",
    )


def seal(plan, review, proof, *, fresh=True, revalidate=True):
    if revalidate:
        validate_plan(plan)
    validate_review(plan, review)
    validate_live(plan, proof, proof["action"], fresh=fresh)
    require(
        proof["preparation"]["receipt"]["review_sha256"] == value_sha(review), "TAIL_REVIEW_RECEIPT"
    )
    return {
        "schema": "MQAR202_FINAL10780_TAIL_ACTION_APPROVED_V1",
        "plan": plan,
        "review": review,
        "proof": proof,
    }


def validate(approval, *, revalidate=True):
    require(
        type(approval) is dict and set(approval) == {"schema", "plan", "review", "proof"},
        "TAIL_APPROVAL_SCHEMA",
    )
    require(
        exact(
            approval,
            seal(
                approval["plan"],
                approval["review"],
                approval["proof"],
                fresh=False,
                revalidate=revalidate,
            ),
        ),
        "TAIL_EXACT_APPROVAL",
    )
    return approval
