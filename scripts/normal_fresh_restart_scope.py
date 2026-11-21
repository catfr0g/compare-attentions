"""One explicitly authorized fresh attempt; no resume, retry loop or review authority."""

from unified_normal_contract import digest, read, require

CASE = "sorting-deltanet-seed303"
NONCE = "ef18969bad3d480c893cd03e92a97eee"
CONSENT = "517da02bd458eb7315781f3d4b5fa5550f87b8131218af1d022628e2a32c169e"
PLAN = "4220d6bb3921ead7f92c0deb4fcce9e868b15ac28f83a8c8a8b01ca970a3f092"
FAILED_KERNEL = 146268145


def permission(generation):
    """Latest direct user choice is distinct from the original eight-case permission."""
    require(
        generation["case_id"] == CASE and generation["nonce"] == NONCE, "FRESH_RESTART_CASE_NONCE"
    )
    authorization = generation["restart"]
    require(
        type(authorization) is dict
        and set(authorization) == {"path", "sha256"}
        and authorization["sha256"] == CONSENT
        and digest(authorization["path"]) == CONSENT,
        "FRESH_RESTART_HUMAN_PROVENANCE",
    )
    value = read(authorization["path"])
    require(
        value["kind"] == "DIRECT_HUMAN_FRESH_RESTART"
        and value["user_text_verbatim"] == "Правим ошибки и обучаем заново"
        and value["case_id"] == CASE
        and value["attempts_authorized"] == 1
        and type(value["attempts_authorized"]) is int
        and type(value["from_step"]) is int
        and value["from_step"] == 0
        and value["seed"] == 303
        and value["resume"] is False
        and value["stop_on_new_failure"] is True,
        "FRESH_RESTART_ONE_FROM_ZERO_ATTEMPT",
    )
    return value


def frozen_plan(case_id, pin):
    require(case_id == CASE and pin == PLAN, "FRESH_RESTART_FROZEN_PLAN")
