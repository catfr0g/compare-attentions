# Tiny Shakespeare: independent NORMAL Kaggle series

`TINY_INDEPENDENT_CURRENT_V1` is a finite six-case profile:
`TinyClassic202`, `TinyDeltanet202`, `TinyClassic303`, `TinyDeltanet303`,
`TinyClassic404`, `TinyDeltanet404`. It does not extend historical Sorting/MQAR
case registries or require their completions as predecessors. The first prefix
is empty; each later case requires the exact genuine Tiny acceptance prefix,
with ordinary whole-object replay. No exception policy admits historical cases.

## Build and inspect (no provider credentials)

Use the existing public builder:

```text
uv run --offline --no-sync python -B scripts/unified_normal_operator.py --inputs INPUTS.json --normal-archive TINY-SOURCE.zip --output NEW-CANDIDATE
```

Inputs use the existing NORMAL request fields: `sorting_candidate` (authenticated
bootstrap lineage only), `normal_package`, `normal_case`, `runtime_record`,
`source_profile`, `generation`, `normal_plan`, `normal_recipe`, `predecessors`.
The generation uses the existing `UNIFIED_NORMAL_GENERATION_INTENT_V1` schema,
Tiny case ID, fresh nonzero 32-hex nonce, exact selected source identity and the
original Tiny user-request receipt. This is provenance, not technical approval.

The emitted `payload/source/src` is the actual import root for
`python -B -m seed202_pilot.cli check`, `make-plan`, and `next`. Pass
`--candidate-dir NEW-CANDIDATE`; `make-plan` also takes `--case TinyClassic202`,
`--journal NEW-JOURNAL-ROOT`, and `--output NEW-SINGLETON.json`. Invoke under the
existing controlled UV environment with that emitted root on `PYTHONPATH`.
The singleton's review domain is canonical serialized JSON, not the JSON file's
SHA alone. The existing full singleton review schema and validators are unchanged.

## Payload and source boundary

The current wheel is byte-authenticated against all 71 maintained scientific
Python members. The Tiny worker source release is separately pinned; historical
source profiles and accepted artifacts are unchanged. `data/tiny` in the actual
worker archive contains `input.txt`, `meta.json`, and all six arrays. The kernel
selects that directory before NORMAL setup/worker/task/finish. The historical
science mount supplies only the already-authenticated UV/runtime bootstrap; its
Sorting arrays are not passed to Tiny. No pip bootstrap or old runner is added.

The case retains full 5e15-FLOP pricing, world2/local32/global64, FP16, compile,
constant LR0.001, original byte-identical recipe and last-budget-iterate selection.
Quality is native cached validation character CE/PPL, not Sorting or MQAR scores.
No extra model forward or TEST scoring is authorized by this adapter.

Existing fresh two-stage target naming is retained deliberately, with fresh
Tiny-owned per-case nonces: `zimike/sorting-mqar-seed202-<nonce16>/1` and
`zimike/seed202-pilot-<nonce16>`. These are resource names, not historical-case
permission or scientific identity. Dataset and kernel remain private.

## External gates (not self-issued)

An unbound candidate can be checked and planned. Preparation and all authenticated
routes reject it before credentials with
`UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED`. The parent supplies the existing
exact NORMAL `independent-review.json` and `execution-decision.json`, then rebuilds
a NEW candidate with `normal_review` and `normal_decision` inputs. Bind review to
the exact normal-plan/source subject; binding does not grant a fresh launch seal.
Recompute and independently review the resulting whole singleton.

Retain the existing Win32 write-denial ownership and `static_preflight` BEFORE
LIVE through probe/seal/action/reconcile. Publish and submit need separate fresh
proofs/seals. The 120-second absence window, quota/privacy/namespace/source guards,
durable mutation intents, SDK request authentication, and GET-only unknown-ACK
reconciliation are unchanged. Stop on new provider/target failure; no automatic
retry/replacement/retraining. No optional GPU probes are introduced.

Local tests cover metadata, byte closure, actual extraction/data selection and
the genuine missing-review boundary with a stdlib hardware facade. They do NOT
qualify CUDA/T4, live HF/ClearML, or an allocated training run. The inherited LM
reporter's CE<=700 representability guard is unchanged in this host-only patch;
its scientific reporting-contract review remains explicit, not silently repaired.
