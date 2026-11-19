# Leased LIVE-window operator (review required)

This is a HOST-only orchestration change. It does not change seal.py, preflight,
120-second absence freshness, the scientific worker, NORMAL wheel or source
identity. The original failed public seal's precise guard identity remains
UNKNOWN. Saved-time pure predicates and the controlled public-chain regression
are separate evidence, not a recovered historical traceback.

## Contract

Use a new reviewed candidate, new canonical singleton and independent singleton
review. Preparation remains a separate public `seed202_pilot.cli prepare`
invocation for exactly one action. The supported driver then acquires real
Win32 FILE_SHARE_READ handles and hashes the complete candidate/input/evidence
closure plus plan, review, preparation receipt and lifetime document before LIVE.
It warms the unchanged full plan verifier, then calls `seed202_pilot.cli.main`
for probe, seal and exactly one publish OR submit. It never retries. Every
original scope, source, import-root closure, native predecessor, signed
preparation, command-wide pending/unknown, before-auth, freshness and transport
capability check still runs. There is no cached approval or fresh-result flag.
No `_HELD_INPUTS` nesting is introduced.

`retained_lease` reuses only a subset of a presently held full-hash lease owned
by the current synchronous thread. Shared owner state is invalidated before the
owning handles close, including exception cleanup; copied Contexts cannot extend
that lifetime. A different thread must acquire its own genuine lease instead of
borrowing the first thread's cached digest. Missing coverage acquires genuine
handles and hashes. A saved dictionary or boolean is not a lease. File stats
are rechecked under handles, never substituted for initial hashing. HOST digest
consumers may reuse that initial SHA only under these same held handles. The
expected root-membership set uses equivalent platform-normalized path components;
every actual directory enumeration, per-file pin and native link/path check still
runs on every predecessor row. Warm and seal-only `held_inputs` scopes use the
existing API sequentially; they are never nested around authenticated commands.

Credentials (`--auth-env` and `--private`) remain process-only origins consumed
by the inherited provider/private-byte path. They are NEVER included in the
artifact lease list, copied to E:, hashed or inspected by the engineering driver.

## Public argv recipe (NOT execution permission)

After independent review and a separate human decision permitting real work:

1. Build a fresh bound candidate with the supported `unified_normal_operator`
   builder, external NORMAL source review/decision and unchanged recipe/source
   pins. Recreate a public singleton with `seed202_pilot.cli make-plan` using the
   request's exact three genuine predecessors (qualified Classic is embedded;
   the two other paths are repeated `--acceptance` arguments). Obtain external
   independent review of that new canonical singleton. Never reuse the old seal,
   old singleton review or a consumed output/journal.
2. Run public `seed202_pilot.cli prepare` for one action with common arguments
   below, `--action ACTION --output NEW_PREPARATION`. Keep new output names.
3. Write a JSON array of three argv arrays. All three arrays carry IDENTICAL
   common values: `--candidate-dir`, `--case`, `--parent-archive`, `--auth-env`,
   `--private`, `--science-dir`, `--mqar-data`, `--lifetime`, `--journal`, `--plan`,
   `--review`. The action journal is exactly `new_plan["journals"][ACTION]`, not
   an illustrative short path. Add these command-specific fields:
   - `probe ... --action ACTION --preparation NEW_PREPARATION --output NEW_PROOF`
   - `seal ... --proof NEW_PROOF --output NEW_APPROVAL`
   - `ACTION ... --preparation NEW_PREPARATION --approval NEW_APPROVAL`
4. With PYTHONPATH selecting the reviewed candidate's `payload/source/src`, run:

   ```text
   uv run --offline --no-project --no-sync --python SELECTED_PYTHON python -B -m unified_normal_live --commands REVIEWED_COMMANDS_JSON
   ```

   Mask HOST CUDA and retain the authorized scratch/cache environment. A nonzero
   return stops the chain; do not regenerate or retry LIVE artifacts automatically.
5. For submit, finish publish acknowledgement first, then separately prepare and
   run a new submit chain. Unknown acknowledgements remain GET-only reconciliation;
   this driver does not implement reconciliation or renewal.

The delivered `unl02` is UNBOUND REVIEW-ONLY. Fixture candidates whose names start
`unlf` contain explicitly SYNTHETIC source authority solely for intercepted HTTP
engineering tests; they must never be selected for actual operations. No new
independent review, human permission, real proof or action approval is issued by
this work. CPU timings and synthetic transport budgets do not establish private
API latency, cloud success or CUDA/T4 qualification.
