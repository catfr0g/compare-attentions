# MQAR Classic final10780 inference-only recovery

## Status and scientific boundary

This is a local implementation candidate, not independent review, execution consent, a scientific result, or ordinary training acceptance. No additional training is permitted. Original provider jobs 145778849/v1 and 145850896/v1 remain ERROR. The original adjacent identity remains initialized; absent original final rank/source/service envelopes and the missing original first HF backup remain explicitly qualified. The continuation's historical inner exception is UNKNOWN; the reproduced scalar-count defect is not claimed as its uniquely proven cause.

The numerical package is the original wheel `512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3`. Do not use the NORMAL lifecycle derivative wheel for these weights. Terminal FULL is `f4cbbbbfa6f1bbae377acf5baa9a07019e00986f26ede4c8c1c76ab22beca3d0`; model.pt is `7e6931f2a0b36083d7f72bb5ceb1fdf83f8b6bab2a773ba2f181cdf9efbf7c7f`; parameter fingerprint is `a21f1b58e50488359e022edb73d6c526814f35b9e8d3d7284d96d778eb897178`.

The recovery worker performs one supported public standalone state-dict + complete-config evaluation, in a new directory without run_identity.json. Public loader provenance stays null. Original native FULL/history/runtime/restore/prefix evidence is a separate structural proof, not a manufactured identity. No optimizer, backward, training entrypoint, or checkpoint RNG restore is admitted.

## Numerical policy

- Existing immutable MQAR K64, length512 VAL bank only: 1000 rows and 64000 supervised query targets. TEST is not evaluated; no arrays are regenerated or copied into this package.
- Public frozen `compare_attention.cli.evaluate.main`, split val, batch32, CUDA FP16 autocast and FP32 stored parameters. Nongenerative same-position label mask (`-1` ignored), not Sorting EOS/greedy decoding.
- A scoped forward-output observer retains chronological typed per-query positions, labels, actual argmax predictions, and FP32 NLL without another model forward. All rows, including the tail batch, close to public correct/target counts and CE. NLL closure is predeclared `rel_tol=2e-5, abs_tol=2e-6` against the frozen summed public loss. Whole-example all-query correctness is separate from query micro accuracy; it is not generated answer EM.
- Every immutable VAL label is independently checked against its designated KV prefix. This checks labels, not model predictions.
- Actual target runtime: two Tesla T4s, Python3.13.16, Torch2.11.0+cu130, CUDA13.0, cuDNN91900, Triton3.6.0. Hardware is checked before setup; prepared interpreter/library pins are checked after setup. The public evaluator is the frozen public reference path, not a claim of native compiled-training bitwise equivalence.

## Package and authority inputs

Use the final frozen candidate selected in ROOT-REPORT.md, not an earlier candidate or rejected quality attempt. Its `payload/src/mqar_final_tail_host_cli.py` is the public host entrypoint. The package includes the saved kernel, operator and compact native/weights input archives, their complete manifests, unchanged audited host primitives, and the actual old scientific input bindings. The large prepared dataset and scientific package remain existing versioned inputs.

Independent operators must supply, without modifying frozen logic:

1. `independent-review.json`: exact worker-policy schema (`status`, `independent`, `scope=MQAR202_FINAL10780_TAIL_RECOVERY`, payload `candidate_sha256`, `policy=RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP`). This agent does not issue it.
2. `hf11-proof.json`: authentic independently reviewed fresh GET-only proof described below. No genuine proof is manufactured by fixtures or this build.
3. `execution-decision.json`, beside those two files: exact fields `status`, `independent`, `scope=MQAR202_FINAL10780_INFERENCE_PUBLICATION_ONLY`, `manifest_sha256`, `execution_authorized`, `telemetry_policy`, `independent_review_sha256`, `historical_hf_proof_sha256`, `nonce`, `kernel_target`, `allocations_permitted`. It must bind the current candidate nonce/target and exactly one allocation. Old train/resume/tail decisions cannot be reused.
4. A distinct host plan review after `bind`: `status`, `independent`, non-owner `reviewer`, `scope=MQAR202_FINAL10780_TAIL_HOST_TWO_STAGE_V1`, canonical `plan_sha256`, and the same qualified policy. It binds both publication actions and all mounted files.
5. Fresh official lifetime evidence and actual owner/private/quota/service-absence observations for each LIVE proof. Engineering fixture documents are not usable authority.

The publisher's exact dataset file set includes both archives, the actual execution decision, independent worker review, and HF11 proof. Final scientific publication derives its file set from `mqar_final_tail_tracking.FILES`, including query records and exact mounted execution-authority evidence. Never substitute a worker-only archive for the host bridge.

### External HF11 proof interface

The required object has exact keys `schema`, `scope`, `private`, `get_only`, `entries`, `raw_evidence`, `missing_original_first_backup`. Schema is `MQAR202_HF11_FRESH_PROOF_V1`; scope is `FRESH_ALL_IMMUTABLE_REVISIONS_ALL_FILES_ALL_BYTES`. The boolean flags must be true. `entries` must equal every verified `{step, revision, files}` object in the genuine frozen recovery receipts: eleven exact revisions, steps980..10780, with each revision's complete filename/size/SHA set. Wrong revisions, swapped steps, omitted or extra assets fail. Terminal revision is `ac58dd57dd1d347fc7bf0c2252ef1147c01ae15c`.

`raw_evidence` is the independently authenticated source evidence index (actual byte counts and SHA256s, not invented fixtures, credentials, private URLs or header dumps). The parent must authenticate the real audit and its transitive raw evidence before providing a reviewed schema projection. No fresh owner-C proof has been assumed or silently read by this implementation. Until it is explicitly supplied, this is a required external proof input, not a claimed successful live audit. The original missing first backup is not backfilled by the continuation's new step980 prefix publication.

## Host commands

All local Python must run through UV with `CUDA_VISIBLE_DEVICES=-1` and `PYTHONDONTWRITEBYTECODE=1` set before imports. Set TMPDIR, TMP and TEMP literally to `C:/Users/Michael/AppData/Local/hermes/cache/scratch`; put heavy runtime/results on E. Do not install dependencies or run a genuine scientific model on HOST.

Set CANDIDATE to the frozen `candidate` directory and HOST to `$CANDIDATE/payload/src/mqar_final_tail_host_cli.py`. Set PY to the already installed selected Python. Set DATA, SCIENCE_DIR, SCIENCE_ARCHIVE and PARENT_ARCHIVE from the authenticated input layout. AUTH_ENV and PRIVATE are existing process-only credential files; never print them. Use a short journal directory such as `$ROOT/j01`: Kaggle2.2.5 encodes the entire absolute upload path as one resumable-state filename. The new guard rejects components over255 bytes before credentials or StartBlobUpload.

Read-only local check and plan binding:

```bash
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" check --candidate "$CANDIDATE"
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" bind --candidate "$CANDIDATE" --journal-root "$JOURNAL" --decision "$EXECUTION_DECISION" --data "$DATA" --output "$PLAN"
```

For each action `publish`, then `submit`, run these actual parser routes with new create-only output paths:

```bash
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" prepare --plan "$PLAN" --review "$PLAN_REVIEW" --action "$ACTION" --science-archive "$SCIENCE_ARCHIVE" --parent-archive "$PARENT_ARCHIVE" --science-dir "$SCIENCE_DIR" --auth-env "$AUTH_ENV" --private "$PRIVATE" --output "$PREPARATION"
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" probe --plan "$PLAN" --review "$PLAN_REVIEW" --action "$ACTION" --receipt "$PREPARATION" --science-archive "$SCIENCE_ARCHIVE" --parent-archive "$PARENT_ARCHIVE" --science-dir "$SCIENCE_DIR" --auth-env "$AUTH_ENV" --private "$PRIVATE" --lifetime "$LIFETIME" --output "$PROOF"
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" seal --plan "$PLAN" --review "$PLAN_REVIEW" --proof "$PROOF" --output "$APPROVAL"
uv run --offline --no-project --no-sync --python "$PY" python -B "$HOST" "$ACTION" --plan "$PLAN" --review "$PLAN_REVIEW" --receipt "$PREPARATION" --approval "$APPROVAL" --science-archive "$SCIENCE_ARCHIVE" --parent-archive "$PARENT_ARCHIVE" --science-dir "$SCIENCE_DIR" --auth-env "$AUTH_ENV" --private "$PRIVATE" --lifetime "$LIFETIME"
```

PREP hashes/reads every immutable mounted file and all eight existing dataset files outside LIVE120; it neither regenerates nor makes a disk copy of the dataset. Actual extracted import roots are authenticated and Win32 write-denial leases are held before credential loading. LIVE refreshes source/private HEAD and GCS generation/checksum evidence, both owner namespaces/private controls, services absence, original ERROR identities, locks, capability, active allocations, quota with20% reserve, official lifetime and disk/process bounds. The transaction fsyncs intent before StartBlobUpload/CreateDataset/SaveKernel. Readback is mandatory; an SDK ACK alone is not success.

If either host action has pending/unknown intent, use only the same `reconcile --action ... --plan ... --review ...` route with the unchanged candidate/decision/nonce/target and usual authenticated source arguments. Never rerun publish/submit, change refs, delete the unknown journal, or create replacement consent.

After the single separately reviewed target allocation: use `status`, then `collect --destination "$DOWNLOAD" --index "$INDEX"`, with the same plan/review/source/credential path arguments. Derive WORK independently from the saved main's owned output directory under the downloaded provider inventory; do not guess a downloaded path or invent an index. `services` additionally takes `--input`, `--work`, `--data`, `--output` and runs a fresh online CPU-only selected-UV observer. `accept` takes `--destination`, `--index`, `--result "$WORK/result"`, `--native "$WORK/native-audit.json"`, `--service-proof`, `--policy-review`, `--data`, `--output`; it replays the whole qualified object and the original final-HF journal revision, not selected fields.

The new distinct inference ClearML task disables repository/framework/arg/stream/resource hooks before the requested seed/model. It flushes and closes (not mark_completed), then independently verifies exact artifacts and full paginated quality events. Private HF uses the unchanged HEAD-CAS publisher and complete immutable-revision byte readback. Original tasks are never turned green or rewritten.

### Final-HF unknown ACK, GET-only

The public `mqar_final_tail_service_cli.py` also accepts `--hf-reconcile-only --hf-revision <exact immutable 40-hex revision>` with its explicit release/science/input/work/private/nonce/data/output arguments. It requires an existing same-operation pending/unknown/verified journal; it cannot publish a new action. It verifies all results and mounted authority, preserves original journal bytes, copies that journal to a new local reconciliation sink, performs only destination GETs, and emits a separately typed receipt with `scientific_acceptance=false`, `clearml_completion=NOT_ASSERTED`, `remote_mutations=0`. It does not score, initialize a task, complete ClearML, or make an ERROR allocation accepted. A verified journal's wrong revision argument is rejected. Any allocation failure is still STOPALL; no second inference allocation is authorized by this path.

## Conditional next witness

`mqar_final_tail_next.py --spec ... --review ... --sorting-classic ... --sorting-delta ... --science ... --work ... --output ...` runs offline CPU-only in a fresh selected UV process. The spec contains exact release/result/native/services/provider/review/plan/data/input/output paths. It replays the native proof and entire existing canonical MQAR composite, pins both original qualified Sorting acceptances, and invokes the unchanged state consumer with a narrowly scoped Classic predecessor callback. It emits a separate tagged Classic→MQAR Delta witness, never an ordinary fabricated training acceptance. The result remains `submitted=false`, `fresh_launch_authority_required=true`, `later_pilots_authorized=false`. It is not a Delta launcher, seed303/404 decision, or permission to skip future NORMAL producer envelopes.

## Verification interpretation

The complete saved-host → two-stage SDK publisher/sealer → saved main/setup → real selected UV child → producer/service/publication → normal collector → fresh real-SDK observer → whole composite → fresh next-consumer tracer is a CPU metadata/HTTP engineering fixture. Its scorer boundary is deliberately substituted and its oracle-shaped records are not scientific results. Genuine saved scientific models/optimizers/RNG are not restored or executed on HOST. Separate fresh tiny-model tests exercise the actual authenticated original public parser/loader/evaluator and query-output observer, including VAL-vs-TEST labels, tail batches and no generation. Genuine terminal tensor/history/runtime/restore/prefix checks are structural only. None of these establish a new T4 score or live publication.

ROOT-REPORT.md, the exact staging manifest, source/test inventories, unique JUnit index, CLI-help proof, archive readback and preservation report identify the final tested bytes. Earlier errors/partial attempts remain evidence, not alternate accepted candidates. No local commit, push, independent PASS, cloud mutation or allocation is part of this deliverable.
