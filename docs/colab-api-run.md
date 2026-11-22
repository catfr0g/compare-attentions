# No-Drive Colab API controller

`python scripts/colab_api_run.py {setup,train,restore,resume,status,logsummary}` uses the existing
full-scope Google OAuth and MCP modules. Without `--execute` it produces a plan:
**no authentication, network calls, private-file reads, directory creation or writes**.
No runtime is created/deleted by this controller. Use the existing managed-runtime
journal/controller for allocation and final deletion.

## Preconditions and commands

Parent/operator must first review the final source, create a **new HF-policy export**
(not the historical Drive-bound export03), verify its report/archive/notebook and
complete the approved HF backup write/readback proof. This implementation was tested
only with synthetic data, real loopback HTTP and benign local child processes; it is
**not evidence of a live Google upload, working target-runtime setup, HF backup or
scientific training**.

Set shell variables to the already-approved private client/token/env paths outside
the repository; never substitute credential **values** in these arguments. `EXPORT`
is the new immutable export directory. The private env file must contain exactly the
seven allowlisted service fields, one `KEY="JSON-quoted value"` per line (the already
prepared private runtime.env format). No dotenv interpolation is performed here.

```bash
# A plan; no actual files need exist yet.
uv run --locked python scripts/colab_api_run.py setup \
  --runtime runtimes/approved-id --run-name deltanet-full-seed42-run04

# Input preflight/transfer/integrity + detached setup. Use a fresh journal.
uv run --locked python scripts/colab_api_run.py setup --execute \
  --runtime "$RUNTIME" --run-name "$RUN" \
  --export-dir "$EXPORT" --env-file "$PRIVATE_ENV" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$SETUP_JOURNAL"

# Reconcile status; choose a NEW observation journal each time.
uv run --locked python scripts/colab_api_run.py status --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --setup-journal "$SETUP_JOURNAL" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$STATUS_JOURNAL"

# Only after status reports setup completed with returncode 0.
uv run --locked python scripts/colab_api_run.py train --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --setup-journal "$SETUP_JOURNAL" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$TRAIN_JOURNAL"
```

Runtime/run identifiers must be literal lowercase RFC1034 labels. A minimal
synthetic plan uses `--runtime runtimes/test --run-name run-1`. No arbitrary code, endpoint, model,
hyperparameter, arbitrary-checkpoint resume, anti-idle or automatic allocation switches are exposed.
The separately reviewed, exact export07 recovery path is documented in
[colab-api-recovery.md](colab-api-recovery.md); it never makes resume automatic.
Training is the pinned runner's **full DeltaNet** default contract (seed42, B32,
4 layers/128 embedding, LR .001, FP16/max scale128, compile, 1e16 FLOPs,
47265 expected updates, no early stopping); the exact manifest config is validated
using the exporter's scientific contract. ClearML and HF remain enabled.

## Transfer and ownership

1. Hash the actual local ZIP; bind its export report, notebook SHA and manifest
   source/data identity, HF storage schema, checkpoint policy and full config.
2. Explicit MCP preflight checks the target Python/T4 before creating owned marker
   and `/content/private` (must be new, mode0700). No private values enter MCP code.
3. GET Runtime obtains `connectionInfo` **in memory only**. Contents requests derive
   from its HTTPS URL, preserve its signed query and use only
   `X-Colab-Runtime-Proxy-Token`, never Google OAuth Authorization. Reject userinfo,
   fragments, non443 ports, non-Google hosts and every redirect. The permitted
   hostname suffixes are `.google.com`, `.googleusercontent.com` and `.colab.dev`.
   The latter was observed in an authenticated Colab Runtime connectionInfo response;
   similar-looking foreign domains remain rejected.
4. Read the unique nonsecret `/content/compare-transport-<nonce>` marker through
   the two documented mapping candidates (root `/` or `/content`). Require exactly
   one byte-identical match; never list the filesystem or guess a private sink.
5. Prove chunk append semantics on a separate new marker file using chunks
   `1, 2, -1` and exact byte readback. Unsupported managers stop with
   `CHUNK_UPLOAD_UNSUPPORTED` **before ZIP upload**. Upload the ZIP in 1MiB chunks
   to a new unique staging name. First chunk truncates only that new staging file;
   later chunks append, and `-1` invokes final save hooks. No blind append retries.
6. Separately PUT the small private env to `/content/private/input.env` and compare
   GET/base64-decoded bytes **in memory**. No credential hashes/sizes enter receipts.
   MIME base64 line wrapping is accepted. MCP chmods the file0600, verifies archive
   SHA without reading back its large contents, then exclusively hard-links to
   `/content/colab-package.zip` and `/content/private/kaggle-env.env`. Exact public
   input receipt is written only after these checks. No archive imports precede SHA.

Existing unknown inputs are never overwritten or deleted. The owned nonsecret
markers/staging hardlinks remain until managed runtime deletion. A partial final
linking failure is ambiguous and is not automatically repaired.

## Detached jobs and reconciliation

Remote ownership lives at `/content/compare-attention-jobs/<run>/identity.json`.
Each phase has an exclusive directory and `spawn_unknown` state written **before**
the only supervisor spawn. Local append-only O_EXCL journals fsync pending records
before MCP calls and HTTP writes. Setup/training supervisors run detached with
DEVNULL stdin and file stdout/stderr; MCP returns immediately, rather than waiting
for installation/training or replaying a 600-second execution.

The supervisor atomically updates state/PIDs, waits for the real child, and records
its actual return code. Bootstrap failure prevents runner launch. Training requires
matching input proof and completed-successful setup, plus a fresh result sink:
`/content/compare-attention-results/<run>`. Package/data paths remain stable between
runtimes; this controller deliberately does not implement automatic resume.

Status/logsummary reuse the confirmed setup MCP session, with new execution IDs;
no repeated kernel creation or completed-ID reuse. Status reads only owned job state,
a bounded metrics.jsonl tail, and the separate last verified HF checkpoint receipt.
Its host-side schema allows only known phase states, return codes, process presence,
finite numeric metrics, SHA/revision and checkpoint step. It does not echo raw logs,
private input or arbitrary strings. `flops` is actual native step telemetry;
checkpoint `step` is the last backed-up update, **not** a guessed current update.
A process-presence check alone does not prove scientific progress or prevent PID reuse.

**After timeout, interruption or any ambiguous response, do not replay setup/train,
change execution IDs to retry the same action, remove the journal, create another
runtime, or relaunch a job.** Use a fresh status journal against the exact setup
journal/runtime/run. Existing local journals and remote phase directories are
permanent duplicate barriers, even after failure. If the setup MCP session is gone,
this first controller fails closed; operator-assisted read-only reconciliation is
required. Release the managed runtime using the existing approved stop workflow
once finished/failed; verify deletion there.

## Protocol references checked

- Google Colab REST runtime `ConnectionInfo` reference:
  `https://developers.google.com/colab/api/reference/rest/v1beta/runtimes`
  documents the proxy header and token expiry.
- Jupyter primary implementation:
  `https://github.com/jupyter-server/jupyter_server/blob/main/jupyter_server/services/contents/largefilemanager.py`
  verifies chunk1 overwrite/new file, subsequent append, chunk-1 final hooks.
- Jupyter REST reference:
  `https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html`.
  Direct retrieval returned403 here; upstream primary implementation was retrieved
  successfully. Live Colab manager compatibility is intentionally checked empirically.

No Google account credentials were opened, no cloud calls/writes/GPU allocations
were made by the implementation worker. Evidence is under
`local/reports/colab-api/no-drive-controller/`.
