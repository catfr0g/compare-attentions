# RULER Variable Tracking: completed three-seed comparison

All six planned trajectories are scientifically accepted. This is the repository's
**tokenized `ruler_variable_tracking` task**, not the complete NVIDIA RULER suite.
Results below are **VAL**, not TEST generalization.

## Matched protocol

- Model seeds: 202, 303, 404; fixed dataset seed: 1337.
- Requested algorithmic training budget: **5e15 FLOPs per accepted trajectory**.
- Two Tesla T4 GPUs; local batch 32 per rank; global batch 64.
- FP16 mixed precision and compiled execution; constant learning rate 0.001.
- Last full-budget iterate, not best-checkpoint selection. No quality-based seed
  removal or replacement. Every accepted run completed the full budget with zero
  recorded AMP-skipped optimizer updates and 11 unique FULL checkpoints.
- Primary metric: prompt-only, free-running whole-chain exact match. Native cloud
  greedy generation uses its own previous predictions and the full vocabulary.
  Teacher-forced accuracy/loss are diagnostic, not substitutes for this metric.

## Accepted final VAL counts

Each run evaluates 1,000 chains, 5,000 positions, and 4,000 links.
The baseline is **Classic at the same model seed and last full training budget**.

| Model seed | Architecture | Exact chains | Correct positions | Correct links |
| --- | --- | ---: | ---: | ---: |
| 202 | Classic | 963 / 1000 | 4883 / 5000 | 3931 / 4000 |
| 202 | DeltaNet | 997 / 1000 | 4992 / 5000 | 3995 / 4000 |
| 303 | Classic | 1000 / 1000 | 5000 / 5000 | 4000 / 4000 |
| 303 | DeltaNet | 1000 / 1000 | 5000 / 5000 | 4000 / 4000 |
| 404 | Classic | 0 / 1000 | 959 / 5000 | 614 / 4000 |
| 404 | DeltaNet | 998 / 1000 | 4993 / 5000 | 3996 / 4000 |

Whole-chain accuracy across seeds, mean +/- **sample** standard deviation (n=3):
**Classic 65.43% +/- 56.70 percentage points; DeltaNet 99.83% +/- 0.15 points**.
DeltaNet was more consistent on these three runs of this task. This does not
establish a universal architecture advantage or statistical significance.
Classic404's low-quality result is retained; its cause is unknown. It completed
all 109,107 optimizer updates, so this is not a two-hour timeout or truncated run.
Links can be correct on a wrong alternative chain; do not equate link accuracy
with whole-chain success. The zero Classic404 baseline makes a relative fold
improvement undefined; use absolute percentage-point differences instead.

## Compute and operational qualifications

- Classic: 109,107 updates, **5,000,042,725,244,928** actual FLOPs per run.
- DeltaNet: 90,944 updates, **5,000,004,431,249,408** actual FLOPs per run.
- Six accepted trajectories: **30,000,141,469,483,008** actual training FLOPs.
  Small requested-budget overshoots come from the final whole optimizer update.
- Original DeltaNet303 job `147014746/v1` completed its training budget but ended
  in provider ERROR before generated VAL when publication acknowledgement became
  unknown. It is not an accepted result. Its additional
  **5,000,004,431,249,408** training FLOPs remain separate operational spend.
  The user-authorized fresh attempt `147044887/v1` supplies the accepted seed303
  result. The original ERROR/journals remain unchanged; its cause is not claimed
  fixed merely because the fresh attempt completed.
- Future-only runtime policies were adjusted to observed execution: Classic404
  two hours plus 30-minute completion tail; DeltaNet404 four hours plus 30 minutes.
  The 20% quota reserve remained enabled. Scientific package/configuration and
  full training budgets were preserved. Historical producer profiles are retained.
- Provider RUNNING/COMPLETE intervals include operational overhead and polling;
  they are not native optimizer timing or architecture-throughput measurements.
- Native evidence and private immutable HF publications were verified for every
  accepted run. ClearML training diagnostics remain auxiliary. No HOST model
  inference, generation, optimizer/scaler/RNG restoration, or TEST scoring was
  used to construct this report.
- Scientific six-case closure is independent of the final finite candidate's
  five-predecessor `next` contract; no synthetic `GROUP_DONE` claim is made.

## Evidence and local report

Acceptance receipt SHA-256 identifiers, in chronological case order:

| Case | Receipt SHA-256 |
| --- | --- |
| Classic202 | `fd9599575ed90f39b244f943003a250f5a08fdc0652d991c97da5e61e172bffa` |
| DeltaNet202 | `f4351a0089419548e024394a906066bb55e9c4ae03b629fed7b42b1288568ed7` |
| Classic303 | `c60d49707d813fa32ff8cce03050453111b2208e6cd152c068b37e17eba8585e` |
| DeltaNet303 | `1ca3429de6388622008d8ef30d2b4182feca9dcf3dd3d17992d791bb6187836c` |
| Classic404 | `afe515975b0bedb9a6cc48a970b1a6e72cdcd015056a024689a6d2683860f445` |
| DeltaNet404 | `370ffb1a5c755a95d39c08762049d543fb726d2ee15e1d2ca208af5217980fa9` |

The scientific wheel SHA-256 is
`dc7f14bfa941624193089c8ca201df9f39c13d5592cb5ad32e2a2dbd92803475`.

The source-pinned metrics, counts CSV, shared 0-100% figure, builder, verification,
and report-only ZIP are stored locally under the Git-ignored directory
`local/reports/ruler-variable-tracking-three-seeds-final-v1/`.
The ZIP SHA-256 is
`8ae6ff25657ff0c46de51ade6ab4b8c9e97e27c21603b467f5a4db6194ed5695`.
Private payloads, raw evidence, credentials, datasets, and model weights are not
included in this public documentation or committed report artifacts.
