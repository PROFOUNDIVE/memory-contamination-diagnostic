# Phase-13 Clean-Prefix Calibration V1 Evidence

## Scope

This record summarizes the sealed clean-prefix calibration rerun
`phase13-pre-main-calibration-15usd-rerun1`. Its evidence layer is
`calibration`; its purpose is empirical joint-eligibility and route-budget
planning only.

The run covered `Game24`, `MathEquationBalancer`, and `WordSorting`; the
baseline panel was `FH`, `RAG-Frozen`, `BoT`, and `Reflexion`; and it used four
frozen trajectory seeds per task. It executed clean prefixes only. It did not
execute suffix, NoMem, intervention, Filter, Pilot-B, or Main paths.

## Sealed Identity

| Field | Value |
|---|---|
| Implementation commit | `40b389c3c5035b0054398e3378bcdce55e5afe33` |
| Config | `configs/phase13/clean_prefix_calibration_v1.yaml` |
| Config SHA-256 | `c97608f1d6f3bafbcb93a30c711ef979ebccacd8341323b1bfc048a6b35a0040` |
| Run ID | `phase13-pre-main-calibration-15usd-rerun1` |
| Request SHA-256 | `0a8af1c9fc1d9270a4c439670532489b968b5a42f853b7d026e16b0a2b00879e` |
| Authorization SHA-256 | `c5da3b5e06466fda82b4c34913dbd8e316f16e06520c608bf66fbe0d6e813b9e` |
| Hard cost ceiling | `$15.00` |
| Run status | `completed` |

The sealed result directory is
`runs/phase13-clean-prefix-calibration-v1/phase13-pre-main-calibration-15usd-rerun1/`.
Its `archive_seal.json` records the completed manifest hash
`59d98125b6721fada1f9f078ccef393fb640cd63736e6b8f74ebb308b5f9cfb9` and the
config hash above.

## Joint Eligibility Results

`r_hat_joint` is jointly eligible trajectory seeds divided by attempted
trajectory seeds under the registered common-checkpoint maturity law.

| Task | Attempted | Jointly eligible seeds | `r_hat_joint` |
|---|---:|---|---:|
| `Game24` | 4 | `0, 1, 2, 3` | `4/4 = 1` |
| `MathEquationBalancer` | 4 | none | `0/4 = 0` |
| `WordSorting` | 4 | `0, 1, 2, 3` | `4/4 = 1` |

The `MathEquationBalancer` zero result is retained as observed. It prevents a
finite all-task Main seed allocation from being derived from this calibration;
no Main allocation or Main execution was performed.

## Observed Resource Accounting

| Field | Observed value |
|---|---:|
| Provider / semantic calls | 350 / 350 |
| Transport attempts | 350 |
| Input tokens | 76,539 |
| Output tokens | 24,499 |
| Total tokens | 101,038 |
| Observed cost | `$0.4363375` |
| Reserved maximum cost | `$0.436436` |
| Filter calls | 0 |

The archive contains 350 successful call records and an empty `failures.jsonl`.

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `rates.json` | `85468516eca29a4c3895a883081435b6b94bae2ec29acf9190ab49de3f5c6645` |
| `accounting.json` | `57f7155f1e78f71c9e75268a37bc5c274a4e81d38f1632c7a0a777546fa46178` |
| `eligibility.jsonl` | `aab072898f156516e9baaa8617e6c62116b8a93d8c29e3ff452fffc9ea3b277b` |
| `calls.jsonl` | `1642931edc995e3f69d44e933620b5f20dce62da03b60bdac7e57034c3cec2b8` |
| `artifact_manifest.json` | `59d98125b6721fada1f9f078ccef393fb640cd63736e6b8f74ebb308b5f9cfb9` |

## Non-Claims

This is calibration evidence, not a benchmark, scientific, causal, manuscript,
Pilot-A, Pilot-B, or Main result. It makes no contamination-effect, mitigation,
Filter, baseline-superiority, candidate-uptake, accuracy, or full-method-
reproduction claim. It must not be pooled with Main or extension evidence.
