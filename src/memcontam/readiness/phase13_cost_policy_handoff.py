from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExactReplacement(_FrozenModel):
    kind: Literal["exact_utf8_replace"]
    old_utf8: Literal["Experiment-v12"]
    new_utf8: Literal["Experiment-v10"]
    expected_match_count: Literal[1]


class AddendumTarget(_FrozenModel):
    filename: Literal["2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md"]
    pre_write_sha256: Literal[
        "786e1ef1db7656e38beb5ab9ec316adc7df9bb1cc2f16d389f3612c76fbd2015"
    ]
    observed_sha256: Literal[
        "e55ab2fe57380f870eecc6331ebf47f7642ddc39807d294a912dd09c9122ca22"
    ]
    expected_post_write_sha256: Literal[
        "5f45108e833cddcd68c612c73f5d56e9b01cb38946bdf89a1f20532351cff9c4"
    ]
    operations: tuple[ExactReplacement]


class ExperimentTarget(_FrozenModel):
    filename: Literal["Phase 13-Compatible Pilot Main and Exploratory Experiment Design revised-v10.md"]
    pre_write_sha256: Literal[
        "bf6cf602d3ead47e95d9e158c1e3fe89ffab1ba4093a40f7d7ccb781faa0e0ec"
    ]
    observed_sha256: Literal[
        "5597f27d688c19efbcf47dc7369de02a947eac55a5493a69a3aa9098dfe25616"
    ]
    expected_post_write_sha256: Literal[
        "5597f27d688c19efbcf47dc7369de02a947eac55a5493a69a3aa9098dfe25616"
    ]
    operations: tuple[()]


class HandoffTargets(_FrozenModel):
    post_cutoff_addendum: AddendumTarget
    experiment_v10: ExperimentTarget


class CanonicalBlockHashes(_FrozenModel):
    CORE_EXECUTION_ENVELOPE_REGISTRY_V2: Literal[
        "4dec48f105c8d4730706d1d99d05bb14bab96a8e643811db1ebdd26e612590d5"
    ]
    CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1: Literal[
        "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
    ]
    COST_ENVELOPE_V2: Literal[
        "806b4f6fe752b3ed12d6dd9c081f75f54873a575870ffab62a84ca1fc032460a"
    ]


class StopConditions(_FrozenModel):
    reference_reviewer_required: Literal[True]
    router_sync_permitted: Literal[False]
    repository_projection_activation_permitted: Literal[False]
    mr_p4_resume_permitted: Literal[False]
    mr_p5_start_permitted: Literal[False]
    mr_p6_start_permitted: Literal[False]
    main_execution_permitted: Literal[False]


class ControlledExternalWrite(_FrozenModel):
    schema_version: Literal["phase13_controlled_external_write_v1"]
    status: Literal["PARTIAL_WRITE_VISIBLE_REFERENCE_REVIEW_REQUIRED"]
    targets: HandoffTargets
    canonical_block_hashes: CanonicalBlockHashes
    residual_patch_path: Literal[
        "data/phase13/main/cost_envelope_v2/post_cutoff_addendum_residual_v1.patch"
    ]
    post_write_stop_conditions: StopConditions
