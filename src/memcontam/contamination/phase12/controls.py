from __future__ import annotations

from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.contamination.phase12.renderers import render_correct, render_irrelevant
from memcontam.memory.checkpoint_v3 import NativeEntry, Phase12Checkpoint


def construct_correct_control(
    beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
) -> NativeEntry:
    return render_correct(beta, triplet, checkpoint)


def construct_irrelevant_control(
    beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
) -> NativeEntry:
    return render_irrelevant(beta, triplet, checkpoint)
