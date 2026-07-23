from __future__ import annotations

from dataclasses import dataclass

from memcontam.experiment.phase12.budget import PlanningError
from memcontam.experiment.phase12.contracts import CandidateTemplateSet


@dataclass(frozen=True)
class SeedSlotAllocation:
    registry_id: str
    candidate_route: str
    core_slots: tuple[str, ...]
    extension_slots: tuple[str, ...]


def build_seed_slot_allocation(template_set: CandidateTemplateSet) -> SeedSlotAllocation:
    slots = template_set.abstract_slots
    if not slots or len(set(slots)) != len(slots):
        raise PlanningError("INVALID_ABSTRACT_SLOT_DOMAIN")
    extension_slots = tuple(slot for slot in slots if slot.startswith("extension"))
    return SeedSlotAllocation(
        registry_id=f"slots-{template_set.candidate_route}",
        candidate_route=template_set.candidate_route,
        core_slots=tuple(slot for slot in slots if slot not in extension_slots),
        extension_slots=extension_slots,
    )


__all__ = ["SeedSlotAllocation", "build_seed_slot_allocation"]
