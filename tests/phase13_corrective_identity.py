from memcontam.readiness.phase13_v3_authority_models import V3Identity


def corrective_identity(generation: str = "disposable-alpha") -> V3Identity:
    return V3Identity(
        run_id=f"phase13-main-a-{generation}-v3",
        package_id=f"phase13-main-a-{generation}-execution-freeze-v3",
        authorization_id=f"phase13-main-a-{generation}-authorized-execution-v3",
        cost_proof_id=f"phase13-main-a-{generation}-cost-proof-v3",
    )
