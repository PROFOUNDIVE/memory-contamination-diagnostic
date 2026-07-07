from memcontam.logging.schema import TrialLog, VerifierResult


def test_trial_log_minimal_shape() -> None:
    log = TrialLog(
        trial_id="t1",
        run_id="r1",
        task_name="game24",
        sample_id="s1",
        baseline="no_memory",
        arm="clean",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        raw_response="final: (6/(1-3/4))",
        verifier_result=VerifierResult(is_correct=True),
    )
    assert log.verifier_result.is_correct is True
