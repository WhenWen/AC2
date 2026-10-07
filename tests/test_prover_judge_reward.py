"""Unit tests for ac2.rewards.prover_judge (no network, no verl).

Focus: the judge-token/output dump bug. Assert that on a successful judge reply the returned
dict carries a non-zero judge_completion_tokens and the full judge_response, that token usage
survives OpenAI/vLLM spelling variants, and that EVERY return path emits the same key set (verl
pins the dumped keys to the first rollout's dict, so a path that drops keys drops them for all).
"""

import asyncio

import ac2.rewards.prover_judge as pj

PROOF_SOLUTION = "<think>scratch</think>\n<proof>Let n=2k. Then n^2=4k^2 is even. QED.</proof>"
EXTRA_INFO = {"theorem": "Prove that the square of an even integer is even.", "mode": "prover"}


def _install_fake_judge(monkey_response):
    """Patch the HTTP layer so compute_score runs offline. `monkey_response` is a chat dict or exc."""

    async def fake_session():
        return object()

    async def fake_post(session, url, payload, max_retries=5):
        if isinstance(monkey_response, Exception):
            raise monkey_response
        return monkey_response

    pj._get_session = fake_session          # type: ignore[assignment]
    pj._post_with_retries = fake_post        # type: ignore[assignment]


def _chat(content, *, prompt_tokens=123, completion_tokens=456, usage_keys=("prompt_tokens", "completion_tokens"),
          finish_reason="stop", reasoning="analysis..."):
    usage = None
    if usage_keys is not None:
        usage = {usage_keys[0]: prompt_tokens, usage_keys[1]: completion_tokens}
    out = {
        "choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        out["usage"] = usage
    return out


def _run(**kwargs):
    return asyncio.run(pj.compute_score(
        solution_str=kwargs.pop("solution_str", PROOF_SOLUTION),
        extra_info=kwargs.pop("extra_info", EXTRA_INFO),
        judge_url="http://127.0.0.1:9999",
        **kwargs,
    ))


EXPECTED_KEYS = set(pj._empty_extras().keys())


def test_success_captures_tokens_and_response():
    _install_fake_judge(_chat("Reasoning...\nScore: 1", completion_tokens=456))
    out = _run()
    assert out["prover_judge_score"] == 1
    assert out["score"] == 1.0 and out["acc"] == 1.0
    assert out["judge_completion_tokens"] == 456      # the bug: this used to be 0
    assert out["judge_prompt_tokens"] == 123
    assert "[content]" in out["judge_response"] and "Score: 1" in out["judge_response"]
    assert "[reasoning]" in out["judge_response"]
    assert out["judge_parse_failed"] == 0 and out["judge_http_error"] == 0
    assert set(out.keys()) == EXPECTED_KEYS


def test_score_zero():
    _install_fake_judge(_chat("Score: 0"))
    out = _run()
    assert out["prover_judge_score"] == 0 and out["score"] == 0.0
    assert out["judge_completion_tokens"] == 456
    assert set(out.keys()) == EXPECTED_KEYS


def test_usage_spelling_variants():
    _install_fake_judge(_chat("Score: 1", prompt_tokens=10, completion_tokens=20,
                              usage_keys=("input_tokens", "output_tokens")))
    out = _run()
    assert out["judge_prompt_tokens"] == 10 and out["judge_completion_tokens"] == 20


def test_missing_usage_sentinels_not_silent_zero():
    _install_fake_judge(_chat("Score: 1", usage_keys=None))
    out = _run()
    # No usage reported -> -1 sentinel (distinguishable from a real 0), verdict still parsed.
    assert out["judge_completion_tokens"] == -1
    assert out["prover_judge_score"] == 1
    assert set(out.keys()) == EXPECTED_KEYS


def test_parse_failure_keeps_tokens():
    _install_fake_judge(_chat("I cannot decide.", completion_tokens=77))
    out = _run()
    assert out["judge_parse_failed"] == 1
    assert out["prover_judge_score"] == 0
    assert out["judge_completion_tokens"] == 77   # tokens captured even when the verdict didn't parse
    assert set(out.keys()) == EXPECTED_KEYS


def test_truncated_flag():
    _install_fake_judge(_chat("....Score: 1", finish_reason="length"))
    out = _run()
    assert out["judge_truncated"] == 1
    assert set(out.keys()) == EXPECTED_KEYS


def test_http_error():
    _install_fake_judge(RuntimeError("connection refused"))
    out = _run()
    assert out["judge_http_error"] == 1
    assert out["prover_judge_score"] == 0
    assert out["judge_response"] == ""
    assert set(out.keys()) == EXPECTED_KEYS


def test_missing_proof_no_judge_call():
    _install_fake_judge(RuntimeError("judge should not be called"))
    out = _run(solution_str="<think>...</think> no proof block here")
    assert out["proof_tag_present"] == 0
    assert out["prover_judge_score"] == 0
    assert set(out.keys()) == EXPECTED_KEYS


def test_key_uniformity_across_paths():
    outs = []
    _install_fake_judge(_chat("Score: 1")); outs.append(_run())
    _install_fake_judge(RuntimeError("x")); outs.append(_run())
    outs.append(_run(solution_str="no proof"))
    outs.append(asyncio.run(pj.compute_score(solution_str=PROOF_SOLUTION, extra_info={}, judge_url="http://x")))
    for o in outs:
        assert set(o.keys()) == EXPECTED_KEYS


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all prover_judge reward tests passed")
