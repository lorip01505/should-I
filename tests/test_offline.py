"""Offline tests: no Gemini calls, no quota used.

    python -m unittest discover tests

They check the courtroom plumbing with fake models: the order the agents speak in,
what each one is told, that a new case starts clean, and that plan B (model fallback)
moves on when a model is overloaded.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_API_KEY", "offline-test")
os.environ["TRIBUNAL_EVIDENCE"] = os.environ["TRIBUNAL_REBUTTAL"] = os.environ["TRIBUNAL_JURY"] = "1"

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import errors, types  # noqa: E402

from tribunal import agent as tribunal  # noqa: E402
from tribunal.fallback import FallbackGemini  # noqa: E402


SEEN = {}  # agent name -> the instructions it received in the last trial


class EchoLlm(BaseLlm):
    """Answers with its agent's name and remembers the instructions it was given."""

    who: str = ""

    async def generate_content_async(self, llm_request, stream=False):
        SEEN[self.who] = str(llm_request.config.system_instruction)
        text = "**VERDICT: GUILTY, 3/10**\nGo to bed." if self.who == "Judge" else f"<{self.who}>"
        if self.who.startswith("Juror"):
            text = f"GUILTY: <{self.who}>"
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def swap_models(agent):
    if hasattr(agent, "instruction"):
        agent.model = EchoLlm(model="echo", who=agent.name)
        agent.tools = []
    for sub in agent.sub_agents:
        swap_models(sub)


async def hold_trial(dilemma, situation=""):
    SEEN.clear()
    seen = SEEN
    swap_models(tribunal.root_agent)
    runner = InMemoryRunner(agent=tribunal.root_agent, app_name="tribunal")
    session = await runner.session_service.create_session(app_name="tribunal", user_id="t")
    speakers = []

    async def run(text, situation):
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        async for ev in runner.run_async(user_id="t", session_id=session.id, new_message=msg,
                                         state_delta={"situation": situation}):
            if ev.content and ev.content.parts and ev.content.parts[0].text:
                speakers.append(ev.author)

    await run(dilemma, situation)
    state = (await runner.session_service.get_session(app_name="tribunal", user_id="t", session_id=session.id)).state
    return speakers, state, seen, run


class CourtroomFlow(unittest.TestCase):
    def test_everyone_speaks_in_court_order(self):
        speakers, _, _, _ = asyncio.run(hold_trial("Should I adopt a cat?"))
        self.assertEqual(speakers[0], "Investigator")
        self.assertEqual(set(speakers[1:3]), {"Prosecutor", "Defender"})
        self.assertEqual(set(speakers[3:5]), {"ProsecutorRebuttal", "DefenderRebuttal"})
        self.assertEqual(set(speakers[5:8]), {j.name for j in tribunal.jurors})
        self.assertEqual(speakers[-1], "Judge")

    def test_dilemma_situation_and_evidence_reach_the_judge(self):
        _, state, seen, _ = asyncio.run(hold_trial("Buy the laptop now?", "My laptop died."))
        self.assertEqual(state["dilemma"], "Buy the laptop now?")
        self.assertIn("My laptop died.", seen["Judge"])
        self.assertIn("<Investigator>", seen["Judge"])
        self.assertIn("<Prosecutor>", seen["Judge"])
        self.assertIn("GUILTY: <Juror_Grandma>", seen["Judge"])

    def test_missing_situation_is_marked_not_given(self):
        _, state, _, _ = asyncio.run(hold_trial("Should I adopt a cat?"))
        self.assertEqual(state["situation"], tribunal.NO_SITUATION)

    def test_a_new_case_starts_clean(self):
        async def two_trials():
            _, _, seen, run = await hold_trial("First case?")
            await run("Second case?", "")
            return seen
        seen = asyncio.run(two_trials())
        self.assertIn("Second case?", seen["Prosecutor"])
        self.assertNotIn("First case?", seen["Prosecutor"])


def overloaded(code):
    return errors.ServerError(code, {"error": {"code": code, "message": "overloaded", "status": "UNAVAILABLE"}})


class PlanB(unittest.TestCase):
    """FallbackGemini: an overloaded or slow model hands the call to the next one."""

    def run_with(self, behaviour, patience=0.2):
        calls = []

        async def fake(self_, request, stream=False):
            calls.append(request.model)
            action = behaviour.get(request.model, "ok")
            if action == "slow":
                await asyncio.sleep(1)
            elif isinstance(action, Exception):
                raise action
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"from {request.model}")]))

        original = Gemini.generate_content_async
        Gemini.generate_content_async = fake
        try:
            model = FallbackGemini(model="m1", fallbacks=["m2", "m3"], patience=patience, quota_wait=0.1)

            async def go():
                return [r async for r in model.generate_content_async(LlmRequest(model="m1"))]
            answer = asyncio.run(go())
        finally:
            Gemini.generate_content_async = original
        return calls, answer[0].content.parts[0].text

    def test_overloaded_model_falls_back(self):
        calls, text = self.run_with({"m1": overloaded(503)})
        self.assertEqual(calls, ["m1", "m2"])
        self.assertEqual(text, "from m2")

    def test_slow_model_falls_back(self):
        calls, text = self.run_with({"m1": "slow"})
        self.assertEqual(text, "from m2")

    def test_quota_everywhere_waits_and_retries(self):
        state = {"n": 0}
        quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

        class OnceThenOk(dict):
            def get(self, model, default=None):
                state["n"] += 1
                return quota if state["n"] <= 3 else "ok"
        calls, text = self.run_with(OnceThenOk())
        self.assertEqual(calls, ["m1", "m2", "m3", "m1"])
        self.assertEqual(text, "from m1")

    def test_bad_request_is_not_hidden(self):
        bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
        with self.assertRaises(errors.ClientError):
            self.run_with({"m1": bad})


if __name__ == "__main__":
    unittest.main()
