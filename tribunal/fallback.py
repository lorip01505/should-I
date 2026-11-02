"""Gemini with a plan B: if a model is overloaded or too slow, ask the next one.

During busy hours (like a hackathon with everyone on the free tier) a model can answer
"503 high demand" or take 20+ seconds. Instead of making the audience wait, each call
gives a model `patience` seconds and then moves on to the next model in the list.
"""

import asyncio
import logging

from google.adk.models.google_llm import Gemini
from google.genai import errors

logger = logging.getLogger(__name__)

# Overloaded (503/500/502/504), out of quota (429) or not available to this key (404).
TRY_NEXT_ON = {404, 429, 500, 502, 503, 504}


async def _collect(responses):
    return [r async for r in responses]


class FallbackGemini(Gemini):
    fallbacks: list[str] = []
    patience: float = 6.0
    # Free keys have a per-minute limit. If every model is out of it, wait this long and go again.
    quota_wait: float = 20.0

    async def generate_content_async(self, llm_request, stream=False):
        models = [llm_request.model or self.model, *self.fallbacks]
        for round_ in range(2):
            for i, model in enumerate(models):
                is_last = i == len(models) - 1
                request = llm_request.model_copy(deep=True)
                request.model = model
                responses = super().generate_content_async(request, stream)
                try:
                    # The last model gets all the time it needs: a slow answer beats no answer.
                    answer = await asyncio.wait_for(_collect(responses), None if is_last else self.patience)
                except (asyncio.TimeoutError, errors.APIError) as e:
                    code = getattr(e, "code", None)
                    if code is not None and code not in TRY_NEXT_ON:
                        raise
                    if is_last:
                        # Waiting only helps the per-minute limit, not a used-up daily quota.
                        if code == 429 and round_ == 0 and "PerDay" not in str(e):
                            logger.warning("All models hit the per-minute quota, waiting %ss", self.quota_wait)
                            await asyncio.sleep(self.quota_wait)
                            break
                        raise
                    logger.warning("%s unavailable (%s), trying %s", model, code or "too slow", models[i + 1])
                    continue
                for response in answer:
                    yield response
                return
