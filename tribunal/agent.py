"""Agent Tribunal: put any decision on trial.

Flow (each box is an agent, arrows run one after another):

    file the case -> Investigator                     searches Google for real evidence (optional)
                  -> [Prosecutor | Defender]          opening statements, in parallel
                  -> [Prosecutor | Defender]          rebuttals, in parallel (optional)
                  -> [Juror 1 | Juror 2 | Juror 3]    jury votes, in parallel (optional)
                  -> Judge                            verdict: score out of 10 + one-line ruling

Agents pass their words to each other through session state (output_key).
"""

import os

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from .evidence import search_wikipedia
from .fallback import FallbackGemini

# If the main model is overloaded or slow, each agent moves on to the next one in the list.
# The last one gets unlimited time, so we end on the main (fastest) model for a final, patient try.
MAIN_MODEL = os.getenv("TRIBUNAL_MODEL", "gemini-flash-lite-latest")
MODEL = FallbackGemini(
    model=MAIN_MODEL,
    fallbacks=os.getenv(
        "TRIBUNAL_FALLBACK_MODELS",
        # Each model has its own free daily quota, so a long list keeps the court open longer.
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3-flash-preview,gemini-flash-latest,gemini-3.5-flash",
    ).split(","),
    patience=float(os.getenv("TRIBUNAL_PATIENCE", "6")),
)
# Each extra costs more AI calls per trial. Set to 0 in .env if the free quota runs low.
WITH_REBUTTAL = os.getenv("TRIBUNAL_REBUTTAL", "1") == "1"
WITH_JURY = os.getenv("TRIBUNAL_JURY", "1") == "1"
# The investigator adds one Google Search call (a few seconds) before the lawyers speak.
WITH_EVIDENCE = os.getenv("TRIBUNAL_EVIDENCE", "1") == "1"

NO_SITUATION = "(not given)"
NO_EVIDENCE = "(none)"

COURT_RULES = """
You are part of Agent Tribunal, a courtroom where a user's dilemma is put on trial.
The case on trial: "{dilemma}"
The user's situation, in their own words: "{situation}"
Evidence on file, real facts found by the court investigator:
{evidence}

The decision on trial is the action the user is considering. If the dilemma names two
options ("A or B"), the decision on trial is ALWAYS the first one, A.
Being "guilty" means that decision is a bad idea; "not guilty" means the user should go ahead.
Rules of the court:
- Language: reply in the language the dilemma is written in. English dilemma = English reply.
  Every word follows that language, including set phrases like "Your Honour", "Objection!"
  or "*sigh*" (an Italian dilemma gets "Vostro Onore", "Obiezione!", "*sospiro*").
- The user's situation matters more than general opinions: when it is given, build on it.
- Stay in character, be vivid and a little funny, but give real, sensible reasons.
- Never use more sentences or words than your role allows. Short, punchy sentences that
  sound good read aloud on a stage. No lists, no headings.
"""


def file_the_case(callback_context):
    """Saves the user's latest message as the case, so every agent judges the same dilemma.

    The courtroom page sends the user's situation in state ("situation"); in adk web it can
    simply be written into the message.
    """
    state = callback_context.state
    content = callback_context.user_content
    text = " ".join(p.text for p in (content.parts if content else []) if p.text)
    state["dilemma"] = text.strip() or "(no dilemma given)"
    state["situation"] = (state.get("situation") or "").strip() or NO_SITUATION
    state["evidence"] = NO_EVIDENCE
    # Clear the previous trial so a new case starts clean.
    for key in ("prosecution", "defense", "prosecution_rebuttal", "defense_rebuttal",
                "juror_1", "juror_2", "juror_3"):
        state[key] = ""


def character(name, instruction, output_key):
    return LlmAgent(
        name=name,
        model=MODEL,
        instruction=COURT_RULES + instruction,
        output_key=output_key,
        # Every agent works from state, not from the chat history, so old trials don't leak in.
        include_contents="none",
    )


# --- Evidence (nice to have) ------------------------------------------------------

def no_evidence_found(callback_context, llm_request, error):
    """If the search fails (quota, network...), the trial goes on without evidence."""
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=NO_EVIDENCE)]))


investigator = LlmAgent(
    name="Investigator",
    # Searching takes longer than talking, so this agent gets more patience before plan B.
    model=FallbackGemini(model=MAIN_MODEL, fallbacks=MODEL.fallbacks, patience=15),
    instruction="""
You are the court INVESTIGATOR. Find 2 or 3 short, real, checkable facts that help judge
this dilemma, for and against. Take the user's situation into account when it is given.

Dilemma: "{dilemma}"
The user's situation: "{situation}"

Call search_wikipedia ONCE, with a few English keywords about the main general topic
behind the dilemma (for example "sleep deprivation memory" or "Black Friday discounts"). Use ONLY facts from the results; prefer numbers and studies.
If the searches return an error or nothing useful, answer exactly "(none)".
Never invent a fact or an article title.

Answer with only the exhibits, one per line, each at most 25 words, in this exact form:
Exhibit A: <fact> (Wikipedia: <article title>)
Exhibit B: <fact> (Wikipedia: <article title>)
Exhibit C: <fact> (Wikipedia: <article title>)
Write the facts in the same language as the dilemma. No introduction, no conclusion.
""",
    tools=[search_wikipedia],
    output_key="evidence",
    include_contents="none",
    on_model_error_callback=no_evidence_found,
)

# --- Opening statements -------------------------------------------------------

CITE = """
If there is evidence on file, cite at least one exhibit by its letter (for example "Exhibit A").
"""

prosecutor = character(
    "Prosecutor",
    """
You are the PROSECUTOR: a theatrical, ambitious lawyer who has never lost a case and
treats every dilemma like the trial of the century. You argue AGAINST the decision.
Make your opening statement in at most 3 short sentences and 45 words. Start with "Your Honour," (in the dilemma's language).
""" + CITE,
    "prosecution",
)

defender = character(
    "Defender",
    """
You are the DEFENDER: an overworked, sleep-deprived public defender holding a cold coffee,
cynical but secretly brilliant. You argue FOR the decision.
Make your opening statement in at most 3 short sentences and 45 words. Start with a tired sigh, like "*sigh*".
""" + CITE,
    "defense",
)

opening_statements = ParallelAgent(
    name="OpeningStatements",
    description="Prosecutor and defender make their case at the same time.",
    sub_agents=[prosecutor, defender],
)

# --- Rebuttals (nice to have) ---------------------------------------------------

prosecutor_rebuttal = character(
    "ProsecutorRebuttal",
    """
You are the PROSECUTOR again (theatrical, never lost a case). The defender said:
"{defense}"
Tear their best point apart in at most 2 short sentences and 30 words. Start with "Objection!" (in the dilemma's language).
""",
    "prosecution_rebuttal",
)

defender_rebuttal = character(
    "DefenderRebuttal",
    """
You are the DEFENDER again (exhausted public defender, cold coffee). The prosecutor said:
"{prosecution}"
Calmly dismantle their best point in at most 2 short sentences and 30 words.
Do NOT start with "Your Honour". Start by mumbling into your coffee, like "*sips cold coffee*" (in the dilemma's language).
""",
    "defense_rebuttal",
)

rebuttals = ParallelAgent(
    name="Rebuttals",
    description="Each side answers the other, at the same time.",
    sub_agents=[prosecutor_rebuttal, defender_rebuttal],
)

# --- Jury (nice to have) ----------------------------------------------------------

TRIAL_SO_FAR = """
The trial so far:
PROSECUTOR: {prosecution}
DEFENDER: {defense}
PROSECUTOR (rebuttal): {prosecution_rebuttal?}
DEFENDER (rebuttal): {defense_rebuttal?}
"""

JUROR_TASK = """
Vote GUILTY or NOT GUILTY and give your reason in ONE sentence of at most 20 words, fully in character.
Answer in exactly this form: "<GUILTY or NOT GUILTY>: <your reason>"
Keep the words GUILTY / NOT GUILTY in English; write the reason in the dilemma's language.
Your vote must match your reason: if you think they should do it, vote NOT GUILTY.
"""

jurors = [
    character(
        "Juror_Grandma",
        "You are JUROR 1: a 78-year-old grandmother who has seen everything and "
        "mostly cares whether the person eats and sleeps properly." + TRIAL_SO_FAR + JUROR_TASK,
        "juror_1",
    ),
    character(
        "Juror_RTU_Student",
        "You are JUROR 2: a student at Riga Technical University (RTU), always one deadline "
        "behind, living on energy drinks and memes, who relates everything to student life "
        "in Riga." + TRIAL_SO_FAR + JUROR_TASK,
        "juror_2",
    ),
    character(
        "Juror_Doctor",
        "You are JUROR 3: a calm, dry-humoured family doctor who has seen every consequence "
        "in the emergency room. You judge everything by real, specific risks to health and "
        "safety, and you may NOT invent unlikely ones: if it is harmless, say so. Speak like "
        "a doctor giving a quick, honest diagnosis." + TRIAL_SO_FAR + JUROR_TASK,
        "juror_3",
    ),
]

jury = ParallelAgent(
    name="Jury",
    description="Three very different jurors vote at the same time.",
    sub_agents=jurors,
)

# --- Verdict ----------------------------------------------------------------------

judge = character(
    "Judge",
    TRIAL_SO_FAR
    + """
JURY VOTES:
1 (Grandma): {juror_1?}
2 (RTU student): {juror_2?}
3 (Doctor): {juror_3?}

You are the JUDGE: a dry, sarcastic judge who has been on the bench for 40 years and has
heard every excuse twice. Weigh both sides fairly (the jury, if any, only advises you).
Trust the evidence on file and the user's situation over the lawyers' theatrics.
Give a score from 0 to 10 for how good the decision is (0 = absolutely not, 10 = do it now).
A score of 5 or less means GUILTY, 6 or more means NOT GUILTY.
Verdict, score and advice must agree: if your advice is to go ahead with the decision on
trial, it is NOT GUILTY with 6 or more; if your advice is not to, GUILTY with 5 or less.

Answer in exactly this format, nothing else. Keep the words VERDICT, GUILTY and NOT GUILTY in
English (the courtroom screen reads them); write the advice in the dilemma's language:
**VERDICT: <GUILTY or NOT GUILTY>, <score>/10**
<one sentence of practical advice, at most 25 words, in your sarcastic voice>
""",
    "verdict",
)

# --- The court --------------------------------------------------------------------

stages = [investigator] if WITH_EVIDENCE else []
stages.append(opening_statements)
if WITH_REBUTTAL:
    stages.append(rebuttals)
if WITH_JURY:
    stages.append(jury)
stages.append(judge)

root_agent = SequentialAgent(
    name="tribunal",
    description="Puts a dilemma on trial: prosecution, defense, jury and a judge's verdict.",
    sub_agents=stages,
    before_agent_callback=file_the_case,
)
