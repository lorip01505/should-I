# Agent Tribunal

**Stuck on a decision? Put it on trial.**

Ask one chatbot and you get one polite answer, usually the one you were hoping for.
Agent Tribunal gives your dilemma a real trial instead: a prosecutor argues against it,
a defender argues for it, a jury of three very different characters votes, and a judge
who has heard every excuse twice gives the verdict.

Built for GDG Riga Community Kickoff 2026, "Coding with AI: Building Agents" (Challenge 02),
with Google's Agent Development Kit (ADK) and Gemini.

## Example

> **You:** Should I pull an all-nighter before tomorrow's exam?
>
> **Prosecutor:** Your Honour, tired students forget what they cram...
>
> **Defender:** *sigh* Two chapters are still unread...
>
> **Jury:** Grandma: GUILTY. RTU student: NOT GUILTY. Doctor: GUILTY.
>
> **Judge:** **VERDICT: GUILTY, 3/10**. Study until 1:00, sleep six hours, review at breakfast.

## Answers that don't fall from the sky

- **Your situation.** Add your own context ("my laptop died and my thesis is due Friday")
  and the trial changes: the same dilemma went from GUILTY 4/10 to NOT GUILTY 9/10.
- **Real evidence.** Before anyone speaks, an Investigator agent looks up facts on
  Wikipedia (a tool the agent calls itself) and files them as Exhibit A, B, C with links.
  The lawyers must cite the exhibits, and the judge trusts evidence over theatrics.

## How it works

```
SequentialAgent "tribunal"
├── LlmAgent       Investigator        search_wikipedia tool -> Exhibits A, B, C
├── ParallelAgent  OpeningStatements   Prosecutor | Defender         (at the same time)
├── ParallelAgent  Rebuttals           Prosecutor | Defender         (each answers the other)
├── ParallelAgent  Jury                Grandma | RTU student | Doctor
└── LlmAgent       Judge               score out of 10 + one-sentence ruling
```

- Each agent saves what it says in shared session state (`output_key`); the next stage
  reads it from there.
- A `before_agent_callback` files the user's message as the case, so every agent judges
  the same dilemma and old trials never leak into new ones.
- The courtroom page passes "your situation" to the agents as session state (`state_delta`).
- Evidence, rebuttal and jury are optional (`TRIBUNAL_EVIDENCE`, `TRIBUNAL_REBUTTAL`,
  `TRIBUNAL_JURY` in `.env`).
- Plan B for busy hours (`tribunal/fallback.py`): if a Gemini model answers "overloaded" or
  takes more than 6 seconds, that agent moves on to the next model, so a trial never stalls.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp tribunal/.env.example tribunal/.env   # then paste your Gemini API key into tribunal/.env
python server.py
```

- **http://localhost:8000/court** is the courtroom: the three characters sit on stage, the
  lawyers type their arguments turn by turn, the jury votes one by one, and the judge bangs
  the gavel (with sound) before the verdict. `/court/classic` is the first, simpler layout.
- **http://localhost:8000/dev-ui/?app=tribunal** is the standard `adk web` screen, which shows
  who said what, step by step.

`server.py` is ADK's own web server (`get_fast_api_app`) plus the courtroom page in `court/`,
which talks to the agents through ADK's `/run_sse` endpoint. Plain `adk web` still works too.

Get a free API key at https://aistudio.google.com/apikey.

## Tests

```bash
python -m unittest discover tests     # offline, no Gemini calls: agent order, state, plan B
python tests/check_live.py            # live, with the server running: grades 6 real trials
```

The live check runs dilemmas chosen to catch the flaws we hit while building: verdicts
that contradict their own advice, wrong language, arguments too long for a stage,
invented sources, A-or-B dilemmas, and a user trying to bully the court into a verdict.
