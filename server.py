"""Runs Agent Tribunal: the courtroom page at /court plus the full adk web UI at /dev-ui.

    python server.py      then open http://localhost:8000/court
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
# Load the key and the TRIBUNAL_* switches before the agent module reads them.
load_dotenv(HERE / "tribunal" / ".env")

from google.adk.cli.fast_api import get_fast_api_app  # noqa: E402

from tribunal import agent as tribunal  # noqa: E402

app = get_fast_api_app(agents_dir=str(HERE), web=True)


@app.get("/court", include_in_schema=False)
async def court_page():
    """The courtroom: cast on stage, trial as a live conversation."""
    return FileResponse(HERE / "court" / "stage.html")


@app.get("/court/classic", include_in_schema=False)
async def court_classic():
    """The first layout: side-by-side columns."""
    return FileResponse(HERE / "court" / "index.html")


@app.get("/court/stage", include_in_schema=False)
async def court_stage():
    return RedirectResponse("/court")


@app.get("/court/mockup", include_in_schema=False)
async def court_mockup():
    return FileResponse(HERE / "court" / "mockup.html")


@app.get("/court/config")
async def court_config():
    """Tells the page which stages this court runs, so it can show the right seats."""
    return {
        "evidence": tribunal.WITH_EVIDENCE,
        "rebuttal": tribunal.WITH_REBUTTAL,
        "jury": tribunal.WITH_JURY,
        "jurors": [j.name for j in tribunal.jurors] if tribunal.WITH_JURY else [],
    }


@app.get("/c", include_in_schema=False)
async def short_link():
    return RedirectResponse("/court")


app.mount("/court/avatars", StaticFiles(directory=HERE / "court" / "avatars"), name="avatars")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
