# main.py — FastAPI backend

import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent

app = FastAPI(title="AI Data Analyst Agent API")

# Allow requests from the frontend (React PWA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "AI Data Analyst Agent API is running!"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    goal: str = Form(...),
    api_key: str = Form(...)
):
    """
    Accepts a CSV file, a goal, and a Gemini API key.
    Returns the agent's summary and any generated charts as base64 strings.
    """
    # Read the uploaded CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Run the agent
    result = run_agent(goal=goal, df=df, api_key=api_key)

    return {
        "summary": result["summary"],
        "charts": result["charts"],
        "turns": result["turns"],
        "rows": len(df),
        "columns": len(df.columns)
    }