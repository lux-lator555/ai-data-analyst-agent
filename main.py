# main.py — FastAPI backend

import os
import io
from typing import List
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent, followup_chat, auto_detect_dataset

app = FastAPI(title="AI Data Analyst Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationEntry(BaseModel):
    role: str
    content: str


class FollowUpRequest(BaseModel):
    question: str
    original_summary: str
    original_recommendations: str
    conversation_history: List[ConversationEntry]
    api_key: str


@app.get("/")
def root():
    return {"status": "AI Data Analyst Agent API is running!"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    goal: str = Form(...),
    api_key: str = Form(...)
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = run_agent(goal=goal, df=df, api_key=api_key)

    return {
        "summary": result["summary"],
        "charts": result["charts"],
        "plotly_charts": result.get("plotly_charts", []),
        "turns": result["turns"],
        "recommendations": result["recommendations"],
        "roi_charts": result.get("roi_charts", []),
        "quality_report": result["quality_report"],
        "confidence_scores": result["confidence_scores"],
        "model_export": result.get("model_export", {}),
        "rows": len(df),
        "columns": len(df.columns)
    }


@app.post("/autodetect")
async def autodetect(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = auto_detect_dataset(df, api_key)
    return result


@app.post("/followup")
async def followup(request: FollowUpRequest):
    response = followup_chat(
        question=request.question,
        original_summary=request.original_summary,
        original_recommendations=request.original_recommendations,
        conversation_history=[
            {"role": e.role, "content": e.content}
            for e in request.conversation_history
        ],
        api_key=request.api_key
    )
    return {"response": response}