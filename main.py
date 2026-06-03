# main.py — FastAPI backend

import os
import io
from typing import List
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent, followup_chat

app = FastAPI(title="AI Data Analyst Agent API")

# Allow requests from the frontend (React PWA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-agent-frontend-2ij.pages.dev",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
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
    """
    Accepts a CSV file, a goal, and a Gemini API key.
    Returns the agent's summary, charts, and business recommendations.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    result = run_agent(goal=goal, df=df, api_key=api_key)

    return {
        "summary": result["summary"],
        "charts": result["charts"],
        "turns": result["turns"],
        "recommendations": result["recommendations"],
        "rows": len(df),
        "columns": len(df.columns)
    }


@app.post("/followup")
async def followup(request: FollowUpRequest):
    """
    Accepts a follow-up question and conversation history.
    Returns the agent's response based on the original analysis context.
    """
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