"""
models.py — Pydantic schemas for the SHL Assessment Recommender API.

Schema spec:
- recommendations: List[Recommendation] defaulting to [] (empty list, never null)
- end_of_conversation: bool defaulting to False
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="Full conversation history, last item is the new user message",
    )


class Recommendation(BaseModel):
    name: str = Field(..., description="Exact assessment name from catalog")
    url: str = Field(..., description="Exact URL from catalog")
    test_type: str = Field(
        ...,
        description="Type letter codes e.g. 'K', 'A,P', 'B,S'",
    )
    keys: str = Field(
        default="",
        description="Full type names e.g. 'Knowledge & Skills'",
    )
    duration: str = Field(
        default="—",
        description="Duration string e.g. '30 minutes', '—'",
    )
    languages: str = Field(
        default="—",
        description="Comma-separated languages or abbreviated list",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's conversational response")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Assessment recommendations (empty list when gathering context)",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True when the user has confirmed satisfaction",
    )


class HealthResponse(BaseModel):
    status: str


class CatalogEntry(BaseModel):
    name: str
    url: str
    description: Optional[str] = None
    test_types: List[str] = []
    test_types_full: List[str] = []
    duration: Optional[int] = None
    duration_str: Optional[str] = None
    remote_testing: bool = True
    adaptive_irt: bool = False
    job_levels: List[str] = []
    languages: List[str] = []
