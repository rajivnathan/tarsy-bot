"""
Structured output models for ReAct responses with Gemini.

This module provides Pydantic models for Gemini's structured output feature,
enabling type-safe ReAct response parsing without complex text parsing.

Uses simple union type pattern:
- ToolAction: When the agent wants to invoke a tool (has action/action_input fields)
- FinalAnswer: When the agent has completed analysis (has final_answer field)

Gemini infers the type based on which fields are present - no discriminator needed.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ToolAction(BaseModel):
    """Response when the agent wants to use a tool."""
    thought: str = Field(description="Your step-by-step reasoning about what to investigate next")
    action: str = Field(description="Tool to call in format: server_name.tool_name (e.g., kubernetes-server.pod_get)")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool as a JSON object")


class FinalAnswer(BaseModel):
    """Response when the agent has completed analysis."""
    thought: str = Field(description="Your final reasoning before concluding")
    final_answer: str = Field(description="Complete analysis with root cause, resolution steps, and recommendations")


class ReActStructuredResponse(BaseModel):
    """
    Union type for ReAct responses.
    
    Gemini infers the type based on which fields are present:
    - ToolAction has: thought, action, action_input
    - FinalAnswer has: thought, final_answer
    
    Usage:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ReActStructuredResponse.model_json_schema(),
            },
        )
        result = ReActStructuredResponse.model_validate_json(response.text)
    """
    response: ToolAction | FinalAnswer = Field(description="Either a tool action or final answer")

