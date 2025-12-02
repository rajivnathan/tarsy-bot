"""
Unit tests for structured output models.

Tests Pydantic models used with Gemini's structured output feature
for type-safe ReAct response parsing.

EP-0028: Gemini Structured Outputs for ReAct Responses
"""

import pytest
from pydantic import ValidationError

from tarsy.agents.parsers.structured_models import (
    ToolAction,
    FinalAnswer,
    ReActStructuredResponse,
)


@pytest.mark.unit
class TestToolAction:
    """Test ToolAction model validation."""

    def test_create_valid_tool_action(self):
        """Test creating a valid tool action."""
        action = ToolAction(
            thought="I need to check the pods in the namespace",
            action="kubernetes-server.pod_list",
            action_input={"namespace": "default"}
        )
        
        assert action.thought == "I need to check the pods in the namespace"
        assert action.action == "kubernetes-server.pod_list"
        assert action.action_input == {"namespace": "default"}

    def test_tool_action_with_empty_params(self):
        """Test tool action with empty parameters."""
        action = ToolAction(
            thought="Checking system status",
            action="system-server.health_check",
            action_input={}
        )
        
        assert action.action_input == {}

    def test_tool_action_default_action_input(self):
        """Test that action_input defaults to empty dict."""
        action = ToolAction(
            thought="Reasoning",
            action="server.tool"
        )
        
        assert action.action_input == {}

    def test_tool_action_with_complex_params(self):
        """Test tool action with complex nested parameters."""
        action = ToolAction(
            thought="Need to query with filters",
            action="kubernetes-server.resources_list",
            action_input={
                "apiVersion": "v1",
                "kind": "Pod",
                "namespace": "production",
                "labelSelector": {"app": "web", "env": "prod"}
            }
        )
        
        assert action.action_input["labelSelector"]["app"] == "web"


@pytest.mark.unit
class TestFinalAnswer:
    """Test FinalAnswer model validation."""

    def test_create_valid_final_answer(self):
        """Test creating a valid final answer."""
        answer = FinalAnswer(
            thought="I have gathered all necessary information",
            final_answer="The root cause is memory exhaustion. Increase limits to 2Gi."
        )
        
        assert answer.thought == "I have gathered all necessary information"
        assert "root cause" in answer.final_answer.lower()

    def test_final_answer_with_markdown(self):
        """Test final answer with markdown formatting."""
        answer = FinalAnswer(
            thought="Analysis complete",
            final_answer="""**Root Cause:** Pod OOM killed.

**Resolution:**
1. Increase memory limits
2. Add resource requests
3. Monitor with alerts"""
        )
        
        assert "**Root Cause:**" in answer.final_answer

    def test_final_answer_requires_both_fields(self):
        """Test that both thought and final_answer are required."""
        with pytest.raises(ValidationError):
            FinalAnswer(thought="Done")  # Missing final_answer
        
        with pytest.raises(ValidationError):
            FinalAnswer(final_answer="Complete")  # Missing thought


@pytest.mark.unit
class TestReActStructuredResponse:
    """Test ReActStructuredResponse union type."""

    def test_create_tool_action_response(self):
        """Test creating response with tool action."""
        response = ReActStructuredResponse(
            response=ToolAction(
                thought="Checking pods",
                action="kubernetes-server.pod_list",
                action_input={"namespace": "default"}
            )
        )
        
        assert isinstance(response.response, ToolAction)
        assert not isinstance(response.response, FinalAnswer)

    def test_create_final_answer_response(self):
        """Test creating response with final answer."""
        response = ReActStructuredResponse(
            response=FinalAnswer(
                thought="Analysis complete",
                final_answer="Issue resolved"
            )
        )
        
        assert isinstance(response.response, FinalAnswer)
        assert not isinstance(response.response, ToolAction)

    def test_parse_tool_action_from_json(self):
        """Test parsing tool action from JSON string (as Gemini would return)."""
        json_str = '''
        {
            "response": {
                "thought": "Need to get namespace info",
                "action": "kubernetes-server.namespace_get",
                "action_input": {"name": "default"}
            }
        }
        '''
        
        result = ReActStructuredResponse.model_validate_json(json_str)
        
        assert isinstance(result.response, ToolAction)
        assert result.response.action == "kubernetes-server.namespace_get"
        assert result.response.action_input["name"] == "default"

    def test_parse_final_answer_from_json(self):
        """Test parsing final answer from JSON string (as Gemini would return)."""
        json_str = '''
        {
            "response": {
                "thought": "I have sufficient data",
                "final_answer": "The namespace is healthy"
            }
        }
        '''
        
        result = ReActStructuredResponse.model_validate_json(json_str)
        
        assert isinstance(result.response, FinalAnswer)
        assert result.response.final_answer == "The namespace is healthy"

    def test_invalid_json_fails(self):
        """Test that invalid JSON structure fails validation."""
        json_str = '''
        {
            "response": {
                "invalid_field": "value"
            }
        }
        '''
        
        with pytest.raises(ValidationError):
            ReActStructuredResponse.model_validate_json(json_str)

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        # Missing 'thought' field
        json_str = '''
        {
            "response": {
                "action": "server.tool",
                "action_input": {}
            }
        }
        '''
        
        with pytest.raises(ValidationError):
            ReActStructuredResponse.model_validate_json(json_str)


@pytest.mark.unit
class TestJsonSchemaGeneration:
    """Test JSON schema generation for Gemini structured outputs."""

    def test_schema_generation(self):
        """Test that model generates valid JSON schema."""
        schema = ReActStructuredResponse.model_json_schema()
        
        assert "properties" in schema
        assert "response" in schema["properties"]

    def test_tool_action_schema(self):
        """Test ToolAction schema structure."""
        schema = ToolAction.model_json_schema()
        
        assert "properties" in schema
        assert "thought" in schema["properties"]
        assert "action" in schema["properties"]
        assert "action_input" in schema["properties"]

    def test_final_answer_schema(self):
        """Test FinalAnswer schema structure."""
        schema = FinalAnswer.model_json_schema()
        
        assert "properties" in schema
        assert "thought" in schema["properties"]
        assert "final_answer" in schema["properties"]

    def test_schema_has_union_definition(self):
        """Test that schema includes union type info for response field."""
        schema = ReActStructuredResponse.model_json_schema()
        
        # The schema should have anyOf for union types
        response_schema = schema["properties"]["response"]
        # Pydantic uses anyOf for union types
        assert "anyOf" in response_schema or "$ref" in response_schema
