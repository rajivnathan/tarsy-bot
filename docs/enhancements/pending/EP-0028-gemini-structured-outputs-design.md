# EP-0028: Gemini Structured Outputs for ReAct Responses

## Problem Statement

Current ReAct response parsing relies on text-based extraction with a complex 3-tier parser that handles various malformed LLM outputs:

1. **Parsing Complexity**: The `ReActParser` implements Tier 1 (standard headers), Tier 2 (mid-line Final Answer), and Tier 3 (mid-line Action fallback) to handle diverse LLM output formats
2. **Format Corrections**: When parsing fails, the system removes malformed messages and sends format correction reminders, consuming additional tokens and iterations
3. **Parameter Flexibility Overhead**: Supporting JSON, YAML, key:value, and key=value parameter formats adds parsing complexity
4. **Provider Variability**: Different LLMs follow the ReAct format with varying reliability, requiring robust fallback logic

**Current Flow** (Text Parsing):
```python
# LLM returns free-form text
response = "Thought: I need to check the pods\nAction: kubernetes-server.pod_list\nAction Input: namespace: default"

# Complex parsing required
parsed = ReActParser.parse_response(response)
if parsed.is_malformed:
    # Remove message, send format correction, retry...
```

**Target Flow** (Structured Output):
```python
# LLM returns validated JSON matching schema (no discriminator needed)
response = '{"response": {"thought": "...", "action": "...", "action_input": {...}}}'

# Direct Pydantic validation - no parsing ambiguity
result = ReActStructuredResponse.model_validate_json(response.text)
```

## Solution Overview

Leverage Gemini's native JSON mode with Pydantic models to eliminate parsing ambiguity:

- **Simple Union Pattern**: Model the ReAct response as a union type (ToolAction | FinalAnswer) - Gemini infers the type from field presence
- **Native JSON Mode**: Use Gemini's `response_mime_type: application/json` with `response_json_schema`
- **Consistent Parsing**: Convert structured response to ReAct text format for `ReActParser` (enables unknown tool detection)
- **Backwards Compatibility**: Keep existing text-based parsing for non-Gemini providers
- **Optional Enhancement**: Make structured outputs opt-in via provider configuration

## Technical Design

### Simple Union Pattern (Like content_moderation.py)

We use a simple union type pattern where Gemini infers the response type from the fields present:

- **ToolAction** has: `thought`, `action`, `action_input`
- **FinalAnswer** has: `thought`, `final_answer`

No discriminator field is needed - Gemini determines the type based on which fields are present.

### Core Pydantic Models

**New File**: `backend/tarsy/agents/parsers/structured_models.py`

```python
"""
Structured output models for ReAct responses with Gemini.
Uses simple union type pattern - Gemini infers type from field presence.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any


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
    Union type for ReAct responses (similar to content_moderation.py pattern).
    
    Gemini infers the type based on which fields are present:
    - ToolAction has: thought, action, action_input
    - FinalAnswer has: thought, final_answer
    """
    response: ToolAction | FinalAnswer = Field(description="Either a tool action or final answer")
```

The structured response is then converted to ReAct text format and parsed by `ReActParser` for consistency and unknown tool detection.

### Provider Configuration

**Update**: `backend/tarsy/models/llm_models.py`

```python
class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    type: LLMProviderType
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tool_result_tokens: Optional[int] = None
    # NEW: Enable structured outputs for Gemini
    use_structured_outputs: bool = Field(
        default=False,
        description="Enable structured JSON outputs for Gemini models (reduces parsing errors)"
    )
```

**Example Configuration** (`config/llm_providers.yaml`):

```yaml
llm_providers:
  # Standard Gemini with text parsing (default)
  gemini-3.0-pro:
    type: google
    model: gemini-3-pro-preview
    api_key_env: GOOGLE_API_KEY
    max_tool_result_tokens: 950000  # Conservative for 1M context
    
  # Gemini with structured outputs enabled
  gemini-3.0-pro-structured:
    type: google
    model: gemini-3-pro-preview
    api_key_env: GOOGLE_API_KEY
    max_tool_result_tokens: 950000  # Conservative for 1M context
    use_structured_outputs: true  # Enable structured JSON responses
```

### LLM Client Integration

**Update**: `backend/tarsy/integrations/llm/client.py`

Use Gemini's native JSON mode via `generation_config`:

```python
from tarsy.agents.parsers.structured_models import ReActStructuredResponse, ToolAction

class LLMClient:
    
    def __init__(self, provider_name: str, config: LLMProviderConfig, settings: Optional[Settings] = None):
        # ... existing init ...
        self.use_structured_outputs = config.use_structured_outputs
    
    def supports_structured_outputs(self) -> bool:
        """Check if this client supports structured outputs."""
        return (
            self.config.type == LLMProviderType.GOOGLE 
            and self.config.use_structured_outputs
        )
    
    async def generate_structured_response(
        self,
        conversation: LLMConversation,
        session_id: str,
        stage_execution_id: Optional[str] = None,
    ) -> LLMConversation:
        """
        Generate response using Gemini's native JSON mode.
        
        Uses response_mime_type and response_json_schema via generation_config.
        Returns LLMConversation with assistant message in ReAct text format.
        """
        if not self.supports_structured_outputs():
            raise ValueError("Structured outputs only available for Gemini with use_structured_outputs=True")
        
        langchain_messages = self._convert_conversation_to_langchain(conversation)
        
        # Use Gemini's native JSON mode
        response = await self.llm_client.ainvoke(
            langchain_messages,
            generation_config={
                "response_mime_type": "application/json",
                "response_json_schema": ReActStructuredResponse.model_json_schema()
            }
        )
        
        # Parse and validate JSON response
        structured = ReActStructuredResponse.model_validate_json(response.content)
        
        # Convert to ReAct text format for ReActParser
        assistant_content = self._structured_to_react_text(structured)
        conversation.append_assistant_message(assistant_content)
        
        return conversation
    
    def _structured_to_react_text(self, structured: ReActStructuredResponse) -> str:
        """Convert structured response to ReAct text format for ReActParser."""
        response = structured.response
        
        if isinstance(response, ToolAction):
            action_input_str = "\n".join(f"{k}: {v}" for k, v in response.action_input.items())
            return f"Thought: {response.thought}\n\nAction: {response.action}\nAction Input: {action_input_str}"
        else:
            return f"Thought: {response.thought}\n\nFinal Answer: {response.final_answer}"
```

**Key Design Decision**: The structured response is converted back to ReAct text format and parsed by `ReActParser`. This:
1. Enables consistent unknown tool detection
2. Keeps existing iteration controller logic unchanged
3. Provides a uniform `ReActResponse` for all code paths

### Iteration Controller Integration

**Update**: `backend/tarsy/agents/iteration_controllers/base_controller.py`

The iteration controller uses a unified flow - both structured and text-based responses are parsed by `ReActParser`:

```python
class ReactController(IterationController):
    
    async def execute_analysis_loop(self, context: 'StageContext') -> str:
        """Consolidated ReAct loop with optional structured output support."""
        # ... existing setup ...
        
        for iteration in range(max_iterations):
            # Choose generation method based on provider support
            if self.llm_client.supports_structured_outputs():
                # Structured output - returns LLMConversation with ReAct-formatted text
                conversation_result = await self.llm_client.generate_structured_response(
                    conversation=conversation,
                    session_id=context.session_id,
                    stage_execution_id=context.agent.get_current_stage_execution_id()
                )
            else:
                # Existing text-based flow
                conversation_result = await self.llm_client.generate_response(...)
            
            # UNIFIED: Both paths use ReActParser for consistent validation
            assistant_message = conversation_result.get_latest_assistant_message()
            parsed_response = ReActParser.parse_response(assistant_message.content)
            
            # Rest of loop unchanged - handle parsed_response
            if parsed_response.is_final_answer:
                # ...
            elif parsed_response.has_action:
                # ...
```

**Key Insight**: By converting structured responses to ReAct text format before parsing, we:
1. Keep unknown tool detection working
2. Maintain consistent error handling
3. Avoid duplicating validation logic

### Prompt Adjustments

**Update**: `backend/tarsy/agents/prompts/templates.py`

Add structured output specific instructions:

```python
REACT_STRUCTURED_FORMATTING_INSTRUCTIONS = """You are an SRE agent using the ReAct framework to analyze Kubernetes incidents.

You must respond with a JSON object in one of two formats:

**To use a tool:**
```json
{
  "response": {
    "thought": "Your reasoning about what to investigate",
    "action": "server-name.tool_name",
    "action_input": {"param1": "value1", "param2": "value2"}
  }
}
```

**To provide final analysis:**
```json
{
  "response": {
    "thought": "Your final reasoning",
    "final_answer": "Complete analysis with root cause, resolution, and recommendations"
  }
}
```

CRITICAL RULES:
1. Always respond with valid JSON matching the schema
2. action_input must be a JSON object (not YAML or key:value text)
3. action format must be "server-name.tool_name"
4. Stop after returning tool_action - the system provides observations
5. Use final_answer when you have sufficient information to conclude"""
```

### Schema Validation at API Level

Gemini validates the response against the schema before returning, providing these guarantees:

1. **Type Safety**: All fields have correct types (string, object, etc.)
2. **Required Fields**: Missing fields cause regeneration
3. **Union Type Inference**: Gemini determines if response is ToolAction or FinalAnswer based on field presence

## Migration Strategy

### Phase 1: Add Infrastructure (Non-Breaking)
1. Add `structured_models.py` with Pydantic models
2. Add `use_structured_outputs` config field (default: false)
3. Add `generate_structured_response()` method to LLMClient using `.with_structured_output()`

### Phase 2: Integration (Opt-In)
1. Update ReactController to check for structured output support
2. Add structured output prompt template
3. Enable via provider config for testing

### Phase 3: Validation & Rollout
1. A/B test structured vs text parsing on same prompts
2. Measure: parsing errors, token usage, iteration counts
3. Consider making default for new Gemini providers

## Benefits

| Aspect | Current (Text Parsing) | With Structured Outputs |
|--------|----------------------|------------------------|
| **Parsing Errors** | ~5-10% malformed responses | Near zero (schema-enforced) |
| **Format Corrections** | Required, uses extra tokens | Not needed |
| **Parameter Format** | Multiple formats to parse | JSON only (simpler) |
| **Code Complexity** | 3-tier parser, ~700 lines | Direct Pydantic validation |
| **Token Overhead** | Format instructions + retries | Schema sent once |

## Limitations

1. **Gemini Only**: Native JSON mode is Gemini-specific; other providers use different structured output mechanisms
2. **Strict JSON Parameters**: action_input must be JSON objects, no YAML/key:value flexibility
3. **Schema Size**: JSON schema adds to request payload (minor)

## Dependencies

**No new packages required!** 

Uses LangChain's existing `langchain-google-genai` package with native Gemini JSON mode:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
response = await llm.ainvoke(
    messages,
    generation_config={
        "response_mime_type": "application/json",
        "response_json_schema": ReActStructuredResponse.model_json_schema()
    }
)
```

This is already installed in TARSy's dependencies.

## Testing Strategy

### Unit Tests
```python
def test_structured_response_tool_action():
    """Test ToolAction parsing - Gemini infers type from fields."""
    json_str = '{"response": {"thought": "...", "action": "k8s.pods", "action_input": {}}}'
    result = ReActStructuredResponse.model_validate_json(json_str)
    assert isinstance(result.response, ToolAction)

def test_structured_response_final_answer():
    """Test FinalAnswer parsing - Gemini infers type from fields."""
    json_str = '{"response": {"thought": "...", "final_answer": "..."}}'
    result = ReActStructuredResponse.model_validate_json(json_str)
    assert isinstance(result.response, FinalAnswer)
```

### Integration Tests
```python
async def test_structured_output_investigation():
    """E2E test with structured outputs enabled."""
    # Configure provider with use_structured_outputs=True
    # Submit alert, verify successful investigation
    # Check no format correction retries occurred
```

## Rollback Plan

Since structured outputs are opt-in via configuration:

1. **Immediate**: Set `use_structured_outputs: false` in provider config
2. **Code Rollback**: All existing text parsing code remains unchanged
3. **No Migration**: No database or data format changes required

## Future Considerations

1. **Other Providers**: OpenAI and Anthropic have different structured output mechanisms; could extend support
2. **Schema Evolution**: Adding new response types (e.g., clarification requests) would require schema updates
3. **Hybrid Mode**: Could use structured outputs for tool actions but allow free-form final answers
4. **Streaming**: Investigate if structured outputs work with streaming responses

## References

- [Gemini Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output)
- [Pydantic JSON Schema Generation](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [LangChain Google GenAI Integration](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
- [EP-0014: Assistant Role ReAct Conversations Design](./implemented/EP-0014-assistant-role-react-conversations-design.md)

