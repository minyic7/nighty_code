"""
Structured LLM Client for Pydantic model generation.

This client ensures LLM outputs conform to Pydantic models,
providing type safety and validation throughout the system.
"""

import json
from typing import Type, TypeVar, Optional, Any, Dict, List
from pydantic import BaseModel, ValidationError
import logging

from .client import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredLLMClient:
    """
    LLM client that returns structured Pydantic models.
    
    Ensures type safety and handles validation/retry logic.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize with an LLM client.
        
        Args:
            llm_client: Base LLM client, None for no LLM
        """
        self.llm_client = llm_client
    
    def generate(
        self,
        prompt: str,
        model_class: Type[T],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ) -> T:
        """
        Generate a Pydantic model instance from LLM.
        
        Args:
            prompt: The prompt to send to LLM
            model_class: The Pydantic model class to generate
            examples: Optional examples to include
            max_retries: Number of retries on validation failure
            temperature: LLM temperature
            
        Returns:
            Instance of model_class with validated data
            
        Raises:
            ValidationError: If LLM output doesn't match schema after retries
        """
        # Build the full prompt with schema
        full_prompt = self._build_prompt(prompt, model_class, examples)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # Get LLM response
                response = self.llm_client.complete(
                    full_prompt,
                    temperature=temperature,
                    max_tokens=2000
                )
                
                # Parse and validate
                model_instance = self._parse_response(response, model_class)
                return model_instance
                
            except ValidationError as e:
                last_error = e
                logger.warning(f"Validation failed (attempt {attempt + 1}): {e}")
                
                # Add error to prompt for retry
                if attempt < max_retries - 1:
                    full_prompt = self._add_error_context(
                        full_prompt, 
                        response, 
                        str(e)
                    )
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")
                
                # Try to fix JSON
                if attempt < max_retries - 1:
                    full_prompt = self._add_json_error_context(
                        full_prompt,
                        response,
                        str(e)
                    )
        
        # All retries failed
        raise ValidationError(f"Failed after {max_retries} attempts: {last_error}")
    
    def _build_prompt(
        self,
        prompt: str,
        model_class: Type[T],
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build prompt with Pydantic schema."""
        schema = model_class.model_json_schema()
        schema_json = json.dumps(schema, indent=2)
        
        full_prompt = f"""{prompt}

You must respond with valid JSON that conforms to this Pydantic model schema:

```json
{schema_json}
```

Important:
1. Your response must be ONLY valid JSON
2. Do not include any explanation before or after the JSON
3. Ensure all required fields are present
4. Use the exact field names from the schema
5. Follow the type requirements for each field
"""
        
        if examples:
            full_prompt += "\n\nHere are some valid examples:\n"
            for i, example in enumerate(examples, 1):
                full_prompt += f"\nExample {i}:\n```json\n{json.dumps(example, indent=2)}\n```\n"
        
        full_prompt += "\n\nYour JSON response:"
        
        return full_prompt
    
    def _parse_response(self, response: str, model_class: Type[T]) -> T:
        """Parse LLM response into Pydantic model."""
        # Handle both string and LLMResponse objects
        if hasattr(response, 'content'):
            response = response.content
        
        # Try to extract JSON from response
        response = str(response).strip()
        
        # Handle common LLM response patterns
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Clean up the response
        response = response.strip()
        
        # Find JSON in response (LLM might add text before/after)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
        else:
            json_str = response
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Create and validate model
        return model_class(**data)
    
    def _add_error_context(self, prompt: str, response: str, error: str) -> str:
        """Add validation error context for retry."""
        return f"""{prompt}

Your previous response had validation errors:
```
{error}
```

Your previous response was:
```
{response}
```

Please fix the validation errors and provide a corrected JSON response:"""
    
    def _add_json_error_context(self, prompt: str, response: str, error: str) -> str:
        """Add JSON error context for retry."""
        return f"""{prompt}

Your previous response was not valid JSON:
```
{error}
```

Your previous response was:
```
{response}
```

Please provide a valid JSON response:"""
    
    def analyze_with_reasoning(
        self,
        prompt: str,
        model_class: Type[T],
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Generate a model with explicit reasoning.
        
        Useful for debugging and transparency.
        """
        reasoning_prompt = f"""Think step by step about this request:

{prompt}
"""
        
        if context:
            reasoning_prompt += f"\nContext:\n{json.dumps(context, indent=2)}\n"
        
        reasoning_prompt += """
First, explain your reasoning about:
1. What is being asked
2. What information is relevant
3. How you'll structure your response

Then provide your response as JSON.

Remember: The final JSON must conform to the schema provided.
"""
        
        return self.generate(reasoning_prompt, model_class)
    
    def batch_generate(
        self,
        prompts: List[str],
        model_class: Type[T]
    ) -> List[T]:
        """
        Generate multiple model instances.
        
        Useful for processing multiple queries.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, model_class)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                # Could append None or a default instance
                continue
        
        return results