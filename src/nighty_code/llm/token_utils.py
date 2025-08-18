"""
Token counting and management utilities.
"""

from typing import List, Dict, Optional, Union, Tuple
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """Handles token counting for different models."""
    
    # Token encoding cache
    _encoders = {}
    
    # Model to encoder mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "code-davinci-002": "p50k_base",
    }
    
    # Approximate tokens per message overhead
    MESSAGE_OVERHEAD = {
        "gpt-4": 3,
        "gpt-3.5-turbo": 4,
    }
    
    @classmethod
    def get_encoder(cls, model: str):
        """Get the appropriate encoder for a model."""
        
        # Check cache first
        if model in cls._encoders:
            return cls._encoders[model]
        
        # Determine encoding
        if model in cls.MODEL_ENCODINGS:
            encoding_name = cls.MODEL_ENCODINGS[model]
        else:
            # Try to get encoding by model name
            try:
                encoding = tiktoken.encoding_for_model(model)
                cls._encoders[model] = encoding
                return encoding
            except KeyError:
                # Default to cl100k_base for unknown models
                logger.warning(f"Unknown model {model}, using cl100k_base encoding")
                encoding_name = "cl100k_base"
        
        # Get encoding
        encoding = tiktoken.get_encoding(encoding_name)
        cls._encoders[model] = encoding
        return encoding
    
    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            model: Model name for encoding selection
            
        Returns:
            Number of tokens
        """
        
        encoder = cls.get_encoder(model)
        return len(encoder.encode(text))
    
    @classmethod
    def count_messages_tokens(
        cls, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in a list of messages (for chat models).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name
            
        Returns:
            Total number of tokens
        """
        
        encoder = cls.get_encoder(model)
        
        # Get per-message overhead
        if model in cls.MESSAGE_OVERHEAD:
            per_message = cls.MESSAGE_OVERHEAD[model]
        else:
            per_message = 3  # Default overhead
        
        total_tokens = 0
        
        for message in messages:
            total_tokens += per_message  # Message overhead
            
            # Count role tokens
            if "role" in message:
                total_tokens += len(encoder.encode(message["role"]))
            
            # Count content tokens
            if "content" in message:
                total_tokens += len(encoder.encode(message["content"]))
            
            # Count name tokens if present
            if "name" in message:
                total_tokens += len(encoder.encode(message["name"]))
                total_tokens += 1  # Extra token for name
        
        total_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        
        return total_tokens
    
    @classmethod
    def truncate_text(
        cls, 
        text: str, 
        max_tokens: int, 
        model: str = "gpt-3.5-turbo",
        from_end: bool = False
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            model: Model name for encoding
            from_end: If True, truncate from end instead of beginning
            
        Returns:
            Truncated text
        """
        
        encoder = cls.get_encoder(model)
        tokens = encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]
        
        return encoder.decode(truncated_tokens)
    
    @classmethod
    def split_text_by_tokens(
        cls,
        text: str,
        chunk_size: int,
        overlap: int = 0,
        model: str = "gpt-3.5-turbo"
    ) -> List[str]:
        """
        Split text into chunks by token count.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            model: Model name for encoding
            
        Returns:
            List of text chunks
        """
        
        encoder = cls.get_encoder(model)
        tokens = encoder.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(encoder.decode(chunk_tokens))
            
            # Move start position
            if overlap > 0 and end < len(tokens):
                start = end - overlap
            else:
                start = end
        
        return chunks
    
    @classmethod
    def estimate_messages_cost(
        cls,
        messages: List[Dict[str, str]],
        model: str,
        input_cost_per_1k: float,
        output_tokens: Optional[int] = None,
        output_cost_per_1k: float = 0.0
    ) -> float:
        """
        Estimate cost for messages.
        
        Args:
            messages: Input messages
            model: Model name
            input_cost_per_1k: Cost per 1000 input tokens
            output_tokens: Expected output tokens (optional)
            output_cost_per_1k: Cost per 1000 output tokens
            
        Returns:
            Estimated cost in dollars
        """
        
        input_tokens = cls.count_messages_tokens(messages, model)
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        
        output_cost = 0.0
        if output_tokens and output_cost_per_1k:
            output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost


class ResponseMerger:
    """Handles merging of multiple response chunks."""
    
    @staticmethod
    def merge_text_responses(responses: List[str]) -> str:
        """
        Merge multiple text responses.
        
        Args:
            responses: List of response chunks
            
        Returns:
            Merged response
        """
        
        if not responses:
            return ""
        
        # Simple concatenation for text
        return "".join(responses)
    
    @staticmethod
    def merge_json_responses(responses: List[Dict]) -> Dict:
        """
        Merge multiple JSON responses.
        
        Args:
            responses: List of JSON response chunks
            
        Returns:
            Merged JSON response
        """
        
        if not responses:
            return {}
        
        if len(responses) == 1:
            return responses[0]
        
        # For JSON, we need to be smarter about merging
        # This is a simple implementation - can be enhanced
        merged = {}
        
        for response in responses:
            if isinstance(response, dict):
                # Merge dictionaries
                for key, value in response.items():
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        # Extend lists
                        merged[key].extend(value)
                    elif isinstance(value, dict) and isinstance(merged[key], dict):
                        # Merge nested dicts
                        merged[key].update(value)
                    # Otherwise keep first value
        
        return merged
    
    @staticmethod
    def detect_truncation(response: str) -> bool:
        """
        Detect if a response was truncated.
        
        Args:
            response: Response text
            
        Returns:
            True if response appears truncated
        """
        
        # Common truncation indicators
        truncation_markers = [
            "...",
            "[truncated]",
            "[continued",
            "```",  # Unclosed code block
        ]
        
        # Check for incomplete sentences
        if response and not response.rstrip().endswith(('.', '!', '?', '"', "'", ')', ']', '}')):
            return True
        
        # Check for truncation markers
        last_line = response.rstrip().split('\n')[-1] if response else ""
        for marker in truncation_markers:
            if last_line.endswith(marker):
                return True
        
        # Check for incomplete code blocks
        code_blocks = response.count("```")
        if code_blocks % 2 != 0:
            return True
        
        # Check for incomplete brackets/parens
        open_brackets = response.count('[') - response.count(']')
        open_parens = response.count('(') - response.count(')')
        open_braces = response.count('{') - response.count('}')
        
        if open_brackets > 0 or open_parens > 0 or open_braces > 0:
            return True
        
        return False
    
    @staticmethod
    def create_continuation_prompt(
        original_prompt: str,
        partial_response: str,
        continuation_instruction: str = "Continue from where you left off:"
    ) -> str:
        """
        Create a prompt to continue generation.
        
        Args:
            original_prompt: Original user prompt
            partial_response: Partial response received
            continuation_instruction: Instruction for continuation
            
        Returns:
            Continuation prompt
        """
        
        # Find natural break point
        break_point = ResponseMerger.find_break_point(partial_response)
        
        if break_point:
            context = partial_response[:break_point]
            incomplete = partial_response[break_point:]
        else:
            context = partial_response
            incomplete = ""
        
        prompt = f"""Original request: {original_prompt}

Previous response (truncated):
{context}

{continuation_instruction}
{incomplete}"""
        
        return prompt
    
    @staticmethod
    def find_break_point(text: str, lookback: int = 500) -> Optional[int]:
        """
        Find a natural break point in text.
        
        Args:
            text: Text to find break point in
            lookback: How far to look back for break point
            
        Returns:
            Index of break point or None
        """
        
        if len(text) < lookback:
            return None
        
        # Look for natural break points
        search_start = max(0, len(text) - lookback)
        search_text = text[search_start:]
        
        # Priority order for break points
        break_sequences = [
            "\n\n",  # Paragraph break
            ".\n",   # Sentence end with newline
            ". ",    # Sentence end
            "\n",    # Line break
            ", ",    # Comma
            " ",     # Space
        ]
        
        for sequence in break_sequences:
            pos = search_text.rfind(sequence)
            if pos != -1:
                return search_start + pos + len(sequence)
        
        # No good break point found
        return len(text) - 100  # Just take last 100 chars as incomplete