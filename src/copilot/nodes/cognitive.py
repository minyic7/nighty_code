# src/copilot/nodes/cognitive.py
"""
Cognitive nodes for the copilot workflow using LLM module
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.llm import (
    get_llm_manager, 
    LLMProvider, 
    Message, 
    MessageRole,
    LLMClient
)
from ..core.types import (
    CopilotState,
    ThoughtProcess,
    ThoughtType,
    ActionPlan,
    ActionType,
    Memory,
    GuardrailCheck,
    Hypothesis,
    ConfidenceScores
)

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs from LLM
class HypothesisOutput(BaseModel):
    """Single hypothesis about user intent"""
    intent: str = Field(description="The hypothesized intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Initial confidence")
    evidence: List[str] = Field(description="Supporting evidence from user input")
    test_approach: str = Field(description="How to test this hypothesis")
    test_cost: float = Field(ge=0.0, le=1.0, description="Cost to test (0=cheap, 1=expensive)")


class MultiHypothesisOutput(BaseModel):
    """Multiple hypotheses from understanding node"""
    hypotheses: List[HypothesisOutput] = Field(description="3-5 possible interpretations")
    primary_intent: str = Field(description="Most likely intent")
    key_entities: List[str] = Field(description="Important entities mentioned")
    context_needed: List[str] = Field(description="Additional context that would be helpful")
    needs_clarification: bool = Field(description="Whether clarification is needed")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions to ask if clarification needed")


class UnderstandingOutput(BaseModel):
    """Output from understanding node"""
    user_intent: str = Field(description="What the user wants to achieve")
    key_entities: List[str] = Field(description="Important entities mentioned")
    context_needed: List[str] = Field(description="Additional context that would be helpful")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in understanding")
    needs_clarification: bool = Field(description="Whether clarification is needed")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions to ask if clarification needed")


class PlanningOutput(BaseModel):
    """Output from planning node"""
    goal: str = Field(description="The main goal to achieve")
    steps: List[str] = Field(description="Steps to achieve the goal")
    tools_needed: List[str] = Field(description="Tools that will be needed")
    reasoning: str = Field(description="Why this plan was chosen")
    alternatives: List[str] = Field(description="Alternative approaches considered")
    estimated_complexity: str = Field(description="simple, medium, or complex")


class ReasoningOutput(BaseModel):
    """Output from reasoning node"""
    analysis: str = Field(description="Analysis of the current situation")
    conclusions: List[str] = Field(description="Conclusions drawn")
    next_action: str = Field(description="Recommended next action")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in reasoning")
    risks: List[str] = Field(default_factory=list, description="Potential risks identified")


class ReflectionOutput(BaseModel):
    """Output from reflection node"""
    what_worked: List[str] = Field(description="Things that worked well")
    what_failed: List[str] = Field(description="Things that didn't work")
    lessons_learned: List[str] = Field(description="Lessons to remember")
    improvements: List[str] = Field(description="Suggestions for improvement")
    should_retry: bool = Field(description="Whether to retry the failed action")


class CognitiveNodes:
    """Cognitive processing nodes using the LLM module"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LLM client if not provided"""
        if not self.llm_client:
            manager = await get_llm_manager()
            self.llm_client = manager.get_client(LLMProvider.ANTHROPIC)
        self._initialized = True
    
    async def ensure_initialized(self):
        """Ensure the node is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def understand(self, state: CopilotState) -> CopilotState:
        """Understanding node - comprehend user intent with multi-hypothesis"""
        await self.ensure_initialized()
        
        # Get the last user message
        last_user_msg = None
        for msg in reversed(state.messages):
            if hasattr(msg, 'content') and not hasattr(msg, 'thoughts'):
                last_user_msg = msg
                break
        
        if not last_user_msg:
            return state
        
        # Check for fast path first
        if self._is_fast_path_query(last_user_msg.content):
            return await self._fast_path_response(state, last_user_msg.content)
        
        # Build context from recent history
        context = self._build_context(state, limit=5)
        
        # Generate multiple hypotheses
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are an AI that generates multiple hypotheses about user intent. "
                "Consider different interpretations and rank them by likelihood."
            ),
            Message(
                MessageRole.USER,
                f"Context from conversation:\n{context}\n\n"
                f"Current user message: {last_user_msg.content}\n\n"
                f"Generate 3-5 hypotheses about what the user wants, ranked by confidence."
            )
        ]
        
        # Get multiple hypotheses
        multi_hypothesis = await self.llm_client.complete(
            messages,
            response_model=MultiHypothesisOutput,
            temperature=0.5
        )
        
        # Convert to internal hypothesis objects
        hypotheses = []
        for h in multi_hypothesis.hypotheses:
            hypothesis = Hypothesis(
                intent=h.intent,
                confidence=h.confidence,
                evidence=h.evidence,
                test_cost=h.test_cost
            )
            hypotheses.append(hypothesis)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        state.current_hypotheses = hypotheses
        
        # Test top hypothesis if confidence < 0.8
        best_hypothesis = hypotheses[0] if hypotheses else None
        if best_hypothesis and best_hypothesis.confidence < 0.8:
            best_hypothesis = await self._test_hypothesis(state, best_hypothesis)
            # Update the hypothesis in the list
            if state.current_hypotheses:
                state.current_hypotheses[0] = best_hypothesis
        
        # Update confidence scores
        state.confidence_scores.understanding = best_hypothesis.confidence if best_hypothesis else 0.5
        
        # Create thought process with hypotheses
        thought = ThoughtProcess(
            type=ThoughtType.UNDERSTANDING,
            content=f"Intent: {multi_hypothesis.primary_intent}",
            confidence=state.confidence_scores.understanding,
            hypotheses=hypotheses,
            metadata={
                "entities": multi_hypothesis.key_entities,
                "context_needed": multi_hypothesis.context_needed,
                "needs_clarification": multi_hypothesis.needs_clarification
            }
        )
        
        state.add_thought(thought)
        
        # Add to memory
        memory = Memory(
            content=f"User wants: {multi_hypothesis.primary_intent}",
            type="conversation",
            importance=0.8
        )
        state.add_memory(memory)
        
        return state
    
    async def plan(self, state: CopilotState) -> CopilotState:
        """Planning node - create action plan"""
        await self.ensure_initialized()
        
        # Get current understanding
        understanding = self._get_last_thought_of_type(state, ThoughtType.UNDERSTANDING)
        if not understanding:
            return state
        
        # Get available tools
        tools_description = self._format_available_tools(state)
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a strategic planner. Create actionable plans to achieve goals. "
                "Consider available tools and break down complex tasks into steps."
            ),
            Message(
                MessageRole.USER,
                f"Goal: {understanding.content}\n\n"
                f"Available tools:\n{tools_description}\n\n"
                f"Create a plan to achieve this goal."
            )
        ]
        
        # Get structured plan
        plan_output = await self.llm_client.complete(
            messages,
            response_model=PlanningOutput,
            temperature=0.5
        )
        
        # Create action plan
        plan = ActionPlan(
            goal=plan_output.goal,
            steps=plan_output.steps,
            reasoning=plan_output.reasoning,
            alternatives=plan_output.alternatives,
            status="executing"  # Mark as ready to execute
        )
        
        state.update_plan(plan)
        
        # Update planning confidence based on complexity
        complexity_confidence = {
            "simple": 0.9,
            "medium": 0.7,
            "complex": 0.5
        }
        plan_confidence = complexity_confidence.get(plan_output.estimated_complexity, 0.7)
        state.confidence_scores.planning = plan_confidence
        
        # Create thought
        thought = ThoughtProcess(
            type=ThoughtType.PLANNING,
            content=f"Created {len(plan.steps)}-step plan: {plan.goal}",
            confidence=plan_confidence,
            metadata={"complexity": plan_output.estimated_complexity}
        )
        state.add_thought(thought)
        
        return state
    
    async def reason(self, state: CopilotState) -> CopilotState:
        """Reasoning node - analyze and make decisions"""
        await self.ensure_initialized()
        
        # Get current context
        current_plan = state.current_plan
        recent_tools = state.tool_history[-3:] if state.tool_history else []
        
        context = {
            "current_goal": current_plan.goal if current_plan else "No active goal",
            "recent_actions": [t.tool_name for t in recent_tools],
            "recent_results": [t.result for t in recent_tools if t.result]
        }
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a logical reasoner. Analyze situations and recommend next actions."
            ),
            Message(
                MessageRole.USER,
                f"Current situation:\n{context}\n\n"
                "Analyze and recommend the next action."
            )
        ]
        
        reasoning = await self.llm_client.complete(
            messages,
            response_model=ReasoningOutput,
            temperature=0.4
        )
        
        thought = ThoughtProcess(
            type=ThoughtType.REASONING,
            content=reasoning.analysis,
            confidence=reasoning.confidence,
            metadata={
                "next_action": reasoning.next_action,
                "risks": reasoning.risks
            }
        )
        state.add_thought(thought)
        
        return state
    
    async def reflect(self, state: CopilotState) -> CopilotState:
        """Reflection node - learn from experience"""
        await self.ensure_initialized()
        
        # Get recent history for reflection
        recent_plan = state.plan_history[-1] if state.plan_history else state.current_plan
        recent_tools = state.tool_history[-5:] if state.tool_history else []
        
        if not recent_plan and not recent_tools:
            return state
        
        # Analyze what happened
        execution_summary = {
            "plan": recent_plan.goal if recent_plan else "No plan",
            "tools_used": [t.tool_name for t in recent_tools],
            "successes": [t for t in recent_tools if t.status == "success"],
            "failures": [t for t in recent_tools if t.status == "failed"]
        }
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a reflective learner. Analyze past actions and extract lessons."
            ),
            Message(
                MessageRole.USER,
                f"Execution summary:\n{execution_summary}\n\n"
                "Reflect on what happened and what can be learned."
            )
        ]
        
        reflection = await self.llm_client.complete(
            messages,
            response_model=ReflectionOutput,
            temperature=0.6
        )
        
        # Create thought
        thought = ThoughtProcess(
            type=ThoughtType.REFLECTION,
            content=f"Reflected on execution: {len(reflection.lessons_learned)} lessons learned",
            confidence=0.9,
            metadata={
                "lessons": reflection.lessons_learned,
                "improvements": reflection.improvements
            }
        )
        state.add_thought(thought)
        
        # Store important lessons in long-term memory
        for lesson in reflection.lessons_learned:
            memory = Memory(
                content=lesson,
                type="learning",
                importance=0.8
            )
            state.add_memory(memory, long_term=True)
        
        return state
    
    async def decide(self, state: CopilotState) -> CopilotState:
        """Decision node - make final decisions on actions"""
        await self.ensure_initialized()
        
        # Gather all recent thoughts
        recent_thoughts = state.thought_history[-5:] if state.thought_history else []
        current_plan = state.current_plan
        
        if not recent_thoughts:
            return state
        
        # Synthesize thoughts into decision
        thought_summary = "\n".join([
            f"- {t.type.value}: {t.content} (confidence: {t.confidence})"
            for t in recent_thoughts
        ])
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a decision maker. Based on analysis, make clear decisions."
            ),
            Message(
                MessageRole.USER,
                f"Recent analysis:\n{thought_summary}\n\n"
                f"Current plan: {current_plan.goal if current_plan else 'None'}\n\n"
                "What is the decision on how to proceed?"
            )
        ]
        
        response = await self.llm_client.complete(
            messages,
            temperature=0.3,
            max_tokens=500
        )
        
        thought = ThoughtProcess(
            type=ThoughtType.DECISION,
            content=response.content,
            confidence=0.85
        )
        state.add_thought(thought)
        
        return state
    
    def _build_context(self, state: CopilotState, limit: int = 5) -> str:
        """Build context from recent messages"""
        recent_messages = state.messages[-limit:] if len(state.messages) > limit else state.messages
        context_parts = []
        
        for msg in recent_messages:
            if hasattr(msg, 'thoughts'):  # Assistant message
                context_parts.append(f"Assistant: {msg.content}")
            else:  # User message
                context_parts.append(f"User: {msg.content}")
        
        return "\n".join(context_parts)
    
    def _get_last_thought_of_type(self, state: CopilotState, thought_type: ThoughtType) -> Optional[ThoughtProcess]:
        """Get the most recent thought of a specific type"""
        for thought in reversed(state.thought_history):
            if thought.type == thought_type:
                return thought
        return None
    
    def _format_available_tools(self, state: CopilotState) -> str:
        """Format available tools for context"""
        if not state.available_tools:
            return "No tools available"
        
        tool_list = []
        for server, tools in state.available_tools.items():
            tool_list.append(f"{server}: {', '.join(tools)}")
        
        return "\n".join(tool_list)
    
    def _is_fast_path_query(self, message: str) -> bool:
        """Check if query qualifies for fast path"""
        # Fast path patterns
        fast_patterns = [
            # Simple questions
            ("what is", 0.9),
            ("how to", 0.8),
            ("when does", 0.9),
            ("where is", 0.9),
            
            # Direct commands
            ("list files", 0.95),
            ("read file", 0.95),
            ("show me", 0.85),
            ("open", 0.9),
            
            # Simple calculations
            ("calculate", 0.9),
            ("convert", 0.9),
            
            # Status checks
            ("status of", 0.95),
            ("check if", 0.9),
        ]
        
        message_lower = message.lower().strip()
        
        # Check for fast patterns
        for pattern, confidence in fast_patterns:
            if pattern in message_lower:
                # Additional checks for complexity
                word_count = len(message.split())
                has_multiple_clauses = ' and ' in message_lower or ' but ' in message_lower
                
                # Simple queries are typically short and single-clause
                if word_count < 15 and not has_multiple_clauses:
                    return True
        
        return False
    
    async def _fast_path_response(self, state: CopilotState, message: str) -> CopilotState:
        """Generate fast response for simple queries"""
        await self.ensure_initialized()
        
        # Quick intent extraction
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a quick responder. Extract the simple intent and respond directly."
            ),
            Message(
                MessageRole.USER,
                f"User query: {message}\n\n"
                "What is the user asking for? Be very concise."
            )
        ]
        
        response = await self.llm_client.complete(
            messages,
            temperature=0.3,
            max_tokens=200
        )
        
        # Create fast response thought
        thought = ThoughtProcess(
            type=ThoughtType.FAST_RESPONSE,
            content=f"Fast response: {response.content}",
            confidence=0.95,
            metadata={"fast_path": True}
        )
        state.add_thought(thought)
        
        # High confidence for fast path
        state.confidence_scores.understanding = 0.95
        state.confidence_scores.execution = 0.95
        state.confidence_scores.update_overall()
        
        return state
    
    async def _test_hypothesis(self, state: CopilotState, hypothesis: Hypothesis) -> Hypothesis:
        """Test a hypothesis to refine confidence"""
        await self.ensure_initialized()
        
        # Don't test if cost is too high or already tested
        if hypothesis.test_cost > 0.7 or hypothesis.tested:
            return hypothesis
        
        # Build test prompt based on hypothesis
        test_prompt = self._build_hypothesis_test(hypothesis, state)
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are testing a hypothesis about user intent. "
                "Based on the evidence, determine if the hypothesis is likely correct."
            ),
            Message(
                MessageRole.USER,
                test_prompt
            )
        ]
        
        # Test with structured output
        class HypothesisTestResult(BaseModel):
            is_likely_correct: bool = Field(description="Whether hypothesis seems correct")
            confidence_adjustment: float = Field(
                ge=-0.5, le=0.5,
                description="How much to adjust confidence (-0.5 to +0.5)"
            )
            additional_evidence: List[str] = Field(
                default_factory=list,
                description="New evidence found"
            )
            reason: str = Field(description="Reasoning for the judgment")
        
        test_result = await self.llm_client.complete(
            messages,
            response_model=HypothesisTestResult,
            temperature=0.4
        )
        
        # Update hypothesis
        hypothesis.tested = True
        hypothesis.confidence = max(0.0, min(1.0, 
            hypothesis.confidence + test_result.confidence_adjustment
        ))
        hypothesis.evidence.extend(test_result.additional_evidence)
        hypothesis.test_result = hypothesis.confidence
        
        return hypothesis
    
    def _build_hypothesis_test(self, hypothesis: Hypothesis, state: CopilotState) -> str:
        """Build test prompt for hypothesis"""
        # Get recent context
        context = self._build_context(state, limit=3)
        
        # Get available tools for testing
        tools = self._format_available_tools(state)
        
        prompt = f"""Hypothesis to test:
        Intent: {hypothesis.intent}
        Current confidence: {hypothesis.confidence:.1%}
        Evidence so far: {', '.join(hypothesis.evidence)}
        
        Conversation context:
        {context}
        
        Available tools that could help verify:
        {tools}
        
        Based on the context and available tools, is this hypothesis likely correct?
        Consider what the user is really asking for.
        """
        
        return prompt