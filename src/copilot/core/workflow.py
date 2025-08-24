# src/copilot/core/workflow.py
"""
LangGraph workflow for copilot cognitive flow orchestration
"""

import logging
import uuid
from typing import Dict, Any, Optional, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.types import (
    CopilotState,
    CopilotConfig,
    ThoughtType,
    ActionType,
    UserMessage,
    AssistantResponse
)
from ..nodes.cognitive import CognitiveNodes
from ..tools.router import MCPToolRouter
from ..memory.manager import MemoryManager
from ..guardrails.checker import GuardrailChecker
from src.llm import LLMClient
from src.mcp import MCPManager

logger = logging.getLogger(__name__)


class CopilotWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(
        self,
        config: Optional[CopilotConfig] = None,
        llm_client: Optional[LLMClient] = None,
        mcp_manager: Optional[MCPManager] = None
    ):
        self.config = config or CopilotConfig()
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager or MCPManager()
        
        # Initialize components
        self.cognitive_nodes = CognitiveNodes(llm_client)
        self.tool_router = MCPToolRouter(self.mcp_manager, llm_client)
        self.memory_manager = MemoryManager() if self.config.enable_memory else None
        self.guardrail_checker = GuardrailChecker() if self.config.enable_guardrails else None
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        # Initialize LLM client if not provided
        if not self.llm_client:
            from src.llm import get_llm_manager, LLMProvider
            manager = await get_llm_manager()
            self.llm_client = manager.get_client(LLMProvider.ANTHROPIC)
            self.cognitive_nodes.llm_client = self.llm_client
            self.tool_router.llm_client = self.llm_client
        
        # Initialize components
        await self.cognitive_nodes.initialize()
        await self.tool_router.initialize()
        
        if self.memory_manager:
            await self.memory_manager.initialize()
        
        if self.guardrail_checker:
            await self.guardrail_checker.initialize()
        
        self._initialized = True
        logger.info("CopilotWorkflow initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(CopilotState)
        
        # Add nodes
        workflow.add_node("entry", self._entry_node)
        workflow.add_node("understand", self._understand_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("respond", self._respond_node)
        workflow.add_node("guardrails", self._guardrails_node)
        
        # Set entry point
        workflow.set_entry_point("entry")
        
        # Add edges with conditions
        workflow.add_conditional_edges(
            "entry",
            self._route_from_entry,
            {
                "understand": "understand",
                "respond": "respond",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "understand",
            self._route_from_understanding,
            {
                "plan": "plan",
                "respond": "respond",
                "guardrails": "guardrails"
            }
        )
        
        workflow.add_edge("plan", "reason")
        
        workflow.add_conditional_edges(
            "reason",
            self._route_from_reasoning,
            {
                "execute_tools": "execute_tools",
                "respond": "respond",
                "reflect": "reflect"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tools",
            self._route_from_execution,
            {
                "reason": "reason",
                "reflect": "reflect",
                "respond": "respond"
            }
        )
        
        workflow.add_edge("reflect", "decide")
        
        workflow.add_conditional_edges(
            "decide",
            self._route_from_decision,
            {
                "plan": "plan",
                "execute_tools": "execute_tools",
                "respond": "respond"
            }
        )
        
        workflow.add_conditional_edges(
            "guardrails",
            self._route_from_guardrails,
            {
                "understand": "understand",
                "respond": "respond",
                "end": END
            }
        )
        
        workflow.add_edge("respond", END)
        
        return workflow
    
    # Node implementations
    async def _entry_node(self, state: CopilotState) -> CopilotState:
        """Entry point - initial processing"""
        logger.debug("Entering workflow")
        
        # Update available tools
        state.available_tools = self.tool_router.get_available_tools()
        
        # Load relevant memories if memory is enabled
        if self.memory_manager and state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, UserMessage):
                relevant_memories = await self.memory_manager.recall(
                    last_message.content,
                    state
                )
                for memory in relevant_memories:
                    state.add_memory(memory)
        
        return state
    
    async def _understand_node(self, state: CopilotState) -> CopilotState:
        """Understanding node"""
        logger.debug("Understanding user intent")
        return await self.cognitive_nodes.understand(state)
    
    async def _plan_node(self, state: CopilotState) -> CopilotState:
        """Planning node"""
        logger.debug("Creating action plan")
        return await self.cognitive_nodes.plan(state)
    
    async def _reason_node(self, state: CopilotState) -> CopilotState:
        """Reasoning node"""
        logger.debug("Reasoning about next steps")
        return await self.cognitive_nodes.reason(state)
    
    async def _execute_tools_node(self, state: CopilotState) -> CopilotState:
        """Tool execution node"""
        logger.debug("Executing tools")
        
        if state.current_plan:
            state.current_plan.status = "executing"
            state = await self.tool_router.execute_plan_step(state)
        
        return state
    
    async def _reflect_node(self, state: CopilotState) -> CopilotState:
        """Reflection node"""
        logger.debug("Reflecting on experience")
        return await self.cognitive_nodes.reflect(state)
    
    async def _decide_node(self, state: CopilotState) -> CopilotState:
        """Decision node"""
        logger.debug("Making decision")
        return await self.cognitive_nodes.decide(state)
    
    async def _respond_node(self, state: CopilotState) -> CopilotState:
        """Response generation node"""
        logger.debug("Generating response")
        
        # Synthesize response from thoughts and actions
        response_content = self._synthesize_response(state)
        
        response = AssistantResponse(
            content=response_content,
            thoughts=state.thought_history[-5:] if state.thought_history else [],
            actions_taken=state.tool_history[-5:] if state.tool_history else [],
            confidence=self._calculate_confidence(state),
            needs_clarification=self._check_needs_clarification(state)
        )
        
        state.messages.append(response)
        
        # Store interaction in memory
        if self.memory_manager:
            await self.memory_manager.store_interaction(state)
        
        return state
    
    async def _guardrails_node(self, state: CopilotState) -> CopilotState:
        """Guardrails checking node"""
        logger.debug("Checking guardrails")
        
        if self.guardrail_checker:
            checks = await self.guardrail_checker.check_all(state)
            
            # Add any warnings or blocks to state
            for check in checks:
                if not check.passed and check.severity == "error":
                    # Block action and add explanation
                    logger.warning(f"Guardrail blocked action: {check.message}")
                    # Could add to state or trigger specific handling
        
        return state
    
    # Routing functions
    def _route_from_entry(self, state: CopilotState) -> Literal["understand", "respond", "end"]:
        """Route from entry node"""
        if not state.messages:
            return "end"
        
        last_message = state.messages[-1]
        if isinstance(last_message, UserMessage):
            return "understand"
        else:
            return "respond"
    
    def _route_from_understanding(self, state: CopilotState) -> Literal["plan", "respond", "guardrails"]:
        """Route from understanding node"""
        if not state.current_thought:
            return "respond"
        
        # Check if guardrails are needed
        if self.config.enable_guardrails and state.current_thought.confidence < 0.5:
            return "guardrails"
        
        # Check if clarification is needed
        metadata = state.current_thought.metadata
        if metadata.get("needs_clarification", False):
            return "respond"
        
        return "plan"
    
    def _route_from_reasoning(self, state: CopilotState) -> Literal["execute_tools", "respond", "reflect"]:
        """Route from reasoning node"""
        if not state.current_plan:
            return "respond"
        
        if state.current_plan.status == "failed":
            return "reflect"
        
        # If we have a plan that's executing and has steps, execute tools
        if state.current_plan.status == "executing" and state.current_plan.steps:
            return "execute_tools"
        
        if state.current_plan.tool_calls:
            return "execute_tools"
        
        return "respond"
    
    def _route_from_execution(self, state: CopilotState) -> Literal["reason", "reflect", "respond"]:
        """Route from tool execution"""
        if not state.current_plan:
            return "respond"
        
        # Check if any tools failed
        recent_tools = state.tool_history[-3:] if state.tool_history else []
        if any(t.status == "failed" for t in recent_tools):
            # If plan is already failed, stop trying
            if state.current_plan.status == "failed":
                return "respond"
            return "reflect"
        
        # Check if plan is complete or failed
        if state.current_plan.status in ["completed", "failed"]:
            return "respond"
        
        return "reason"
    
    def _route_from_decision(self, state: CopilotState) -> Literal["plan", "execute_tools", "respond"]:
        """Route from decision node"""
        if not state.current_thought:
            return "respond"
        
        decision_content = state.current_thought.content.lower()
        
        if "replan" in decision_content or "new plan" in decision_content:
            return "plan"
        elif "execute" in decision_content or "tool" in decision_content:
            return "execute_tools"
        else:
            return "respond"
    
    def _route_from_guardrails(self, state: CopilotState) -> Literal["understand", "respond", "end"]:
        """Route from guardrails check"""
        # This would check guardrail results and determine next action
        return "respond"
    
    # Helper methods
    def _synthesize_response(self, state: CopilotState) -> str:
        """Synthesize a response from the current state"""
        parts = []
        
        # Check if this is a fast path response
        if state.current_thought and state.current_thought.metadata.get("fast_path", False):
            # Direct fast response
            return state.current_thought.content.replace("Fast response: ", "")
        
        # Add confidence-based preface
        if state.confidence_scores.overall < 0.5:
            parts.append("I'm not entirely certain, but")
        elif state.confidence_scores.overall < 0.7:
            parts.append("Based on my understanding,")
        
        # Add main understanding with hypotheses
        if state.current_hypotheses and state.current_thought:
            best_hypothesis = state.current_hypotheses[0]
            if state.current_thought.type == ThoughtType.UNDERSTANDING:
                parts.append(f"I understand you want to {best_hypothesis.intent}.")
                
                # Show alternatives if confidence is moderate
                if state.current_thought.metadata.get("show_alternatives", False) and len(state.current_hypotheses) > 1:
                    alternatives = state.current_hypotheses[1:3]  # Show top 2 alternatives
                    if alternatives:
                        alt_text = " or ".join([f"{h.intent} ({h.confidence:.0%} confident)" for h in alternatives])
                        parts.append(f"\n\nAlternatively, you might want to {alt_text}.")
                        parts.append("Let me know if I misunderstood.")
        elif state.current_thought:
            if state.current_thought.type == ThoughtType.UNDERSTANDING:
                parts.append(f"I understand you want to {state.current_thought.metadata.get('user_intent', 'help with something')}.")
        
        # Add clarification questions if needed
        if state.confidence_scores.should_clarify() and state.current_thought:
            questions = state.current_thought.metadata.get("clarification_questions", [])
            if questions:
                parts.append("\n\nTo better help you, could you clarify:")
                for q in questions[:2]:  # Limit to 2 questions
                    parts.append(f"- {q}")
        
        # Add plan summary if exists
        if state.current_plan and state.current_plan.status == "completed":
            parts.append(f"\n\nI've completed the task: {state.current_plan.goal}")
        elif state.current_plan and state.current_plan.status == "executing":
            current_step = state.current_plan.current_step
            total_steps = len(state.current_plan.steps)
            parts.append(f"\n\nWorking on step {current_step}/{total_steps}: {state.current_plan.steps[current_step-1] if current_step > 0 else 'Starting...'}")
        
        # Add tool results with more detail
        recent_tools = state.tool_history[-3:] if state.tool_history else []
        successful_tools = [t for t in recent_tools if t.status == "success"]
        if successful_tools:
            parts.append("\n\nCompleted actions:")
            for tool in successful_tools:
                parts.append(f"✓ {tool.tool_name}")
        
        # Default response if nothing specific
        if not parts:
            parts.append("I'm ready to help. What would you like to do?")
        
        return " ".join(parts)
    
    def _calculate_confidence(self, state: CopilotState) -> float:
        """Calculate overall confidence from confidence scores"""
        state.confidence_scores.update_overall()
        return state.confidence_scores.overall
    
    def _check_needs_clarification(self, state: CopilotState) -> bool:
        """Check if clarification is needed based on confidence"""
        # Check confidence-based clarification
        if state.confidence_scores.should_clarify():
            return True
        
        # Check explicit clarification flag
        if state.current_thought:
            metadata = state.current_thought.metadata
            return metadata.get("needs_clarification", False)
        return False
    
    # Public methods
    async def process_message(self, message: str, session_id: Optional[str] = None) -> AssistantResponse:
        """Process a user message through the workflow"""
        await self.initialize()
        
        # Input validation
        if not message or not message.strip():
            return AssistantResponse(
                content="I need a message to process. Please provide your question or request.",
                confidence=0.0,
                needs_clarification=True
            )
        
        # Smart truncation for extremely long messages
        MAX_MESSAGE_LENGTH = 10000
        if len(message) > MAX_MESSAGE_LENGTH:
            # Preserve beginning, middle, and end
            start_size = int(MAX_MESSAGE_LENGTH * 0.4)  # 40% from start
            middle_size = int(MAX_MESSAGE_LENGTH * 0.3)  # 30% from middle
            end_size = int(MAX_MESSAGE_LENGTH * 0.3)  # 30% from end
            
            total_len = len(message)
            middle_start = (total_len // 2) - (middle_size // 2)
            
            truncated_parts = [
                message[:start_size],
                "\n... [content truncated] ...\n",
                message[middle_start:middle_start + middle_size],
                "\n... [content truncated] ...\n",
                message[-end_size:]
            ]
            message = "".join(truncated_parts)
            logger.warning(f"Message truncated from {total_len} to {MAX_MESSAGE_LENGTH} characters (preserved start, middle, end)")
        
        # Create or get session state
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create initial state if new session
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state or create new one
        current_state = self.app.get_state(config)
        if current_state.values is None:
            # New session
            state = CopilotState(session_id=session_id)
        else:
            # Reconstruct state from dict if needed
            state_data = current_state.values
            if isinstance(state_data, dict):
                # Ensure session_id is present
                if 'session_id' not in state_data:
                    state_data['session_id'] = session_id
                state = CopilotState(**state_data)
            else:
                state = state_data
        
        # Add user message
        user_msg = UserMessage(content=message)
        state.messages.append(user_msg)
        
        # Run workflow with increased recursion limit
        config["recursion_limit"] = 50  # Increase from default 25
        result = await self.app.ainvoke(state, config)
        
        # Result is the final state - if it's a dict, reconstruct the state
        if isinstance(result, dict):
            # LangGraph returns dict representation
            final_state = CopilotState(**result)
        else:
            final_state = result
        
        # Get the response
        if final_state.messages:
            last_message = final_state.messages[-1]
            if isinstance(last_message, AssistantResponse):
                return last_message
        
        # Fallback response
        return AssistantResponse(
            content="I processed your message but couldn't generate a proper response.",
            confidence=0.3
        )