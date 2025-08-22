"""
Session management for Copilot conversations.

Handles session lifecycle, persistence, and recovery.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CRASHED = "crashed"
    RECOVERED = "recovered"


@dataclass
class SessionMetadata:
    """Metadata for a conversation session."""
    session_id: str
    project_path: str
    persona_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SessionStatus = SessionStatus.ACTIVE
    message_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "persona_type": self.persona_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "tags": self.tags,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            project_path=data["project_path"],
            persona_type=data["persona_type"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=SessionStatus(data.get("status", "active")),
            message_count=data.get("message_count", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            tags=data.get("tags", []),
            notes=data.get("notes", "")
        )


class SessionManager:
    """
    Manages conversation sessions with persistence and recovery.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            base_dir: Base directory for session storage
        """
        self.base_dir = base_dir or Path.home() / ".copilot" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_session: Optional[SessionMetadata] = None
        self.session_history: List[SessionMetadata] = []
        
        # Load session history
        self._load_session_history()
    
    def create_session(
        self,
        project_path: str,
        persona_type: str = "default"
    ) -> SessionMetadata:
        """
        Create a new session.
        
        Args:
            project_path: Path to the project
            persona_type: Type of persona being used
            
        Returns:
            New session metadata
        """
        # End previous session if active
        if self.active_session and self.active_session.status == SessionStatus.ACTIVE:
            self.end_session()
        
        # Create new session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        session = SessionMetadata(
            session_id=session_id,
            project_path=project_path,
            persona_type=persona_type,
            start_time=datetime.now()
        )
        
        self.active_session = session
        self.session_history.append(session)
        
        # Create session directory
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save initial metadata
        self._save_session_metadata(session)
        
        logger.info(f"Created new session: {session_id}")
        return session
    
    def end_session(
        self,
        status: SessionStatus = SessionStatus.COMPLETED,
        notes: str = ""
    ) -> Optional[SessionMetadata]:
        """
        End the current session.
        
        Args:
            status: Final status of the session
            notes: Optional notes about the session
            
        Returns:
            Ended session metadata
        """
        if not self.active_session:
            return None
        
        self.active_session.end_time = datetime.now()
        self.active_session.status = status
        if notes:
            self.active_session.notes = notes
        
        # Save final metadata
        self._save_session_metadata(self.active_session)
        
        logger.info(f"Ended session: {self.active_session.session_id} with status: {status.value}")
        
        session = self.active_session
        self.active_session = None
        return session
    
    def update_session_stats(
        self,
        message_count_delta: int = 0,
        token_count_delta: int = 0,
        cost_delta: float = 0.0
    ):
        """
        Update active session statistics.
        
        Args:
            message_count_delta: Messages to add
            token_count_delta: Tokens to add
            cost_delta: Cost to add
        """
        if not self.active_session:
            return
        
        self.active_session.message_count += message_count_delta
        self.active_session.total_tokens += token_count_delta
        self.active_session.total_cost += cost_delta
        
        # Periodically save metadata
        if self.active_session.message_count % 5 == 0:
            self._save_session_metadata(self.active_session)
    
    def get_session_path(self, session_id: Optional[str] = None) -> Path:
        """
        Get path for a session.
        
        Args:
            session_id: Session ID or None for active session
            
        Returns:
            Path to session directory
        """
        if session_id is None and self.active_session:
            session_id = self.active_session.session_id
        
        if session_id:
            return self.base_dir / session_id
        
        raise ValueError("No session ID provided and no active session")
    
    def save_conversation(self, conversation_data: Dict[str, Any]):
        """
        Save conversation data for the active session.
        
        Args:
            conversation_data: Data to save
        """
        if not self.active_session:
            logger.warning("No active session to save conversation")
            return
        
        session_path = self.get_session_path()
        conversation_file = session_path / "conversation.json"
        
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.debug(f"Saved conversation to: {conversation_file}")
    
    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation data from a session.
        
        Args:
            session_id: Session to load from
            
        Returns:
            Conversation data or None
        """
        session_path = self.base_dir / session_id
        conversation_file = session_path / "conversation.json"
        
        if conversation_file.exists():
            with open(conversation_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def recover_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Recover a crashed or interrupted session.
        
        Args:
            session_id: Session to recover
            
        Returns:
            Recovered session metadata
        """
        # Load session metadata
        metadata_file = self.base_dir / session_id / "metadata.json"
        
        if not metadata_file.exists():
            logger.error(f"Session not found: {session_id}")
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        session = SessionMetadata.from_dict(data)
        session.status = SessionStatus.RECOVERED
        
        self.active_session = session
        logger.info(f"Recovered session: {session_id}")
        
        return session
    
    def list_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10
    ) -> List[SessionMetadata]:
        """
        List recent sessions.
        
        Args:
            project_path: Filter by project path
            limit: Maximum number of sessions
            
        Returns:
            List of session metadata
        """
        sessions = self.session_history
        
        if project_path:
            sessions = [s for s in sessions if s.project_path == project_path]
        
        # Sort by start time (most recent first)
        sessions.sort(key=lambda x: x.start_time, reverse=True)
        
        return sessions[:limit]
    
    def cleanup_old_sessions(self, days: int = 30):
        """
        Clean up sessions older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                start_time = datetime.fromisoformat(data["start_time"])
                if start_time < cutoff_date:
                    # Remove old session
                    import shutil
                    shutil.rmtree(session_dir)
                    logger.info(f"Cleaned up old session: {session_dir.name}")
    
    def _save_session_metadata(self, session: SessionMetadata):
        """Save session metadata to file."""
        session_path = self.base_dir / session.session_id
        session_path.mkdir(exist_ok=True)
        
        metadata_file = session_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def _load_session_history(self):
        """Load session history from disk."""
        self.session_history = []
        
        if not self.base_dir.exists():
            return
        
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    session = SessionMetadata.from_dict(data)
                    self.session_history.append(session)
                except Exception as e:
                    logger.warning(f"Failed to load session {session_dir.name}: {e}")
        
        logger.info(f"Loaded {len(self.session_history)} sessions from history")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all sessions."""
        total_sessions = len(self.session_history)
        active_sessions = sum(1 for s in self.session_history if s.status == SessionStatus.ACTIVE)
        completed_sessions = sum(1 for s in self.session_history if s.status == SessionStatus.COMPLETED)
        
        total_messages = sum(s.message_count for s in self.session_history)
        total_tokens = sum(s.total_tokens for s in self.session_history)
        total_cost = sum(s.total_cost for s in self.session_history)
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "sessions_by_project": self._get_sessions_by_project()
        }
    
    def _get_sessions_by_project(self) -> Dict[str, int]:
        """Get count of sessions by project."""
        project_counts = {}
        for session in self.session_history:
            project = Path(session.project_path).name
            project_counts[project] = project_counts.get(project, 0) + 1
        return project_counts