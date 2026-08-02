from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.logging import logger
from backend.app.db.firestore import firestore_db

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    id: str
    name: str = "New Chat"
    searchScope: str = "session"
    attachedFiles: List[str] = []


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    searchScope: Optional[str] = None
    attachedFiles: Optional[List[str]] = None


@router.get("")
async def list_sessions():
    """Fetch all active chat sessions from GCP Firestore."""
    try:
        sessions = firestore_db.list_chat_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_session(req: CreateSessionRequest):
    """Create a new persistent chat session record in Firestore."""
    try:
        session_data = {
            "id": req.id,
            "name": req.name,
            "createdAt": int(datetime.utcnow().timestamp() * 1000),
            "searchScope": req.searchScope,
            "attachedFiles": req.attachedFiles,
        }
        firestore_db.save_chat_session(req.id, session_data)
        return {"status": "success", "session": session_data}
    except Exception as e:
        logger.error(f"Error creating session [{req.id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Fetch message history for a specific session from Firestore."""
    try:
        messages = firestore_db.get_chat_messages(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error fetching messages for session [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/messages")
async def add_session_message(session_id: str, message: Dict[str, Any]):
    """Add a user or assistant message to a session in Firestore."""
    try:
        firestore_db.save_chat_message(session_id, message)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding message to session [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest):
    """Update session metadata (title, searchScope, attachedFiles) in Firestore."""
    try:
        update_data = {}
        if req.name is not None:
            update_data["name"] = req.name
        if req.searchScope is not None:
            update_data["searchScope"] = req.searchScope
        if req.attachedFiles is not None:
            update_data["attachedFiles"] = req.attachedFiles

        if update_data:
            firestore_db.save_chat_session(session_id, update_data)
        return {"status": "success", "updated": update_data}
    except Exception as e:
        logger.error(f"Error updating session [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete session and all its messages from Firestore."""
    try:
        firestore_db.delete_chat_session(session_id)
        return {"status": "success", "deleted_session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting session [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
