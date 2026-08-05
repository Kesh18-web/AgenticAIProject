from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.app.core.logging import logger

import firebase_admin
from firebase_admin import firestore
from google.cloud import firestore as gfirestore


class FirestoreManager:
    """
    Strict Production GCP Firestore Client Manager.
    Does NOT contain local fallbacks. If GCP Cloud Firestore credentials, 
    billing, or connection fail, it raises exceptions loudly so system administrators 
    and developers are notified of the integration failure immediately.
    """

    def __init__(self):
        self.db = None
        self._initialize()

    def _initialize(self):
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            
            # Connect strictly to the custom database 'enterprise-analyst-db'
            self.db = gfirestore.Client(database="enterprise-analyst-db")
            logger.info("Successfully connected to Firestore database 'enterprise-analyst-db' via ADC")
        except Exception as e:
            logger.error(f"Critical Firestore initialization failure: {e}")
            raise e

    def save_document(self, collection_name: str, doc_id: str, data: Dict[str, Any]) -> bool:
        """Save or update a document in Firestore. Raises exceptions on failure."""
        self.db.collection(collection_name).document(doc_id).set(data)
        logger.debug(f"Saved document '{doc_id}' into Firestore collection '{collection_name}'")
        return True

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID. Raises exceptions on failure."""
        doc_snap = self.db.collection(collection_name).document(doc_id).get()
        if doc_snap.exists:
            return doc_snap.to_dict()
        return None

    def save_chat_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Save or update chat session metadata in Firestore. Raises exceptions on failure."""
        self.db.collection("chat_sessions").document(session_id).set(data, merge=True)
        logger.debug(f"Saved chat session '{session_id}' in Firestore")
        return True

    def list_chat_sessions(self) -> List[Dict[str, Any]]:
        """List all persistent chat sessions from Firestore ordered by creation timestamp. Raises exceptions on failure."""
        docs = self.db.collection("chat_sessions").stream()
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            if not data.get("id"):
                data["id"] = doc.id
            sessions.append(data)
        sessions.sort(key=lambda s: s.get("createdAt", 0))
        return sessions

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete chat session and all its messages. Raises exceptions on failure."""
        if not session_id or session_id == "undefined":
            return False
        # Delete session document
        self.db.collection("chat_sessions").document(session_id).delete()
        # Delete messages subcollection documents
        msg_docs = self.db.collection("chat_sessions").document(session_id).collection("messages").stream()
        for doc in msg_docs:
            doc.reference.delete()
        logger.info(f"Deleted chat session '{session_id}' and all messages from Firestore")
        return True

    def save_chat_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Save a message into Firestore subcollection 'chat_sessions/{session_id}/messages'. Raises exceptions on failure."""
        if not session_id or session_id == "undefined":
            return False
        msg_id = message.get("id") or f"msg_{int(datetime.utcnow().timestamp()*1000)}"
        self.db.collection("chat_sessions").document(session_id).collection("messages").document(msg_id).set(message)
        logger.debug(f"Saved message '{msg_id}' to session '{session_id}' in Firestore")
        return True

    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a session from Firestore. Raises exceptions on failure."""
        if not session_id or session_id == "undefined":
            return []
        msg_docs = self.db.collection("chat_sessions").document(session_id).collection("messages").stream()
        messages = [doc.to_dict() for doc in msg_docs]
        messages.sort(key=lambda m: str(m.get("timestamp") or ""))
        return messages


# Global singleton instance
firestore_db = FirestoreManager()
