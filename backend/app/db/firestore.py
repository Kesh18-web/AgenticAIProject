import os
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud import firestore as gfirestore

    HAS_FIREBASE_LIB = True
except ImportError:
    HAS_FIREBASE_LIB = False


class MockFirestoreDB:
    """In-memory dictionary fallback when Firebase credentials are not yet configured."""

    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {}
        logger.warning(
            "Using MockFirestoreDB (In-Memory). Provide valid FIREBASE_CREDENTIALS_PATH for production persistence."
        )

    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = {}
        return MockCollection(self._collections[name])


class MockCollection:

    def __init__(self, store: Dict[str, Any]):
        self._store = store

    def document(self, doc_id: str):
        return MockDocumentRef(self._store, doc_id)


class MockDocumentRef:

    def __init__(self, store: Dict[str, Any], doc_id: str):
        self._store = store
        self._doc_id = doc_id

    def set(self, data: Dict[str, Any]):
        self._store[self._doc_id] = data
        return True

    def get(self):
        data = self._store.get(self._doc_id)

        class Snap:

            def __init__(self, exists, val):
                self.exists = exists
                self._val = val

            def to_dict(self):
                return self._val

        return Snap(data is not None, data)


class FirestoreManager:
    """Firestore Client Manager handling live Firebase Firestore and local mock fallback."""

    def __init__(self):
        self.db = None
        self.is_mock = True
        self._initialize()

    def _initialize(self):
        if HAS_FIREBASE_LIB:
            try:
                # Initialize Firebase App using Application Default Credentials (ADC)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app()
                
                # Connect to the custom database 'enterprise-analyst-db' using google-cloud-firestore Client
                self.db = gfirestore.Client(database="enterprise-analyst-db")
                self.is_mock = False
                logger.info(
                    "Successfully connected to Firestore database 'enterprise-analyst-db' using Application Default Credentials (ADC)"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize Firestore via ADC: {e}. Falling back to MockFirestore."
                )
                self.db = MockFirestoreDB()
        else:
            self.db = MockFirestoreDB()

    def save_document(
        self, collection_name: str, doc_id: str, data: Dict[str, Any]
    ) -> bool:
        """Save or update a document in Firestore."""
        try:
            self.db.collection(collection_name).document(doc_id).set(data)
            logger.debug(
                f"Saved document '{doc_id}' into Firestore collection '{collection_name}'"
            )
            return True
        except Exception as e:
            logger.error(f"Error writing to Firestore [{collection_name}/{doc_id}]: {e}")
            return False

    def get_document(
        self, collection_name: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document from Firestore by ID."""
        try:
            doc_ref = self.db.collection(collection_name).document(doc_id)
            doc_snap = doc_ref.get()
            if doc_snap.exists:
                return doc_snap.to_dict()
            return None
        except Exception as e:
            logger.error(
                f"Error reading from Firestore [{collection_name}/{doc_id}]: {e}"
            )
            return None

    def get_session_history(
        self, session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve past conversation turns for multi-turn chat memory."""
        try:
            turns_ref = self.db.collection("sessions").document(session_id).collection("turns")
            if self.is_mock:
                return []
            
            docs = turns_ref.order_by("timestamp", direction=gfirestore.Query.DESCENDING).limit(limit).stream()
            history = [doc.to_dict() for doc in docs]
            history.reverse()
            return history
        except Exception as e:
            logger.error(f"Error fetching session history for [{session_id}]: {e}")
            return []

    def list_collection_documents(
        self, collection_name: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List uploaded document metadata records from Firestore."""
        try:
            if self.is_mock:
                return []
            docs = self.db.collection(collection_name).limit(limit).stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error listing collection [{collection_name}]: {e}")
            return []


# Global singleton instance
firestore_db = FirestoreManager()
