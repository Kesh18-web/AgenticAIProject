import os
from typing import Any, Dict, List
from backend.app.core.logging import logger

UPLOADS_DIR = os.path.abspath("backend/app/data/uploads")


def list_uploads_files() -> List[Dict[str, Any]]:
    """MCP Tool: List all uploaded raw files stored in the server's uploads folder."""
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        files = os.listdir(UPLOADS_DIR)
        file_list = []
        for fname in files:
            fpath = os.path.join(UPLOADS_DIR, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                file_list.append(
                    {
                        "filename": fname,
                        "size_bytes": stat.st_size,
                        "modified_timestamp": stat.st_mtime,
                    }
                )
        logger.info(f"[MCP Filesystem] Listed {len(file_list)} files from '{UPLOADS_DIR}'")
        return file_list
    except Exception as e:
        logger.error(f"[MCP Filesystem] Error listing uploaded files: {e}")
        return []


def read_upload_file(filename: str) -> Dict[str, Any]:
    """MCP Tool: Read raw text content of a file from the server's uploads folder with path traversal safety."""
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        target_path = os.path.abspath(os.path.join(UPLOADS_DIR, filename))

        # Security check: Ensure target_path stays strictly inside UPLOADS_DIR
        if not target_path.startswith(UPLOADS_DIR):
            logger.warning(f"[MCP Filesystem] Security Violation: Path traversal attempt '{filename}'")
            return {"error": "Access Denied: Path traversal outside uploads directory is forbidden."}

        if not os.path.exists(target_path):
            return {"error": f"File '{filename}' not found in uploads directory."}

        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        logger.info(f"[MCP Filesystem] Read {len(content)} characters from '{filename}'")
        return {
            "filename": filename,
            "content": content[:10000],  # Safety cap to 10k chars
            "total_chars": len(content),
        }
    except Exception as e:
        logger.error(f"[MCP Filesystem] Error reading file '{filename}': {e}")
        return {"error": f"Failed to read file: {str(e)}"}
