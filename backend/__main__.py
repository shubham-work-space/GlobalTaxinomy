"""Run the embedding API: python -m backend"""

import uvicorn

from backend.app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=settings.api_port,
        reload=False,
    )
