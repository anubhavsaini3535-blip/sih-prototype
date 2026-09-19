import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", message=".*ccache.*")

from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR
from app.database import engine, Base
import app.models  # Ensures all SQLAlchemy models are registered
from app.routers import auth_router, scans_router, dashboard_router
from app.services.ocr_service import warmup_ocr

# Auto-create tables on startup (zero migration setup for prototype)
Base.metadata.create_all(bind=engine)


from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-seed database if fresh/empty (vital for Vercel /tmp SQLite storage)
    try:
        from app.database import SessionLocal
        from app.models.user import User
        db = SessionLocal()
        try:
            if db.query(User).count() == 0:
                from seed import seed_database
                seed_database()
        finally:
            db.close()
    except Exception as e:
        import logging
        logging.warning("Auto-bootstrap database notice: %s", e)

    # Pre-warm PaddleOCR model weights if installed
    try:
        warmup_ocr()
    except Exception as e:
        pass
    yield


app = FastAPI(
    title="Legal Metrology Compliance Checking System",
    description=(
        "Automated AI-powered regulatory compliance verification system for packaged "
        "commodities under the Legal Metrology Act, 2009 and Packaged Commodities Rules, 2011."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local uploads directory for inspecting stored packaging images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Mount static HTML test pages (e.g. /static/upload.html)
import os as _os
_STATIC_DIR = _os.path.join(_os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Register API Routers
app.include_router(auth_router)
app.include_router(scans_router)
app.include_router(dashboard_router)


@app.get("/", include_in_schema=False)
def index():
    index_file = _os.path.join(_STATIC_DIR, "upload.html")
    if _os.path.exists(index_file):
        return FileResponse(index_file)
    return health_check()


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "Legal Metrology Compliance Checking Backend",
        "version": "1.0.0",
        "documentation": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
