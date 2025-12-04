"""
Production Multi-Agent Peer Review Application

A streamlined interface for:
- Submitting manuscripts for AI-powered peer review
- Finding similar papers in the literature

Designed for easy deployment with user-configurable API keys.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path("data")
SETTINGS_FILE = DATA_DIR / "settings.json"

# Create data directory
DATA_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="AI Peer Review",
    description="Multi-Agent Peer Review System",
    version="1.0.0",
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state
db_engine = None
session_maker = None
active_reviews: dict[int, asyncio.Task] = {}
live_output_store: dict[int, list[dict]] = {}


def load_settings() -> dict:
    """Load settings from file or return defaults."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "openai_api_key": "",
        "anthropic_api_key": "",
        "google_api_key": "",
        "perplexity_api_key": "",
        "pubmed_email": "",
    }


def save_settings(settings: dict) -> None:
    """Save settings to file."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def apply_settings(settings: dict) -> None:
    """Apply settings to environment variables."""
    key_mapping = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "perplexity_api_key": "PERPLEXITY_API_KEY",
        "pubmed_email": "PUBMED_EMAIL",
    }
    for key, env_var in key_mapping.items():
        if settings.get(key):
            os.environ[env_var] = settings[key]


def check_api_status() -> dict:
    """Check which APIs are configured."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY")),
        "perplexity": bool(os.environ.get("PERPLEXITY_API_KEY")),
        "pubmed": bool(os.environ.get("PUBMED_EMAIL")),
    }


def get_available_providers() -> list[dict]:
    """Get list of available search providers based on configured API keys."""
    from src.search_providers import list_providers
    
    providers = []
    api_status = check_api_status()
    
    provider_key_map = {
        "openai": "openai",
        "claude": "anthropic",
        "gemini": "google",
        "perplexity": "perplexity",
        "pubmed": "pubmed",
    }
    
    for p in list_providers():
        key_name = provider_key_map.get(p["name"], "")
        p["available"] = api_status.get(key_name, False)
        providers.append(p)
    
    return providers


@app.on_event("startup")
async def startup():
    """Initialize database and apply saved settings."""
    global db_engine, session_maker
    
    from src.database import init_async_db, get_async_session_maker
    
    db_engine = await init_async_db()
    session_maker = get_async_session_maker(db_engine)
    
    # Apply saved settings
    settings = load_settings()
    apply_settings(settings)


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources."""
    global db_engine
    if db_engine:
        await db_engine.dispose()


# ============================================================================
# PAGES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - manuscript submission and recent reviews."""
    from src.database import get_all_reviews
    from src.config import load_config
    
    async with session_maker() as session:
        reviews = await get_all_reviews(session, limit=20)
    
    config = load_config()
    api_status = check_api_status()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reviews": reviews,
            "config": config,
            "api_status": api_status,
            "any_api_configured": any(api_status.values()),
        },
    )


@app.get("/review/{review_id}", response_class=HTMLResponse)
async def view_review(request: Request, review_id: int):
    """View a completed review."""
    from src.database import get_review
    
    async with session_maker() as session:
        review = await get_review(session, review_id)
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return templates.TemplateResponse(
        "review.html",
        {"request": request, "review": review.to_dict()},
    )


@app.get("/similar", response_class=HTMLResponse)
async def find_similar_page(request: Request):
    """Find similar papers page."""
    providers = get_available_providers()
    api_status = check_api_status()
    
    return templates.TemplateResponse(
        "similar.html",
        {
            "request": request,
            "providers": providers,
            "api_status": api_status,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page for API key configuration."""
    settings = load_settings()
    api_status = check_api_status()
    
    # Mask API keys for display
    masked_settings = {}
    for key, value in settings.items():
        if value and "key" in key.lower():
            masked_settings[key] = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        else:
            masked_settings[key] = value
    
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": masked_settings,
            "api_status": api_status,
        },
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update API key settings."""
    form_data = await request.form()
    
    current_settings = load_settings()
    
    # Only update non-empty values
    for key in current_settings.keys():
        value = form_data.get(key, "")
        if value and not value.startswith("***") and "..." not in value:
            current_settings[key] = value
    
    save_settings(current_settings)
    apply_settings(current_settings)
    
    return JSONResponse({"success": True, "message": "Settings saved"})


@app.post("/upload")
async def upload_manuscript(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    search_literature: bool = Form(True),
    use_pdf_vision: bool = Form(True),
    show_live_output: bool = Form(False),
):
    """Upload a manuscript for review."""
    from src.database import create_review
    from src.document import get_manuscript_content
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check if at least one API is configured
    api_status = check_api_status()
    if not any([api_status["openai"], api_status["anthropic"], api_status["google"]]):
        raise HTTPException(
            status_code=400,
            detail="No API keys configured. Please add at least one API key in Settings."
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract text and optionally PDF data
        manuscript = get_manuscript_content(tmp_path, include_pdf_data=use_pdf_vision)
        manuscript_text = manuscript["text"]
        manuscript_title = title or manuscript["metadata"].get("title") or file.filename
        pdf_base64 = manuscript.get("pdf_base64") if use_pdf_vision else None
        
        # Create database record
        async with session_maker() as session:
            review = await create_review(
                session=session,
                manuscript_filename=file.filename,
                manuscript_title=manuscript_title,
                manuscript_text=manuscript_text,
            )
            review_id = review.id
        
        # Start background review task
        task = asyncio.create_task(
            run_review_task(
                review_id, manuscript_text, file.filename,
                search_literature, pdf_base64, show_live_output
            )
        )
        active_reviews[review_id] = task
        
        return JSONResponse({
            "success": True,
            "review_id": review_id,
            "message": "Review started",
            "show_live_output": show_live_output,
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def run_review_task(
    review_id: int,
    manuscript_text: str,
    manuscript_path: str,
    search_literature: bool = True,
    pdf_base64: Optional[str] = None,
    show_live_output: bool = False,
):
    """Background task to run the review process."""
    from src.config import load_config
    from src.database import update_review_results, update_review_status
    from src.orchestrator import create_orchestrator_from_config
    
    # Initialize live output store
    if show_live_output:
        live_output_store[review_id] = []
    
    async def update_status(status: str):
        async with session_maker() as sess:
            await update_review_status(sess, review_id, status)
    
    async with session_maker() as session:
        try:
            await update_review_status(session, review_id, "processing: initializing...")
            
            config = load_config()
            orchestrator = create_orchestrator_from_config(config)
            
            # Set up callbacks
            import threading
            from datetime import datetime
            status_lock = threading.Lock()
            pending_status = [None]
            
            def sync_status_callback(status: str):
                with status_lock:
                    pending_status[0] = status
            
            def sync_output_callback(agent_name: str, output_type: str, content: str):
                if show_live_output and review_id in live_output_store:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "agent": agent_name,
                        "type": output_type,
                        "content": content[:2000] + "..." if len(content) > 2000 else content,
                    }
                    live_output_store[review_id].append(entry)
            
            orchestrator.status_callback = sync_status_callback
            orchestrator.output_callback = sync_output_callback
            
            # Get PubMed email
            pubmed_email = os.environ.get("PUBMED_EMAIL")
            do_literature_search = search_literature and bool(pubmed_email)
            
            # Run review
            import concurrent.futures
            
            async def run_with_status_updates():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: orchestrator.run_review(
                            manuscript_text=manuscript_text,
                            manuscript_path=manuscript_path,
                            search_literature=do_literature_search,
                            pubmed_email=pubmed_email,
                            pdf_base64=pdf_base64,
                        )
                    )
                    
                    while not future.done():
                        await asyncio.sleep(1)
                        with status_lock:
                            if pending_status[0]:
                                await update_status(pending_status[0])
                                pending_status[0] = None
                    
                    return future.result()
            
            review_session = await run_with_status_updates()
            
            # Extract recommendation
            recommendation = extract_recommendation(review_session.final_review)
            
            # Update database
            await update_review_results(
                session=session,
                review_id=review_id,
                initial_reviews=review_session.initial_reviews,
                debate_rounds=review_session.debate_rounds,
                final_positions=review_session.final_positions,
                final_review=review_session.final_review,
                literature_context=review_session.literature_context,
                cost_summary=review_session.cost_summary,
                recommendation=recommendation,
            )
            
            # Clean up
            if review_id in live_output_store:
                del live_output_store[review_id]
        
        except Exception as e:
            await update_review_status(session, review_id, f"failed: {str(e)[:100]}")
            if review_id in live_output_store:
                del live_output_store[review_id]
            raise


def extract_recommendation(review_text: str) -> str:
    """Extract recommendation from review text."""
    import re
    text_lower = review_text.lower()
    
    patterns = [
        (r"recommendation[:\s]*(accept|reject|major revision|minor revision)", 1),
        (r"(accept|reject|major revision|minor revision)\s+recommendation", 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(group).title()
    
    if "reject" in text_lower:
        return "Reject"
    elif "major revision" in text_lower:
        return "Major Revision"
    elif "minor revision" in text_lower:
        return "Minor Revision"
    elif "accept" in text_lower:
        return "Accept"
    
    return "Unknown"


@app.get("/api/review/{review_id}/status")
async def get_review_status(review_id: int):
    """Get review status for polling."""
    from src.database import get_review
    
    async with session_maker() as session:
        review = await get_review(session, review_id)
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return {
        "id": review.id,
        "status": review.status,
        "recommendation": review.recommendation,
    }


@app.get("/api/review/{review_id}/live-output")
async def get_live_output(review_id: int, since: int = 0):
    """Get live output entries."""
    if review_id not in live_output_store:
        return {"entries": [], "total": 0}
    
    entries = live_output_store[review_id]
    new_entries = entries[since:] if since < len(entries) else []
    
    return {
        "entries": new_entries,
        "total": len(entries),
        "since": since,
    }


@app.delete("/api/review/{review_id}")
async def delete_review(review_id: int):
    """Delete a review."""
    from src.database import get_review
    from sqlalchemy import delete
    from src.database import ReviewRecord
    
    async with session_maker() as session:
        review = await get_review(session, review_id)
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")
        
        await session.execute(
            delete(ReviewRecord).where(ReviewRecord.id == review_id)
        )
        await session.commit()
    
    return {"success": True}


@app.post("/api/similar/find")
async def find_similar_papers(
    file: UploadFile = File(...),
    max_results: int = Form(10),
    provider: str = Form("pubmed"),
    use_pdf_vision: bool = Form(True),
):
    """Find similar papers for an uploaded PDF."""
    from src.document import get_manuscript_content
    from src.search_providers import get_provider, PROVIDERS
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        manuscript = get_manuscript_content(tmp_path, include_pdf_data=use_pdf_vision)
        manuscript_text = manuscript["text"]
        pdf_base64 = manuscript.get("pdf_base64") if use_pdf_vision else None
        
        search_provider = get_provider(provider)
        provider_pdf = pdf_base64 if (use_pdf_vision and search_provider.supports_pdf) else None
        
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(
            None,
            lambda: search_provider.search(
                manuscript_text=manuscript_text,
                max_results=max_results,
                focus_pubmed=True,
                pdf_base64=provider_pdf,
            )
        )
        
        papers = []
        for result in session.results:
            papers.append({
                "pmid": result.pmid,
                "title": result.title,
                "authors": result.authors[:3] if result.authors else [],
                "journal": result.journal,
                "pub_date": result.pub_date,
                "abstract": result.abstract[:300] + "..." if len(result.abstract) > 300 else result.abstract,
                "url": result.url,
                "relevance_score": result.relevance_score,
                "relevance_reason": result.relevance_reason,
            })
        
        return {
            "success": True,
            "provider": provider,
            "total_found": len(papers),
            "papers": papers,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Run with: uvicorn src.web.production.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

