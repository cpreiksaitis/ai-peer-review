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
from fastapi.responses import StreamingResponse

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

# Global exception handler to prevent 502s
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(f"[ERROR] Unhandled exception: {exc}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)[:100]}"}
    )

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


def sanitize_api_keys() -> None:
    """Trim whitespace/newlines from API keys to avoid header errors."""
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "PERPLEXITY_API_KEY",
    ]:
        val = os.environ.get(key)
        if val:
            os.environ[key] = val.strip()


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


def sanitize_base64_data(data: str | None) -> str | None:
    """Strip data: prefix and fix padding for base64 strings."""
    if not data:
        return None
    clean = data.split(",", 1)[-1]
    # Fix padding
    padding = len(clean) % 4
    if padding:
        clean += "=" * (4 - padding)
    return clean


# Load environment variables and configure providers
load_dotenv()
sanitize_api_keys()
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
os.environ.setdefault("ANTHROPIC_API_URL", "https://api.anthropic.com")
os.environ.setdefault("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")

import litellm
# Use provider defaults; OpenAI base can still be set via env if needed.


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
    print("[DEBUG] Starting up application...")
    
    from src.database import init_async_db, get_async_session_maker
    
    db_engine = await init_async_db()
    session_maker = get_async_session_maker(db_engine)
    print("[DEBUG] Database initialized")
    
    # Apply saved settings
    settings = load_settings()
    apply_settings(settings)
    print("[DEBUG] Settings applied, startup complete")


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

@app.get("/test-models", response_class=HTMLResponse)
async def test_models_page(request: Request):
    """Simple UI to run model connectivity checks."""
    return templates.TemplateResponse(
        "test_models.html",
        {"request": request},
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
        raw_pdf_base64 = manuscript.get("pdf_base64") if use_pdf_vision else None
        pdf_base64 = sanitize_base64_data(raw_pdf_base64)
        
        # Create database record
        print(f"[DEBUG] Creating database record for {file.filename}")
        async with session_maker() as session:
            review = await create_review(
                session=session,
                manuscript_filename=file.filename,
                manuscript_title=manuscript_title,
                manuscript_text=manuscript_text,
            )
            review_id = review.id
        print(f"[DEBUG] Created review with id={review_id}")
        
        # Start background review task
        print(f"[DEBUG] Creating background task for review_id={review_id}")
        try:
            task = asyncio.create_task(
                run_review_task(
                    review_id, manuscript_text, file.filename,
                    search_literature, pdf_base64, show_live_output
                )
            )
            # Add error callback to catch unhandled exceptions
            def task_done_callback(t):
                if t.exception():
                    print(f"[ERROR] Background task failed: {t.exception()}")
            task.add_done_callback(task_done_callback)
            active_reviews[review_id] = task
            print(f"[DEBUG] Task created and stored, returning response")
        except Exception as e:
            print(f"[ERROR] Failed to create task: {e}")
            raise
        
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
    print(f"[DEBUG] Starting review task for review_id={review_id}")
    
    try:
        from src.config import load_config
        from src.database import update_review_results, update_review_status
        from src.orchestrator import create_orchestrator_from_config
        print("[DEBUG] Imports successful")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        raise
    
    # Initialize live output store
    if show_live_output:
        live_output_store[review_id] = []
    
    async def update_status(status: str):
        async with session_maker() as sess:
            await update_review_status(sess, review_id, status)
    
    async with session_maker() as session:
        try:
            print(f"[DEBUG] Updating status to initializing...")
            await update_review_status(session, review_id, "processing: initializing...")
            
            print("[DEBUG] Loading config...")
            config = load_config()
            print(f"[DEBUG] Config loaded: {list(config.keys())}")
            
            print("[DEBUG] Creating orchestrator...")
            orchestrator = create_orchestrator_from_config(config)
            print("[DEBUG] Orchestrator created successfully")
            
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
            print(f"[DEBUG] Starting review execution (search_literature={do_literature_search}, pdf_vision={pdf_base64 is not None})")
            
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
            import traceback
            print(f"[ERROR] Review task failed: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
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
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_review_status failed: {e}")
        return {"id": review_id, "status": "error: server issue", "recommendation": None}


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
    request: Request,
    file: UploadFile = File(...),
    max_results: int = Form(10),
    provider: str = Form("pubmed"),
    use_pdf_vision: bool = Form(True),
):
    """Find similar papers for an uploaded PDF."""
    from src.document import get_manuscript_content
    from src.search_providers import get_provider, PROVIDERS
    from src.search_providers.openai_provider import OpenAISearchProvider
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    # Special streaming route support via header
    try:
        stream = request.headers.get("x-stream", "false").lower() == "true"
    except NameError:
        print("[WARN] 'request' not defined in find_similar_papers - disabling streaming")
        stream = False
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        print(f"[DEBUG] find_similar: Processing {file.filename} with provider={provider}")
        manuscript = get_manuscript_content(tmp_path, include_pdf_data=use_pdf_vision)
        manuscript_text = manuscript["text"]
        raw_pdf_base64 = manuscript.get("pdf_base64") if use_pdf_vision else None
        pdf_base64 = sanitize_base64_data(raw_pdf_base64)
        print(f"[DEBUG] find_similar: Extracted {len(manuscript_text)} chars, pdf_base64={'yes' if pdf_base64 else 'no'}")
        
        search_provider = get_provider(provider)
        provider_pdf = pdf_base64 if (use_pdf_vision and search_provider.supports_pdf) else None
        print(f"[DEBUG] find_similar: Using provider {provider}, supports_pdf={search_provider.supports_pdf}")
        
        # Streaming mode for OpenAI provider
        if stream and provider == "openai":
            def event_generator():
                try:
                    # Build instruction using provider helper
                    if isinstance(search_provider, OpenAISearchProvider):
                        instruction = search_provider._build_instruction(manuscript_text, max_results, provider_pdf)
                    else:
                        instruction = manuscript_text[:5000]
                    
                    tools = [
                        {
                            "type": "web_search",
                            "user_location": {"type": "approximate"},
                            "search_context_size": "high",
                            "filters": {
                                "allowed_domains": [
                                    "pubmed.ncbi.nlm.nih.gov",
                                    "ncbi.nlm.nih.gov",
                                    "scholar.google.com",
                                ]
                            }
                        }
                    ]
                    content = [{"type": "input_text", "text": instruction}]
                    client = search_provider.client if isinstance(search_provider, OpenAISearchProvider) else None
                    resp = client.responses.create(
                        model=search_provider.model,
                        input=[{"role": "user", "content": content}],
                        tools=tools,
                        reasoning={"effort": "medium", "summary": "auto"},
                        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
                        max_output_tokens=5000,
                        stream=True,
                    )
                    for event in resp:
                        try:
                            if hasattr(event, "output_text"):
                                delta = getattr(event.output_text, "delta", "")
                                if delta:
                                    yield f"data: {delta}\n\n"
                            elif event.type == "response.output_text.delta":
                                delta = getattr(event, "delta", "")
                                if delta:
                                    yield f"data: {delta}\n\n"
                        except Exception:
                            continue
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: [ERROR] {str(e)}\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        loop = asyncio.get_event_loop()
        try:
            session = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: search_provider.search(
                        manuscript_text=manuscript_text,
                        max_results=max_results,
                        focus_pubmed=True,
                        pdf_base64=provider_pdf,
                    )
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Search timed out. Please retry.")
        print(f"[DEBUG] find_similar: Got {len(session.results)} results, summary={session.query_summary}")
        
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
            "query_summary": session.query_summary,
            "reasoning": session.reasoning,
            "queries_used": session.queries_used,
            "search_steps": session.search_steps,
            "cost": session.total_cost,
            "tokens": session.tokens_used,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/debug/test-models")
async def test_models():
    """Test API connectivity for all configured models."""
    import litellm
    import os
    
    # Turn on verbose logging for this endpoint to surface HTTP errors in logs
    os.environ["LITELLM_LOG"] = "DEBUG"
    litellm._turn_on_debug()
    
    results = {}
    models_to_test = [
        ("openai_litellm", "gpt-5-mini"),
        ("anthropic", "anthropic/claude-haiku-4-5"),
        ("gemini", "gemini-flash-latest"),
    ]
    
    # Direct OpenAI check (bypasses litellm) to isolate network/auth issues
    try:
        from openai import OpenAI
        client = OpenAI()
        model_list = client.models.list()
        first_model = model_list.data[0].id if getattr(model_list, "data", []) else "unknown"
        results["openai_direct"] = {
            "status": "ok",
            "model": "models.list",
            "response": f"Fetched models, first={first_model}",
        }
    except Exception as e:
        import traceback
        results["openai_direct"] = {
            "status": "error",
            "model": "models.list",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()[-500:],
        }
    
    for provider, model in models_to_test:
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            results[provider] = {
                "status": "ok",
                "model": model,
                "response": response.choices[0].message.content[:50],
            }
        except Exception as e:
            import traceback
            results[provider] = {
                "status": "error",
                "model": model,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()[-500:],
            }
    
    return results


# Run with: uvicorn src.web.production.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
