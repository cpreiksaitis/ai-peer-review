"""FastAPI web application for the multi-agent peer review system."""

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.costs import estimate_review_cost
from src.database import (
    DATABASE_PATH,
    ReviewRecord,
    create_review,
    delete_review,
    get_all_reviews,
    get_async_session_maker,
    get_review,
    init_async_db,
    update_review_results,
    update_review_status,
)
from src.document import get_manuscript_content
from src.orchestrator import Orchestrator, create_orchestrator_from_config

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# Global state
db_engine = None
session_maker = None
active_reviews: dict[int, asyncio.Task] = {}
live_output_store: dict[int, list[dict]] = {}  # review_id -> list of output entries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global db_engine, session_maker

    # Initialize database
    db_engine = await init_async_db()
    session_maker = get_async_session_maker(db_engine)

    yield

    # Cleanup
    if db_engine:
        await db_engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Peer Review",
    description="AI-powered peer review system for biomedical manuscripts",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "perplexity": bool(os.environ.get("PERPLEXITY_API_KEY")),
        "pubmed": bool(os.environ.get("PUBMED_EMAIL")),
    }


def extract_recommendation(final_review: str) -> str:
    """Extract recommendation from final review text."""
    review_lower = final_review.lower()
    if "reject" in review_lower:
        return "Reject"
    elif "major revision" in review_lower:
        return "Major Revision"
    elif "minor revision" in review_lower:
        return "Minor Revision"
    elif "accept" in review_lower:
        return "Accept"
    return "Unknown"


async def run_review_task(review_id: int, manuscript_text: str, manuscript_path: str, search_literature: bool = True, pdf_base64: str | None = None, show_live_output: bool = False):
    """Background task to run the review process."""
    # Initialize live output store for this review
    if show_live_output:
        live_output_store[review_id] = []
    
    # We need a fresh session for each status update since we're in a background task
    async def update_status(status: str):
        """Update review status in database."""
        async with session_maker() as sess:
            await update_review_status(sess, review_id, status)
    
    async with session_maker() as session:
        try:
            await update_review_status(session, review_id, "processing: initializing...")

            # Load config and create orchestrator
            config = load_config()
            orchestrator = create_orchestrator_from_config(config)
            
            # Set up status callback - we need to handle async from sync context
            import threading
            from datetime import datetime
            status_lock = threading.Lock()
            pending_status = [None]  # Use list to allow mutation in closure
            
            def sync_status_callback(status: str):
                """Sync callback that stores status for async update."""
                with status_lock:
                    pending_status[0] = status
            
            def sync_output_callback(agent_name: str, output_type: str, content: str):
                """Sync callback to capture live output."""
                if show_live_output and review_id in live_output_store:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "agent": agent_name,
                        "type": output_type,  # "review", "debate", "final", "synthesis"
                        "content": content[:2000] + "..." if len(content) > 2000 else content,
                    }
                    live_output_store[review_id].append(entry)
            
            orchestrator.status_callback = sync_status_callback
            orchestrator.output_callback = sync_output_callback

            # Get PubMed email (required for literature search)
            pubmed_email = os.environ.get("PUBMED_EMAIL")
            
            # Only search literature if toggle is on AND we have an email
            do_literature_search = search_literature and bool(pubmed_email)

            # Run review in executor with periodic status updates
            loop = asyncio.get_event_loop()
            
            async def run_with_status_updates():
                """Run review while periodically updating status."""
                import concurrent.futures
                
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
                    
                    # Poll for completion while updating status
                    while not future.done():
                        await asyncio.sleep(1)  # Check every second
                        with status_lock:
                            if pending_status[0]:
                                await update_status(pending_status[0])
                                pending_status[0] = None
                    
                    return future.result()
            
            review_session = await run_with_status_updates()

            # Extract recommendation
            recommendation = extract_recommendation(review_session.final_review)

            # Update database with results
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
            
            # Clean up live output store
            if review_id in live_output_store:
                del live_output_store[review_id]

        except Exception as e:
            await update_review_status(session, review_id, f"failed: {str(e)[:100]}")
            # Clean up live output store on failure too
            if review_id in live_output_store:
                del live_output_store[review_id]
            raise


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form and review list."""
    async with session_maker() as session:
        reviews = await get_all_reviews(session)

    api_keys = check_api_keys()
    config = load_config()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reviews": [r.to_dict() for r in reviews],
            "api_keys": api_keys,
            "config": config,
        },
    )


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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text and optionally PDF data from the document
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

        # Start background review task with PDF data
        task = asyncio.create_task(
            run_review_task(review_id, manuscript_text, file.filename, search_literature, pdf_base64, show_live_output)
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
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/review/{review_id}", response_class=HTMLResponse)
async def view_review(request: Request, review_id: int):
    """View a specific review."""
    async with session_maker() as session:
        review = await get_review(session, review_id)

    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "review": review.to_dict(),
        },
    )


@app.get("/api/review/{review_id}")
async def get_review_api(review_id: int):
    """API endpoint to get review data."""
    async with session_maker() as session:
        review = await get_review(session, review_id)

    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    return review.to_dict()


@app.get("/api/review/{review_id}/status")
async def get_review_status(review_id: int):
    """Get just the status of a review (for polling)."""
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
    """Get live output entries for a review (for streaming updates)."""
    if review_id not in live_output_store:
        return {"entries": [], "total": 0}
    
    entries = live_output_store[review_id]
    # Return entries after the 'since' index
    new_entries = entries[since:] if since < len(entries) else []
    
    return {
        "entries": new_entries,
        "total": len(entries),
        "since": since,
    }


@app.delete("/api/review/{review_id}")
async def delete_review_api(review_id: int):
    """Delete a review."""
    # Cancel if still running
    if review_id in active_reviews:
        active_reviews[review_id].cancel()
        del active_reviews[review_id]

    async with session_maker() as session:
        success = await delete_review(session, review_id)

    if not success:
        raise HTTPException(status_code=404, detail="Review not found")

    return {"success": True}


@app.get("/api/estimate")
async def estimate_cost(
    manuscript_length: int = 50000,
    debate_rounds: int = 2,
):
    """Estimate review cost before running."""
    estimate = estimate_review_cost(
        manuscript_length=manuscript_length,
        num_agents=4,
        debate_rounds=debate_rounds,
    )
    return estimate


@app.get("/literature", response_class=HTMLResponse)
async def literature_test_page(request: Request):
    """Literature search test page."""
    api_keys = check_api_keys()
    return templates.TemplateResponse(
        "literature.html",
        {"request": request, "api_keys": api_keys},
    )


@app.get("/similar", response_class=HTMLResponse)
async def find_similar_page(request: Request):
    """Find similar papers page."""
    from src.search_providers import list_providers
    
    api_keys = check_api_keys()
    config = load_config()
    
    # Get provider info with availability
    providers = []
    for p in list_providers():
        p["available"] = check_provider_available(p["name"])
        providers.append(p)
    
    return templates.TemplateResponse(
        "similar.html",
        {
            "request": request,
            "api_keys": api_keys,
            "config": config,
            "providers": providers,
        },
    )


def check_provider_available(name: str) -> bool:
    """Check if a search provider is available."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "pubmed": "PUBMED_EMAIL",
    }
    env_var = key_map.get(name)
    if env_var:
        # Also check GEMINI_API_KEY for Gemini
        if name == "gemini":
            return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        return bool(os.environ.get(env_var))
    return False


@app.post("/api/literature/search")
async def search_literature_api(
    query: str = Form(...),
    max_results: int = Form(10),
    score_relevance: bool = Form(True),
    manuscript_context: str = Form(""),
):
    """API endpoint to test literature search."""
    import asyncio
    from src.literature import search_related_literature, filter_and_rank_papers

    email = os.environ.get("PUBMED_EMAIL")
    if not email:
        raise HTTPException(status_code=400, detail="PUBMED_EMAIL not configured")

    # Run search in executor (blocking call)
    loop = asyncio.get_event_loop()

    try:
        papers = await loop.run_in_executor(
            None,
            lambda: search_related_literature(
                query=query,
                max_results=max_results * 2 if score_relevance else max_results,
                email=email,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Score relevance if requested
    if score_relevance and papers:
        context = manuscript_context if manuscript_context else query
        papers = await loop.run_in_executor(
            None,
            lambda: filter_and_rank_papers(
                papers=papers,
                manuscript_summary=context[:2000],
                top_n=max_results,
                min_score=1.0,  # Return all for inspection
                model="gpt-5-nano",
            )
        )

    # Format response
    results = []
    for paper in papers:
        results.append({
            "pmid": paper.pmid,
            "title": paper.title,
            "authors": paper.authors[:3],
            "journal": paper.journal,
            "pub_date": paper.pub_date,
            "abstract": paper.abstract[:500] if paper.abstract else "",
            "doi": paper.doi,
            "relevance_score": paper.relevance_score,
            "relevance_reason": paper.relevance_reason,
        })

    return {
        "query": query,
        "total_found": len(results),
        "papers": results,
    }


@app.post("/api/similar/find")
async def find_similar_papers(
    file: UploadFile = File(...),
    max_results: int = Form(10),
    provider: str = Form("pubmed"),
    focus_pubmed: bool = Form(True),
    use_pdf_vision: bool = Form(True),
):
    """Find similar papers for an uploaded PDF using selected provider."""
    import asyncio
    from src.search_providers import get_provider, PROVIDERS

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate provider
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # Check provider is available
    if not check_provider_available(provider):
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' is not configured. Please add the required API key."
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
        manuscript_title = manuscript["metadata"].get("title") or file.filename
        pdf_base64 = manuscript.get("pdf_base64") if use_pdf_vision else None

        loop = asyncio.get_event_loop()

        # Get provider and run search
        search_provider = get_provider(provider)
        
        # Only pass PDF if provider supports it
        provider_pdf = pdf_base64 if (use_pdf_vision and search_provider.supports_pdf) else None
        
        session = await loop.run_in_executor(
            None,
            lambda: search_provider.search(
                manuscript_text=manuscript_text,
                max_results=max_results,
                focus_pubmed=focus_pubmed,
                pdf_base64=provider_pdf,
            )
        )

        # Format response
        papers = []
        for result in session.results:
            papers.append({
                "pmid": result.pmid,
                "title": result.title,
                "authors": result.authors[:3] if result.authors else [],
                "journal": result.journal,
                "pub_date": result.pub_date,
                "abstract": result.abstract[:500] if result.abstract else "",
                "doi": result.doi,
                "url": result.url,
                "relevance_score": result.relevance_score,
                "relevance_reason": result.relevance_reason,
            })

        return {
            "provider": provider,
            "manuscript_title": manuscript_title,
            "queries_generated": session.queries_used,
            "queries_used": session.queries_used,
            "search_results": session.search_steps,
            "reasoning": session.reasoning,
            "total_unique_papers": len(session.results),
            "papers": papers,
            "cost": session.total_cost,
            "tokens": session.tokens_used,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/status")
async def system_status():
    """Get system status."""
    api_keys = check_api_keys()
    config = load_config()

    return {
        "api_keys": api_keys,
        "config": config,
        "active_reviews": len(active_reviews),
        "database": DATABASE_PATH.exists(),
    }


# =============================================================================
# EVALUATION ENDPOINTS
# =============================================================================

@app.get("/evaluate/literature", response_class=HTMLResponse)
async def evaluate_literature_page(request: Request):
    """Literature provider comparison page."""
    from src.search_providers import list_providers, PROVIDERS, get_provider
    
    api_keys = check_api_keys()
    
    # Get available providers
    providers = []
    for p in list_providers():
        if p["name"] in PROVIDERS:
            provider_instance = get_provider(p["name"])
            p["available"] = provider_instance.is_available()
            providers.append(p)
    
    return templates.TemplateResponse(
        "evaluate_literature.html",
        {
            "request": request,
            "api_keys": api_keys,
            "providers": providers,
        },
    )


@app.post("/api/evaluate/literature")
async def evaluate_literature_api(
    file: UploadFile = File(...),
    max_results: int = Form(10),
    providers: str = Form(""),
    judge_model: str = Form("gpt-5-nano"),
):
    """API endpoint to compare literature providers."""
    import asyncio
    from src.evaluation.compare_providers import compare_all_providers

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        
        provider_list = None
        if providers:
            provider_list = [p.strip() for p in providers.split(",") if p.strip()]

        comparison = await loop.run_in_executor(
            None,
            lambda: compare_all_providers(
                manuscript_path=tmp_path,
                max_results=max_results,
                providers=provider_list,
                judge_model=judge_model,
                verbose=False,
            )
        )

        return comparison.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/evaluate/review", response_class=HTMLResponse)
async def evaluate_review_page(request: Request):
    """Review quality evaluation page."""
    api_keys = check_api_keys()
    
    return templates.TemplateResponse(
        "evaluate_review.html",
        {
            "request": request,
            "api_keys": api_keys,
        },
    )


@app.post("/api/evaluate/review")
async def evaluate_review_api(
    manuscript: UploadFile = File(...),
    review_text: str = Form(...),
    judge_model: str = Form("gpt-5-nano"),
):
    """API endpoint to evaluate review quality."""
    import asyncio
    from src.evaluation.review_eval import ReviewEvaluator

    if not manuscript.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await manuscript.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        
        # Get manuscript content
        manuscript_content = get_manuscript_content(tmp_path)
        
        def run_evaluation():
            evaluator = ReviewEvaluator(model=judge_model)
            summary = evaluator.generate_manuscript_summary(manuscript_content["text"])
            result = evaluator.evaluate(
                review_text=review_text,
                manuscript_summary=summary,
            )
            return {
                "manuscript_summary": summary,
                "evaluation": result.to_dict(),
                "report": result.format_report(),
                "grade": result.get_grade(),
            }
        
        result = await loop.run_in_executor(None, run_evaluation)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": "Page not found", "status_code": 404},
        status_code=404,
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": str(exc), "status_code": 500},
        status_code=500,
    )

