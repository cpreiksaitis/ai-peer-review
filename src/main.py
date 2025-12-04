"""CLI entry point for the multi-agent peer review system."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.document import get_manuscript_content
from src.orchestrator import Orchestrator, create_orchestrator_from_config, ReviewSession

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="reviewer",
    help="Multi-agent peer review system for biomedical manuscripts",
    add_completion=False,
)
console = Console()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
        console.print("[dim]Using default configuration[/dim]")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def save_review_session(session: ReviewSession, output_dir: Path, include_debate: bool = True):
    """Save review session to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manuscript_name = Path(session.manuscript_path).stem if session.manuscript_path else "manuscript"

    # Save final review
    review_path = output_dir / f"{manuscript_name}_review_{timestamp}.md"
    with open(review_path, "w") as f:
        f.write(f"# Peer Review: {manuscript_name}\n\n")
        f.write(f"**Generated:** {session.timestamp}\n\n")
        f.write("---\n\n")
        f.write(session.final_review)

    console.print(f"[green]✓[/green] Final review saved to: {review_path}")

    # Save full debate log if requested
    if include_debate:
        debate_path = output_dir / f"{manuscript_name}_debate_{timestamp}.md"
        with open(debate_path, "w") as f:
            f.write(f"# Review Debate Log: {manuscript_name}\n\n")
            f.write(f"**Generated:** {session.timestamp}\n\n")
            f.write("---\n\n")

            f.write("## Initial Reviews\n\n")
            for name, review in session.initial_reviews.items():
                f.write(f"### {name}\n\n{review}\n\n---\n\n")

            if session.debate_rounds:
                f.write("## Debate Rounds\n\n")
                for i, round_responses in enumerate(session.debate_rounds, 1):
                    f.write(f"### Round {i}\n\n")
                    for name, response in round_responses.items():
                        f.write(f"#### {name}\n\n{response}\n\n")
                    f.write("---\n\n")

            f.write("## Final Positions\n\n")
            for name, position in session.final_positions.items():
                f.write(f"### {name}\n\n{position}\n\n---\n\n")

        console.print(f"[green]✓[/green] Debate log saved to: {debate_path}")


@app.command()
def review(
    manuscript: Path = typer.Argument(
        ...,
        help="Path to the manuscript PDF file",
        exists=True,
        readable=True,
    ),
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="Directory to save review output",
    ),
    no_literature: bool = typer.Option(
        False,
        "--no-literature",
        help="Skip literature search",
    ),
    pubmed_email: Optional[str] = typer.Option(
        None,
        "--email",
        "-e",
        help="Email for PubMed API (or set PUBMED_EMAIL env var)",
    ),
    debate_rounds: Optional[int] = typer.Option(
        None,
        "--rounds",
        "-r",
        help="Number of debate rounds (overrides config)",
    ),
    save_debate: bool = typer.Option(
        True,
        "--save-debate/--no-save-debate",
        help="Save full debate transcript",
    ),
):
    """
    Run multi-agent peer review on a manuscript.

    The review process:
    1. Extract text from PDF
    2. Search PubMed for related literature (optional)
    3. Collect initial reviews from 4 specialized agents
    4. Run debate rounds where agents respond to each other
    5. Synthesize final consensus review
    """
    console.print(Panel(
        "[bold]Multi-Agent Peer Review System[/bold]\n"
        "For biomedical manuscripts",
        style="blue",
    ))

    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    optional_keys = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

    missing_keys = [k for k in required_keys if not os.environ.get(k)]
    if missing_keys:
        console.print(f"[red]Error: Missing required API keys: {', '.join(missing_keys)}[/red]")
        console.print("[dim]Set these as environment variables before running.[/dim]")
        raise typer.Exit(1)

    has_gemini = any(os.environ.get(k) for k in optional_keys)
    if not has_gemini:
        console.print("[yellow]Warning: No Gemini API key found. Domain Expert will use fallback model.[/yellow]")

    # Load configuration
    cfg = load_config(config)

    # Get PubMed email
    email = pubmed_email or os.environ.get("PUBMED_EMAIL") or cfg.get("literature", {}).get("email")
    if not no_literature and not email:
        console.print("[yellow]Warning: No PubMed email provided. Literature search may fail.[/yellow]")
        console.print("[dim]Set PUBMED_EMAIL env var or use --email flag.[/dim]")

    # Extract manuscript
    console.print(f"\n[bold]Loading manuscript:[/bold] {manuscript}")
    try:
        manuscript_content = get_manuscript_content(manuscript)
        console.print(f"[green]✓[/green] Extracted {len(manuscript_content['text'])} characters")
    except Exception as e:
        console.print(f"[red]Error reading manuscript: {e}[/red]")
        raise typer.Exit(1)

    # Create orchestrator
    orchestrator = create_orchestrator_from_config(cfg)

    # Override debate rounds if specified
    if debate_rounds is not None:
        orchestrator.debate_rounds = debate_rounds

    # Run review
    try:
        session = orchestrator.run_review(
            manuscript_text=manuscript_content["text"],
            manuscript_path=str(manuscript),
            search_literature=not no_literature,
            pubmed_email=email,
            max_literature_results=cfg.get("literature", {}).get("max_results", 10),
        )
    except Exception as e:
        console.print(f"[red]Error during review: {e}[/red]")
        raise typer.Exit(1)

    # Save results
    console.print("\n[bold]Saving results...[/bold]")
    save_review_session(
        session=session,
        output_dir=output_dir,
        include_debate=save_debate,
    )

    console.print("\n[bold green]Review complete![/bold green]")


@app.command()
def check():
    """Check that all required dependencies and API keys are configured."""
    console.print("[bold]Checking configuration...[/bold]\n")

    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI (GPT-4o)",
        "ANTHROPIC_API_KEY": "Anthropic (Claude)",
        "GEMINI_API_KEY": "Google (Gemini)",
        "GOOGLE_API_KEY": "Google (Gemini, alternative)",
        "PUBMED_EMAIL": "PubMed API",
    }

    console.print("[bold]API Keys:[/bold]")
    for key, description in api_keys.items():
        value = os.environ.get(key)
        if value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            console.print(f"  [green]✓[/green] {key}: {masked} ({description})")
        else:
            status = "[yellow]○[/yellow]" if key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "PUBMED_EMAIL"] else "[red]✗[/red]"
            console.print(f"  {status} {key}: not set ({description})")

    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")

    deps = [
        ("litellm", "LiteLLM"),
        ("pypaperretriever", "PyPaperRetriever"),
        ("fitz", "PyMuPDF"),
        ("yaml", "PyYAML"),
        ("typer", "Typer"),
        ("rich", "Rich"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {name}")
        except ImportError:
            console.print(f"  [red]✗[/red] {name} (not installed)")

    # Check config file
    console.print("\n[bold]Configuration:[/bold]")
    config_path = Path("config.yaml")
    if config_path.exists():
        console.print(f"  [green]✓[/green] config.yaml found")
    else:
        console.print(f"  [yellow]○[/yellow] config.yaml not found (will use defaults)")


@app.command()
def agents():
    """List the reviewer agents and their roles."""
    console.print("[bold]Multi-Agent Review Team[/bold]\n")

    # Load config to get actual models
    cfg = load_config(Path("config.yaml"))
    models = cfg.get("models", {})

    agents_info = [
        ("Orchestrator", models.get("orchestrator", "claude-haiku-4-5"), "Workflow management, debate facilitation, consensus synthesis"),
        ("Methodologist", models.get("methodologist", "gpt-5-nano"), "Study design, statistics, reproducibility, bias assessment"),
        ("Domain Expert", models.get("domain_expert", "gemini/gemini-flash-lite-latest"), "Clinical relevance, novelty, field context, prior work"),
        ("Communication Reviewer", models.get("communication", "gemini/gemini-flash-lite-latest"), "Writing clarity, figures, organization, accessibility"),
        ("Ethics Reviewer", models.get("ethics", "gpt-5-nano"), "Research ethics, patient safety, limitations, broader impact"),
    ]

    for name, model, role in agents_info:
        console.print(Panel(
            f"[bold]{name}[/bold]\n"
            f"[dim]Model: {model}[/dim]\n\n"
            f"{role}",
            border_style="blue" if name == "Orchestrator" else "dim",
        ))

    console.print("\n[bold]Available Models[/bold]")
    console.print("[dim]Configure in config.yaml[/dim]\n")
    console.print("  [cyan]Claude:[/cyan] claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-5-20251101")
    console.print("  [green]GPT:[/green] gpt-5.1-chat-latest, gpt-5-mini, gpt-5-nano")
    console.print("  [yellow]Gemini:[/yellow] gemini/gemini-3-pro-preview, gemini/gemini-flash-latest, gemini/gemini-flash-lite-latest")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the web interface."""
    import uvicorn

    console.print(Panel(
        f"[bold]Starting Multi-Agent Reviewer Web Interface[/bold]\n\n"
        f"Open your browser to: [cyan]http://{host}:{port}[/cyan]",
        style="blue",
    ))

    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def literature(
    query: str = typer.Argument(..., help="Search query for PubMed"),
    max_results: int = typer.Option(10, "--max", "-m", help="Maximum results"),
    score: bool = typer.Option(True, "--score/--no-score", help="Score relevance"),
    manuscript: Optional[Path] = typer.Option(None, "--manuscript", "-f", help="Optional manuscript PDF for context"),
):
    """Test literature search independently."""
    from src.literature import search_related_literature, filter_and_rank_papers, format_literature_context
    from src.document import get_manuscript_content

    console.print(Panel(f"[bold]Literature Search Test[/bold]\n\nQuery: {query}", style="blue"))

    # Check for PubMed email
    email = os.environ.get("PUBMED_EMAIL")
    if not email:
        console.print("[red]Error: PUBMED_EMAIL not set[/red]")
        raise typer.Exit(1)

    # Search PubMed
    console.print(f"\n[bold]Searching PubMed...[/bold]")
    try:
        papers = search_related_literature(query, max_results=max_results * 2 if score else max_results, email=email)
        console.print(f"[green]Found {len(papers)} papers[/green]")
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise typer.Exit(1)

    # Score relevance if requested
    if score and papers:
        manuscript_summary = query  # Default to query as context
        if manuscript and manuscript.exists():
            console.print(f"\n[bold]Loading manuscript for context...[/bold]")
            content = get_manuscript_content(manuscript)
            manuscript_summary = content["text"][:2000]

        console.print(f"\n[bold]Scoring relevance...[/bold]")
        papers = filter_and_rank_papers(
            papers=papers,
            manuscript_summary=manuscript_summary,
            top_n=max_results,
            min_score=1.0,  # Show all for testing
            model="gpt-5-nano",
        )

    # Display results
    console.print(f"\n[bold]Results ({len(papers)} papers):[/bold]\n")
    for i, paper in enumerate(papers, 1):
        score_str = f" [cyan](Score: {paper.relevance_score:.1f})[/cyan]" if paper.relevance_score > 0 else ""
        console.print(f"[bold]{i}. {paper.title[:80]}{'...' if len(paper.title) > 80 else ''}[/bold]{score_str}")
        console.print(f"   [dim]PMID: {paper.pmid} | {paper.journal}[/dim]")
        if paper.relevance_reason:
            console.print(f"   [yellow]{paper.relevance_reason}[/yellow]")
        if paper.abstract:
            console.print(f"   {paper.abstract[:200]}...")
        console.print()


@app.command()
def estimate(
    manuscript_length: int = typer.Option(50000, "--length", "-l", help="Manuscript length in characters"),
    debate_rounds: int = typer.Option(2, "--rounds", "-r", help="Number of debate rounds"),
):
    """Estimate the cost of a review before running."""
    from src.costs import estimate_review_cost

    console.print("[bold]Cost Estimate[/bold]\n")

    estimate = estimate_review_cost(
        manuscript_length=manuscript_length,
        num_agents=4,
        debate_rounds=debate_rounds,
    )

    console.print(f"Manuscript length: ~{manuscript_length:,} characters")
    console.print(f"Debate rounds: {debate_rounds}\n")

    console.print("[bold]Estimated costs:[/bold]")
    for key, value in estimate.items():
        if key != "total_estimated":
            console.print(f"  {key.replace('_', ' ').title()}: ${value:.4f}")

    console.print(f"\n[bold green]Total Estimated: ${estimate['total_estimated']:.4f}[/bold green]")


@app.command("compare-literature")
def compare_literature(
    manuscript: Path = typer.Argument(
        ...,
        help="Path to the manuscript PDF file",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for results",
    ),
    max_results: int = typer.Option(10, "--max", "-m", help="Maximum papers per provider"),
    providers: Optional[str] = typer.Option(
        None,
        "--providers",
        "-p",
        help="Comma-separated list of providers to test (default: all available)",
    ),
    judge_model: str = typer.Option(
        "gpt-5-nano",
        "--judge",
        "-j",
        help="Model to use for LLM-as-judge",
    ),
):
    """
    Compare literature search providers on a manuscript.
    
    Runs all available providers, evaluates results with LLM-as-judge,
    and ranks them by quality score.
    """
    from src.evaluation.compare_providers import compare_all_providers

    provider_list = None
    if providers:
        provider_list = [p.strip() for p in providers.split(",")]

    # Run comparison
    comparison = compare_all_providers(
        manuscript_path=str(manuscript),
        max_results=max_results,
        providers=provider_list,
        judge_model=judge_model,
        verbose=True,
    )

    # Save results if output specified
    if output:
        comparison.save(output)
        console.print(f"\n[green]✓[/green] Results saved to: {output}")

    # Print detailed report
    console.print("\n" + "=" * 60)
    console.print(comparison.format_detailed_report())


@app.command("evaluate-review")
def evaluate_review(
    manuscript: Path = typer.Argument(
        ...,
        help="Path to the manuscript PDF file",
        exists=True,
        readable=True,
    ),
    review: Path = typer.Argument(
        ...,
        help="Path to the review text file",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for results",
    ),
    judge_model: str = typer.Option(
        "gpt-5-nano",
        "--judge",
        "-j",
        help="Model to use for LLM-as-judge",
    ),
):
    """
    Evaluate the quality of a peer review.
    
    Uses LLM-as-judge to score the review against quality criteria
    and provide improvement suggestions.
    """
    from src.evaluation.review_eval import ReviewEvaluator

    # Load manuscript
    console.print(f"[bold]Loading manuscript:[/bold] {manuscript}")
    manuscript_content = get_manuscript_content(str(manuscript))

    # Load review
    console.print(f"[bold]Loading review:[/bold] {review}")
    with open(review) as f:
        review_text = f.read()

    # Run evaluation
    evaluator = ReviewEvaluator(model=judge_model)
    
    console.print("\n[bold blue]Generating manuscript summary...[/bold blue]")
    summary = evaluator.generate_manuscript_summary(manuscript_content["text"])
    
    console.print("[bold blue]Evaluating review quality...[/bold blue]")
    result = evaluator.evaluate(
        review_text=review_text,
        manuscript_summary=summary,
    )

    # Display results
    console.print("\n" + "=" * 60)
    console.print(result.format_report())

    # Save if output specified
    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]✓[/green] Results saved to: {output}")


@app.command("ab-test")
def ab_test(
    manuscript: Path = typer.Argument(
        ...,
        help="Path to the manuscript PDF file",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for results",
    ),
    configs_file: Optional[Path] = typer.Option(
        None,
        "--configs",
        "-c",
        help="YAML file with test configurations",
    ),
    pubmed_email: Optional[str] = typer.Option(
        None,
        "--email",
        "-e",
        help="Email for PubMed API",
    ),
    judge_model: str = typer.Option(
        "gpt-5-nano",
        "--judge",
        "-j",
        help="Model to use for LLM-as-judge",
    ),
):
    """
    Run A/B tests on different review configurations.
    
    Tests multiple configurations and evaluates which produces
    the highest quality reviews.
    """
    from src.evaluation.ab_test import run_ab_test, create_default_configs, ABTestConfig

    # Get PubMed email
    email = pubmed_email or os.environ.get("PUBMED_EMAIL")

    # Load configs
    configs = None
    if configs_file and configs_file.exists():
        with open(configs_file) as f:
            config_data = yaml.safe_load(f)
            configs = [ABTestConfig.from_dict(c) for c in config_data.get("configs", [])]
    
    if not configs:
        console.print("[dim]Using default test configurations (PubMed for literature)[/dim]")
        configs = create_default_configs()
    
    console.print(f"[bold]Testing {len(configs)} configurations[/bold]")

    # Run A/B tests
    session = run_ab_test(
        manuscript_path=str(manuscript),
        configs=configs,
        judge_model=judge_model,
        pubmed_email=email,
        output_path=str(output) if output else None,
        verbose=True,
    )

    # Print best config details
    best = session.get_best_config()
    if best:
        console.print("\n[bold]Best Configuration Details:[/bold]")
        for key, value in best.to_dict().items():
            if value is not None and key != "name":
                console.print(f"  {key}: {value}")


@app.command("compare-models")
def compare_models(
    manuscript: Path = typer.Argument(
        ...,
        help="Path to the manuscript PDF file",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for results",
    ),
    pubmed_email: Optional[str] = typer.Option(
        None,
        "--email",
        "-e",
        help="Email for PubMed API",
    ),
    judge_model: str = typer.Option(
        "gpt-5-nano",
        "--judge",
        "-j",
        help="Model to use for LLM-as-judge",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Run only 3 configs: baseline, all_premium, midtier_core",
    ),
):
    """
    Compare different model configurations to find cost-effective setups.
    
    Tests combinations of cheap/mid-tier/premium models across agents
    to determine which configurations provide the best quality-per-dollar.
    
    Answers:
    - When is upgrading to expensive models worth it?
    - Which agents benefit most from better models?
    - What's the optimal cost-quality tradeoff?
    """
    from src.evaluation.ab_test import (
        run_ab_test,
        create_model_comparison_configs,
        calculate_cost_effectiveness,
        format_cost_effectiveness_table,
    )

    # Get PubMed email
    email = pubmed_email or os.environ.get("PUBMED_EMAIL")

    # Get model comparison configs
    configs = create_model_comparison_configs()
    
    if quick:
        # Just test baseline, premium, and best-guess middle ground
        configs = [c for c in configs if c.name in ["baseline_cheap", "all_premium", "midtier_core"]]
        console.print(f"[dim]Quick mode: testing {len(configs)} configurations[/dim]")
    else:
        console.print(f"[bold]Testing {len(configs)} model configurations[/bold]")
    
    for config in configs:
        console.print(f"  • {config.name}: {config.description}")
    console.print()

    # Run tests
    session = run_ab_test(
        manuscript_path=str(manuscript),
        configs=configs,
        judge_model=judge_model,
        pubmed_email=email,
        output_path=str(output) if output else None,
        verbose=True,
    )

    # Calculate and display cost-effectiveness
    console.print("\n")
    analysis = calculate_cost_effectiveness(session.results)
    if analysis:
        console.print(format_cost_effectiveness_table(analysis))
        
        # Print recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        
        # Find best value (highest marginal efficiency that improves quality)
        best_value = None
        for row in analysis:
            if not row["is_baseline"] and row["marginal_gain"] > 0:
                eff = row["marginal_efficiency"]
                if isinstance(eff, (int, float)) and eff > 0:
                    best_value = row
                    break
        
        if best_value:
            console.print(f"  [green]✓[/green] Best value: [bold]{best_value['config_name']}[/bold]")
            console.print(f"    Score: {best_value['score']:.1f}/10 (+{best_value['marginal_gain']:.2f} over baseline)")
            console.print(f"    Cost: ${best_value['cost']:.4f} (+${best_value['marginal_cost']:.4f})")
            console.print(f"    Efficiency: {best_value['marginal_efficiency']} points per dollar")
        
        # Find highest quality
        highest_quality = max(analysis, key=lambda x: x["score"])
        if highest_quality["config_name"] != (best_value["config_name"] if best_value else ""):
            console.print(f"\n  [cyan]★[/cyan] Highest quality: [bold]{highest_quality['config_name']}[/bold]")
            console.print(f"    Score: {highest_quality['score']:.1f}/10")
            console.print(f"    Cost: ${highest_quality['cost']:.4f}")
        
        # Identify which agent upgrades matter
        console.print("\n[bold]Agent Upgrade Impact:[/bold]")
        agent_impact = {}
        baseline_score = next((r["score"] for r in analysis if r["is_baseline"]), 0)
        
        for row in analysis:
            name = row["config_name"]
            if "orchestrator" in name and "premium" in name:
                agent_impact["Orchestrator (synthesis)"] = row["marginal_gain"]
            elif "methodologist" in name and "premium" in name and "core" not in name:
                agent_impact["Methodologist (technical)"] = row["marginal_gain"]
            elif "domain_expert" in name and "premium" in name:
                agent_impact["Domain Expert (clinical)"] = row["marginal_gain"]
            elif "core_reviewers" in name:
                agent_impact["Core reviewers (method+domain)"] = row["marginal_gain"]
        
        for agent, gain in sorted(agent_impact.items(), key=lambda x: -x[1]):
            indicator = "[green]↑[/green]" if gain > 0.5 else "[yellow]→[/yellow]" if gain > 0 else "[red]↓[/red]"
            console.print(f"  {indicator} {agent}: {'+' if gain >= 0 else ''}{gain:.2f} points")


@app.command("batch-compare")
def batch_compare(
    folder: Path = typer.Argument(
        Path("test_papers"),
        help="Folder containing PDF manuscripts to test",
    ),
    output: Path = typer.Option(
        Path("output/batch_comparison.json"),
        "--output",
        "-o",
        help="Output JSON file for results",
    ),
    pubmed_email: Optional[str] = typer.Option(
        None,
        "--email",
        "-e",
        help="Email for PubMed API",
    ),
    judge_model: str = typer.Option(
        "gpt-5-nano",
        "--judge",
        "-j",
        help="Model to use for LLM-as-judge",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="[DEPRECATED] Use --suite instead. Forces quick mode (3 configs)",
    ),
    max_papers: Optional[int] = typer.Option(
        None,
        "--max",
        "-m",
        help="Maximum number of papers to test",
    ),
    workers: int = typer.Option(
        3,
        "--workers",
        "-w",
        help="Number of parallel workers (papers tested simultaneously)",
    ),
    parallel_configs: bool = typer.Option(
        False,
        "--parallel-configs",
        "-P",
        help="Also parallelize configs within each paper (more aggressive)",
    ),
    save_reviews: bool = typer.Option(
        False,
        "--save-reviews",
        "-R",
        help="Save full review text (makes output much larger)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Skip tests already in output file (resume interrupted run)",
    ),
    test_suite: str = typer.Option(
        "quick",
        "--suite",
        "-s",
        help="Test suite: quick, full, extended, providers, all",
    ),
):
    """
    Run model comparison tests across all papers in a folder.
    
    Aggregates results to show which model configurations work best
    across different types of manuscripts.
    
    Test suites:
    - quick: 3 configs (baseline, premium, midtier_core)
    - full: 8 configs (all tier permutations)
    - extended: 10 more configs (single agent upgrades, support agents)
    - providers: 9 configs (Claude vs GPT vs Gemini comparisons)
    - all: All available configs (~27)
    
    Options:
    - --workers N: Test N papers simultaneously (default: 3)
    - --save-reviews: Include full review text in output
    - --resume: Skip already-completed tests
    
    Usage:
        reviewer batch-compare --suite extended --save-reviews --resume
    """
    import json
    import statistics
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    from src.evaluation.ab_test import (
        run_ab_test,
        create_model_comparison_configs,
        create_extended_tier_configs,
        create_provider_comparison_configs,
        get_completed_configs,
        ABTestConfig,
        ABTestRunner,
    )
    from src.document import get_manuscript_content
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
    from rich.live import Live

    # Find PDFs in folder
    if not folder.exists():
        console.print(f"[yellow]Folder not found: {folder}[/yellow]")
        console.print(f"Creating folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[cyan]Add PDF manuscripts to {folder}/ and run again.[/cyan]")
        raise typer.Exit(0)
    
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {folder}/[/yellow]")
        console.print(f"\nAdd PDF manuscripts to test and run again.")
        raise typer.Exit(0)
    
    if max_papers:
        pdfs = pdfs[:max_papers]
    
    # Get configs based on test suite
    base_configs = create_model_comparison_configs()
    extended_configs = create_extended_tier_configs()
    provider_configs = create_provider_comparison_configs()
    
    if test_suite == "quick" or quick:
        configs = [c for c in base_configs if c.name in ["baseline_cheap", "all_premium", "midtier_core"]]
    elif test_suite == "full":
        configs = base_configs
    elif test_suite == "extended":
        configs = extended_configs
    elif test_suite == "providers":
        configs = provider_configs
    elif test_suite == "all":
        configs = base_configs + extended_configs + provider_configs
    else:
        console.print(f"[red]Unknown test suite: {test_suite}[/red]")
        console.print("Valid options: quick, full, extended, providers, all")
        raise typer.Exit(1)
    
    # Load completed tests if resuming
    completed_tests = set()
    existing_results = []
    if resume and output.exists():
        completed_tests = get_completed_configs(str(output))
        console.print(f"[dim]Resuming: found {len(completed_tests)} completed tests[/dim]")
        
        # Load existing results to preserve them
        try:
            with open(output) as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("per_paper_results", [])
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Estimate total work
    total_tests = len(pdfs) * len(configs)
    skipped_tests = sum(1 for pdf in pdfs for c in configs if (pdf.name, c.name) in completed_tests)
    
    new_tests = total_tests - skipped_tests
    
    console.print(Panel(
        f"[bold]Batch Model Comparison[/bold]\n\n"
        f"Papers: {len(pdfs)} manuscripts in {folder}/\n"
        f"Suite: {test_suite} ({len(configs)} configs)\n"
        f"Total tests: {total_tests}" + (f" ({skipped_tests} skipped, {new_tests} new)" if skipped_tests else "") + "\n"
        f"Workers: {workers} parallel\n"
        f"Save reviews: {save_reviews}",
        style="blue",
    ))
    
    for pdf in pdfs:
        console.print(f"  • {pdf.name}")
    
    console.print(f"\n[bold]Configurations:[/bold]")
    for config in configs:
        status = ""
        # Check how many papers have this config completed
        completed_for_config = sum(1 for pdf in pdfs if (pdf.name, config.name) in completed_tests)
        if completed_for_config > 0:
            status = f" [dim]({completed_for_config}/{len(pdfs)} done)[/dim]"
        console.print(f"  • {config.name}{status}")
    
    # Get PubMed email
    email = pubmed_email or os.environ.get("PUBMED_EMAIL")
    
    # Thread-safe results collection
    results_lock = threading.Lock()
    all_results = list(existing_results)  # Start with existing results if resuming
    aggregated_scores = {c.name: [] for c in configs}
    aggregated_costs = {c.name: [] for c in configs}
    
    # Pre-populate aggregated data from existing results
    for paper in existing_results:
        for config_name, data in paper.get("results", {}).items():
            if config_name in aggregated_scores and "score" in data:
                aggregated_scores[config_name].append(data["score"])
                aggregated_costs[config_name].append(data["cost"])
    
    # Ensure output directory exists for incremental saves
    output.parent.mkdir(parents=True, exist_ok=True)
    
    def save_incremental_results():
        """Save current results incrementally."""
        with results_lock:
            # Calculate current summary
            current_summary = []
            for config_name in aggregated_scores:
                scores = aggregated_scores[config_name]
                costs = aggregated_costs[config_name]
                if scores:
                    avg_score = statistics.mean(scores)
                    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
                    avg_cost = statistics.mean(costs)
                    score_per_dollar = avg_score / avg_cost if avg_cost > 0 else 0
                    current_summary.append({
                        "config": config_name,
                        "avg_score": avg_score,
                        "std_dev": std_dev,
                        "avg_cost": avg_cost,
                        "score_per_dollar": score_per_dollar,
                        "n_papers": len(scores),
                    })
            
            output_data = {
                "folder": str(folder),
                "n_papers_total": len(pdfs),
                "n_papers_completed": len(all_results),
                "test_suite": test_suite,
                "configs_tested": [c.name for c in configs],
                "workers": workers,
                "save_reviews": save_reviews,
                "status": "in_progress" if len(all_results) < len(pdfs) else "completed",
                "summary": current_summary,
                "per_paper_results": list(all_results),
            }
            
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
    
    def test_single_paper(pdf: Path) -> dict:
        """Test a single paper with all configs."""
        # Filter configs to only those not yet completed for this paper
        configs_to_run = [c for c in configs if (pdf.name, c.name) not in completed_tests]
        
        if not configs_to_run:
            # All configs already done for this paper, return existing results
            for paper in existing_results:
                if paper.get("manuscript") == pdf.name:
                    return paper
            return {"manuscript": pdf.name, "results": {}, "skipped": True}
        
        try:
            # Run only the configs that haven't been completed
            session = run_ab_test(
                manuscript_path=str(pdf),
                configs=configs_to_run,
                judge_model=judge_model,
                pubmed_email=email,
                verbose=False,
            )
            
            paper_result = {
                "manuscript": pdf.name,
                "results": {},
            }
            
            # Add results from existing data first
            for paper in existing_results:
                if paper.get("manuscript") == pdf.name:
                    paper_result["results"].update(paper.get("results", {}))
                    break
            
            # Add new results
            for result in session.results:
                if not result.error and result.evaluation:
                    config_name = result.config.name
                    result_data = {
                        "score": result.evaluation.weighted_score,
                        "cost": result.cost_usd,
                        "grade": result.evaluation.get_grade(),
                    }
                    
                    # Include full review if requested
                    if save_reviews and result.review_session:
                        result_data["final_review"] = result.review_session.final_review
                        result_data["initial_reviews"] = result.review_session.initial_reviews
                        result_data["evaluation_details"] = result.evaluation.to_dict()
                    
                    paper_result["results"][config_name] = result_data
            
            return paper_result
            
        except Exception as e:
            return {
                "manuscript": pdf.name,
                "error": str(e),
            }
    
    # Run tests in parallel
    console.print(f"\n[bold]Running tests with {workers} workers...[/bold]\n")
    console.print(f"[dim]Results saved incrementally to: {output}[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Testing {len(pdfs)} papers...", total=len(pdfs))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all papers
            future_to_pdf = {executor.submit(test_single_paper, pdf): pdf for pdf in pdfs}
            
            # Process results as they complete
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    paper_result = future.result()
                    
                    with results_lock:
                        # Check if this paper already exists in results (from resume)
                        existing_idx = None
                        for i, r in enumerate(all_results):
                            if r.get("manuscript") == paper_result.get("manuscript"):
                                existing_idx = i
                                break
                        
                        if existing_idx is not None:
                            # Merge new results into existing
                            all_results[existing_idx]["results"].update(paper_result.get("results", {}))
                        else:
                            all_results.append(paper_result)
                        
                        # Only aggregate NEW scores (not from resume)
                        if "results" in paper_result and not paper_result.get("skipped"):
                            for config_name, data in paper_result["results"].items():
                                # Skip if this was already in completed_tests
                                if (pdf.name, config_name) not in completed_tests:
                                    if "score" in data and config_name in aggregated_scores:
                                        aggregated_scores[config_name].append(data["score"])
                                        aggregated_costs[config_name].append(data["cost"])
                        
                        if "error" in paper_result:
                            console.print(f"  [red]✗[/red] {pdf.name}: {paper_result['error'][:50]}")
                        else:
                            n_successful = len(paper_result.get("results", {}))
                            console.print(f"  [green]✓[/green] {pdf.name} ({n_successful}/{len(configs)} configs)")
                    
                    # Save incrementally after each paper completes
                    save_incremental_results()
                    
                except Exception as e:
                    console.print(f"  [red]✗[/red] {pdf.name}: {str(e)[:50]}")
                    with results_lock:
                        all_results.append({"manuscript": pdf.name, "error": str(e)})
                    save_incremental_results()
                
                progress.advance(task)
    
    # Calculate aggregated statistics
    console.print(f"\n[bold]Aggregated Results ({len(pdfs)} papers):[/bold]\n")
    
    summary_table = Table(title="Model Configuration Comparison (Averaged)")
    summary_table.add_column("Config", style="cyan")
    summary_table.add_column("Avg Score", justify="center")
    summary_table.add_column("Std Dev", justify="center")
    summary_table.add_column("Avg Cost", justify="center")
    summary_table.add_column("Score/$", justify="center", style="green")
    summary_table.add_column("Papers", justify="center")
    
    summary_data = []
    for config_name in aggregated_scores:
        scores = aggregated_scores[config_name]
        costs = aggregated_costs[config_name]
        
        if scores:
            avg_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
            avg_cost = statistics.mean(costs)
            score_per_dollar = avg_score / avg_cost if avg_cost > 0 else 0
            
            summary_data.append({
                "config": config_name,
                "avg_score": avg_score,
                "std_dev": std_dev,
                "avg_cost": avg_cost,
                "score_per_dollar": score_per_dollar,
                "n_papers": len(scores),
            })
            
            summary_table.add_row(
                config_name,
                f"{avg_score:.2f}",
                f"±{std_dev:.2f}",
                f"${avg_cost:.4f}",
                f"{score_per_dollar:.1f}",
                str(len(scores)),
            )
    
    console.print(summary_table)
    
    # Find best configurations
    if summary_data:
        # Sort by avg_score
        sorted_by_score = sorted(summary_data, key=lambda x: x["avg_score"], reverse=True)
        sorted_by_value = sorted(summary_data, key=lambda x: x["score_per_dollar"], reverse=True)
        
        console.print("\n[bold]Recommendations:[/bold]")
        
        best_quality = sorted_by_score[0]
        console.print(f"  [cyan]★[/cyan] Highest quality: [bold]{best_quality['config']}[/bold]")
        console.print(f"    Avg score: {best_quality['avg_score']:.2f} ± {best_quality['std_dev']:.2f}")
        console.print(f"    Avg cost: ${best_quality['avg_cost']:.4f}")
        
        best_value = sorted_by_value[0]
        if best_value["config"] != best_quality["config"]:
            console.print(f"\n  [green]✓[/green] Best value: [bold]{best_value['config']}[/bold]")
            console.print(f"    Avg score: {best_value['avg_score']:.2f}")
            console.print(f"    Score per $: {best_value['score_per_dollar']:.1f}")
        
        # Calculate if premium is worth it
        baseline = next((d for d in summary_data if "baseline" in d["config"]), None)
        premium = next((d for d in summary_data if "premium" in d["config"] and "all" in d["config"]), None)
        
        if baseline and premium:
            score_gain = premium["avg_score"] - baseline["avg_score"]
            cost_increase = premium["avg_cost"] - baseline["avg_cost"]
            
            console.print(f"\n[bold]Premium vs Baseline:[/bold]")
            console.print(f"  Score improvement: +{score_gain:.2f} points")
            console.print(f"  Cost increase: +${cost_increase:.4f}")
            
            if cost_increase > 0:
                efficiency = score_gain / cost_increase
                console.print(f"  Efficiency: {efficiency:.2f} points per dollar")
                
                if efficiency > 1:
                    console.print(f"  [green]→ Premium models ARE worth the cost[/green]")
                else:
                    console.print(f"  [yellow]→ Premium models may NOT be worth the cost[/yellow]")
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "folder": str(folder),
        "n_papers": len(pdfs),
        "configs_tested": [c.name for c in configs],
        "workers": workers,
        "parallel_configs": parallel_configs,
        "summary": summary_data,
        "per_paper_results": all_results,
    }
    
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    console.print(f"\n[green]✓[/green] Results saved to: {output}")
    console.print(f"\n[dim]Per-paper details available in the JSON output.[/dim]")


if __name__ == "__main__":
    app()

