"""Provider comparison tool for literature search evaluation."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.document import get_manuscript_content
from src.search_providers import PROVIDERS, get_provider, list_providers

from .literature_eval import LiteratureEvaluator, LiteratureEvalResult

console = Console()


@dataclass
class ProviderComparison:
    """Results of comparing multiple providers."""
    
    manuscript_path: str
    manuscript_summary: str
    provider_results: dict[str, list[dict]] = field(default_factory=dict)
    evaluations: dict[str, LiteratureEvalResult] = field(default_factory=dict)
    rankings: list[tuple[str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "manuscript_path": self.manuscript_path,
            "manuscript_summary": self.manuscript_summary,
            "provider_results": {
                name: [p for p in papers]
                for name, papers in self.provider_results.items()
            },
            "evaluations": {
                name: result.to_dict()
                for name, result in self.evaluations.items()
            },
            "rankings": self.rankings,
            "timestamp": self.timestamp,
        }
    
    def save(self, output_path: str | Path):
        """Save comparison results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def format_summary_table(self) -> Table:
        """Create a Rich table summarizing the comparison."""
        table = Table(title="Provider Comparison Results")
        
        table.add_column("Rank", justify="center", style="bold")
        table.add_column("Provider", style="cyan")
        table.add_column("Papers", justify="center")
        table.add_column("Weighted Score", justify="center")
        table.add_column("Relevance", justify="center")
        table.add_column("Methods", justify="center")
        table.add_column("Recency", justify="center")
        
        for rank, (provider, score) in enumerate(self.rankings, 1):
            result = self.evaluations.get(provider)
            if not result or result.error:
                table.add_row(
                    str(rank),
                    provider,
                    "0",
                    "Error",
                    "-",
                    "-",
                    "-",
                )
            else:
                scores = result.scores
                table.add_row(
                    str(rank),
                    provider,
                    str(result.papers_count),
                    f"{score:.1f}",
                    f"{scores.get('relevance', 0):.0f}",
                    f"{scores.get('methodological_match', 0):.0f}",
                    f"{scores.get('recency', 0):.0f}",
                )
        
        return table
    
    def get_winner(self) -> Optional[str]:
        """Get the winning provider."""
        if self.rankings:
            return self.rankings[0][0]
        return None
    
    def format_detailed_report(self) -> str:
        """Format detailed comparison report."""
        lines = [
            "# Literature Provider Comparison Report",
            f"**Manuscript:** {self.manuscript_path}",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            "",
        ]
        
        if self.rankings:
            winner = self.rankings[0]
            lines.append(f"**Winner:** {winner[0]} (score: {winner[1]:.1f}/10)")
            lines.append("")
        
        # Rankings table
        lines.append("### Rankings")
        lines.append("")
        lines.append("| Rank | Provider | Papers | Score |")
        lines.append("|------|----------|--------|-------|")
        
        for rank, (provider, score) in enumerate(self.rankings, 1):
            result = self.evaluations.get(provider)
            papers = result.papers_count if result else 0
            lines.append(f"| {rank} | {provider} | {papers} | {score:.1f} |")
        
        lines.append("")
        
        # Detailed results per provider
        lines.append("## Detailed Evaluations")
        lines.append("")
        
        for provider, _ in self.rankings:
            result = self.evaluations.get(provider)
            if result:
                lines.append(result.format_report())
                lines.append("")
        
        return "\n".join(lines)


def compare_all_providers(
    manuscript_path: str,
    max_results: int = 10,
    providers: list[str] | None = None,
    judge_model: str = "gpt-5-nano",
    verbose: bool = True,
) -> ProviderComparison:
    """
    Compare all available providers on a single manuscript.
    
    Args:
        manuscript_path: Path to manuscript PDF
        max_results: Maximum papers per provider
        providers: List of provider names to test (None = all available)
        judge_model: Model to use for LLM-as-judge
        verbose: Whether to print progress
        
    Returns:
        ProviderComparison with results and rankings
    """
    # Load manuscript
    if verbose:
        console.print(Panel(f"[bold]Comparing Literature Providers[/bold]\n\nManuscript: {manuscript_path}"))
    
    manuscript = get_manuscript_content(manuscript_path)
    manuscript_text = manuscript["text"]
    
    # Generate manuscript summary
    if verbose:
        console.print("[bold blue]Generating manuscript summary...[/bold blue]")
    
    evaluator = LiteratureEvaluator(model=judge_model)
    manuscript_summary = evaluator.generate_manuscript_summary(manuscript_text)
    
    if verbose:
        console.print(Panel(manuscript_summary, title="Manuscript Summary", border_style="dim"))
    
    # Determine which providers to test
    available_providers = []
    for p in list_providers():
        if p["name"] in PROVIDERS:
            provider_instance = get_provider(p["name"])
            if provider_instance.is_available():
                available_providers.append(p["name"])
    
    if providers:
        # Filter to requested providers
        test_providers = [p for p in providers if p in available_providers]
    else:
        test_providers = available_providers
    
    if verbose:
        console.print(f"[dim]Testing providers: {', '.join(test_providers)}[/dim]")
    
    # Collect results from each provider
    provider_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Searching...", total=len(test_providers))
        
        for provider_name in test_providers:
            progress.update(task, description=f"[bold]{provider_name}[/bold]...")
            
            try:
                provider = get_provider(provider_name)
                session = provider.search(
                    manuscript_text=manuscript_text,
                    max_results=max_results,
                    focus_pubmed=True,
                )
                
                # Convert SearchResult objects to dicts
                papers = []
                for result in session.results:
                    papers.append({
                        "title": result.title,
                        "authors": result.authors,
                        "abstract": result.abstract,
                        "pmid": result.pmid,
                        "doi": result.doi,
                        "journal": result.journal,
                        "pub_date": result.pub_date,
                        "url": result.url,
                        "relevance_score": result.relevance_score,
                        "relevance_reason": result.relevance_reason,
                    })
                
                provider_results[provider_name] = papers
                
                if verbose:
                    console.print(f"  [green]✓[/green] {provider_name}: {len(papers)} papers")
                    
            except Exception as e:
                if verbose:
                    console.print(f"  [red]✗[/red] {provider_name}: {e}")
                provider_results[provider_name] = []
            
            progress.advance(task)
    
    # Evaluate each provider's results
    if verbose:
        console.print("\n[bold blue]Evaluating results with LLM-as-judge...[/bold blue]")
    
    evaluations = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(provider_results))
        
        for provider_name, papers in provider_results.items():
            progress.update(task, description=f"[bold]Evaluating {provider_name}[/bold]...")
            
            result = evaluator.evaluate(
                provider_name=provider_name,
                papers=papers,
                manuscript_summary=manuscript_summary,
            )
            evaluations[provider_name] = result
            
            if verbose:
                if result.error:
                    console.print(f"  [yellow]⚠[/yellow] {provider_name}: {result.error}")
                else:
                    console.print(f"  [green]✓[/green] {provider_name}: {result.weighted_score:.1f}/10")
            
            progress.advance(task)
    
    # Rank providers
    rankings = evaluator.rank_providers(evaluations)
    
    # Create comparison object
    comparison = ProviderComparison(
        manuscript_path=manuscript_path,
        manuscript_summary=manuscript_summary,
        provider_results=provider_results,
        evaluations=evaluations,
        rankings=rankings,
    )
    
    # Display results
    if verbose:
        console.print()
        console.print(comparison.format_summary_table())
        
        if rankings:
            winner = rankings[0]
            console.print(f"\n[bold green]Winner: {winner[0]} (score: {winner[1]:.1f}/10)[/bold green]")
    
    return comparison


async def compare_all_providers_async(
    manuscript_path: str,
    max_results: int = 10,
    providers: list[str] | None = None,
    judge_model: str = "gpt-5-nano",
) -> ProviderComparison:
    """Async version for web interface."""
    import asyncio
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: compare_all_providers(
            manuscript_path=manuscript_path,
            max_results=max_results,
            providers=providers,
            judge_model=judge_model,
            verbose=False,
        )
    )

