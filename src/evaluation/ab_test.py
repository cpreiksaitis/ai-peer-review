"""A/B testing framework for review configuration optimization."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.document import get_manuscript_content
from src.orchestrator import Orchestrator, create_orchestrator_from_config, ReviewSession

from .review_eval import ReviewEvaluator, ReviewEvalResult

console = Console()


@dataclass
class ABTestConfig:
    """Configuration for an A/B test variant."""
    
    name: str
    description: str = ""
    
    # Literature search config
    search_literature: bool = True
    literature_provider: str = "pubmed"
    max_literature_results: int = 10
    
    # Debate config
    debate_rounds: int = 2
    temperature: float = 0.7
    
    # Model config (overrides)
    orchestrator_model: Optional[str] = None
    methodologist_model: Optional[str] = None
    domain_expert_model: Optional[str] = None
    communication_model: Optional[str] = None
    ethics_model: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "search_literature": self.search_literature,
            "literature_provider": self.literature_provider,
            "max_literature_results": self.max_literature_results,
            "debate_rounds": self.debate_rounds,
            "temperature": self.temperature,
            "orchestrator_model": self.orchestrator_model,
            "methodologist_model": self.methodologist_model,
            "domain_expert_model": self.domain_expert_model,
            "communication_model": self.communication_model,
            "ethics_model": self.ethics_model,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ABTestConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ABTestResult:
    """Result of an A/B test variant."""
    
    config: ABTestConfig
    review_session: Optional[ReviewSession] = None
    evaluation: Optional[ReviewEvalResult] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "final_review": self.review_session.final_review if self.review_session else None,
        }


@dataclass
class ABTestSession:
    """Complete A/B test session results."""
    
    manuscript_path: str
    manuscript_summary: str = ""
    results: list[ABTestResult] = field(default_factory=list)
    rankings: list[tuple[str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "manuscript_path": self.manuscript_path,
            "manuscript_summary": self.manuscript_summary,
            "results": [r.to_dict() for r in self.results],
            "rankings": self.rankings,
            "timestamp": self.timestamp,
        }
    
    def save(self, output_path: str | Path):
        """Save to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def format_summary_table(self) -> Table:
        """Create summary table."""
        table = Table(title="A/B Test Results")
        
        table.add_column("Rank", justify="center", style="bold")
        table.add_column("Config", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Grade", justify="center")
        table.add_column("Time (s)", justify="center")
        table.add_column("Cost ($)", justify="center")
        
        for rank, (name, score) in enumerate(self.rankings, 1):
            result = next((r for r in self.results if r.config.name == name), None)
            if result and result.evaluation:
                grade = result.evaluation.get_grade()
                time_str = f"{result.duration_seconds:.1f}"
                cost_str = f"{result.cost_usd:.4f}"
            else:
                grade = "Error"
                time_str = "-"
                cost_str = "-"
            
            table.add_row(
                str(rank),
                name,
                f"{score:.1f}",
                grade,
                time_str,
                cost_str,
            )
        
        return table
    
    def get_best_config(self) -> Optional[ABTestConfig]:
        """Get the best performing configuration."""
        if self.rankings:
            best_name = self.rankings[0][0]
            for result in self.results:
                if result.config.name == best_name:
                    return result.config
        return None


def create_default_configs() -> list[ABTestConfig]:
    """Create a set of default A/B test configurations."""
    return [
        ABTestConfig(
            name="baseline",
            description="Default: PubMed literature, 2 debate rounds",
            search_literature=True,
            literature_provider="pubmed",
            debate_rounds=2,
        ),
        ABTestConfig(
            name="no_literature",
            description="No literature search - tests if lit adds value",
            search_literature=False,
            debate_rounds=2,
        ),
        ABTestConfig(
            name="extended_debate",
            description="3 debate rounds - more agent discussion",
            search_literature=True,
            literature_provider="pubmed",
            debate_rounds=3,
        ),
        ABTestConfig(
            name="minimal_debate",
            description="1 debate round - faster, less consensus",
            search_literature=True,
            literature_provider="pubmed",
            debate_rounds=1,
        ),
        ABTestConfig(
            name="high_temperature",
            description="Higher temperature (0.9) for more creative reviews",
            search_literature=True,
            literature_provider="pubmed",
            debate_rounds=2,
            temperature=0.9,
        ),
        ABTestConfig(
            name="low_temperature",
            description="Lower temperature (0.5) for more focused reviews",
            search_literature=True,
            literature_provider="pubmed",
            debate_rounds=2,
            temperature=0.5,
        ),
    ]


def create_model_comparison_configs() -> list[ABTestConfig]:
    """
    Create configs to test model quality vs cost tradeoffs.
    
    Tests:
    1. Baseline (cheap models everywhere)
    2. Premium models everywhere (upper bound on quality)
    3. Upgraded orchestrator only (synthesis quality)
    4. Upgraded methodologist only (technical accuracy)
    5. Upgraded domain expert only (clinical relevance)
    6. Upgraded methodologist + domain expert (core reviewers)
    """
    # Cheap models (baseline)
    CHEAP_ORCHESTRATOR = "claude-haiku-4-5"
    CHEAP_METHODOLOGIST = "gpt-5-nano"
    CHEAP_DOMAIN = "gemini/gemini-flash-lite-latest"
    CHEAP_COMMUNICATION = "gemini/gemini-flash-lite-latest"
    CHEAP_ETHICS = "gpt-5-nano"
    
    # Premium models (using gpt-4o instead of gpt-5.1 due to temperature restrictions)
    PREMIUM_ORCHESTRATOR = "claude-sonnet-4-5"
    PREMIUM_METHODOLOGIST = "gpt-4o"
    PREMIUM_DOMAIN = "gemini/gemini-3-pro-preview"
    PREMIUM_COMMUNICATION = "claude-sonnet-4-5"
    PREMIUM_ETHICS = "claude-sonnet-4-5"
    
    # Mid-tier models
    MID_ORCHESTRATOR = "claude-sonnet-4-5"
    MID_METHODOLOGIST = "gpt-5-mini"
    MID_DOMAIN = "gemini/gemini-flash-latest"
    MID_COMMUNICATION = "gpt-5-mini"
    MID_ETHICS = "gpt-5-mini"
    
    return [
        # Baseline - cheap everywhere
        ABTestConfig(
            name="baseline_cheap",
            description="Baseline: all cheap models (~$0.06)",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Premium everywhere (quality upper bound)
        ABTestConfig(
            name="all_premium",
            description="All premium models (~$0.95) - quality ceiling",
            orchestrator_model=PREMIUM_ORCHESTRATOR,
            methodologist_model=PREMIUM_METHODOLOGIST,
            domain_expert_model=PREMIUM_DOMAIN,
            communication_model=PREMIUM_COMMUNICATION,
            ethics_model=PREMIUM_ETHICS,
        ),
        
        # Upgrade orchestrator only (affects synthesis)
        ABTestConfig(
            name="premium_orchestrator",
            description="Premium orchestrator only - tests synthesis quality",
            orchestrator_model=PREMIUM_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Upgrade methodologist only (affects technical accuracy)
        ABTestConfig(
            name="premium_methodologist",
            description="Premium methodologist - tests technical accuracy",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=PREMIUM_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Upgrade domain expert only (affects clinical relevance)
        ABTestConfig(
            name="premium_domain_expert",
            description="Premium domain expert - tests clinical depth",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=PREMIUM_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Upgrade core reviewers (methodologist + domain expert)
        ABTestConfig(
            name="premium_core_reviewers",
            description="Premium methodologist + domain expert",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=PREMIUM_METHODOLOGIST,
            domain_expert_model=PREMIUM_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Mid-tier everywhere (balanced)
        ABTestConfig(
            name="all_midtier",
            description="All mid-tier models (~$0.25) - balanced",
            orchestrator_model=MID_ORCHESTRATOR,
            methodologist_model=MID_METHODOLOGIST,
            domain_expert_model=MID_DOMAIN,
            communication_model=MID_COMMUNICATION,
            ethics_model=MID_ETHICS,
        ),
        
        # Mid-tier core + cheap support
        ABTestConfig(
            name="midtier_core",
            description="Mid-tier orchestrator/methodologist/domain, cheap rest",
            orchestrator_model=MID_ORCHESTRATOR,
            methodologist_model=MID_METHODOLOGIST,
            domain_expert_model=MID_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
    ]


def create_extended_tier_configs() -> list[ABTestConfig]:
    """
    Create extended configs testing more tier permutations.
    
    Covers combinations not in the basic set:
    - Mid-tier single agent upgrades
    - Communication/Ethics upgrades
    - Mixed tier combinations
    """
    # Cheap models (baseline)
    CHEAP_ORCHESTRATOR = "claude-haiku-4-5"
    CHEAP_METHODOLOGIST = "gpt-5-nano"
    CHEAP_DOMAIN = "gemini/gemini-flash-lite-latest"
    CHEAP_COMMUNICATION = "gemini/gemini-flash-lite-latest"
    CHEAP_ETHICS = "gpt-5-nano"
    
    # Mid-tier models
    MID_ORCHESTRATOR = "claude-sonnet-4-5"
    MID_METHODOLOGIST = "gpt-5-mini"
    MID_DOMAIN = "gemini/gemini-flash-latest"
    MID_COMMUNICATION = "gpt-5-mini"
    MID_ETHICS = "gpt-5-mini"
    
    # Premium models
    PREMIUM_ORCHESTRATOR = "claude-sonnet-4-5"
    PREMIUM_METHODOLOGIST = "gpt-4o"
    PREMIUM_DOMAIN = "gemini/gemini-3-pro-preview"
    PREMIUM_COMMUNICATION = "claude-sonnet-4-5"
    PREMIUM_ETHICS = "claude-sonnet-4-5"
    
    return [
        # Mid-tier single agent upgrades (from cheap baseline)
        ABTestConfig(
            name="midtier_orchestrator",
            description="Mid-tier orchestrator only",
            orchestrator_model=MID_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        ABTestConfig(
            name="midtier_methodologist",
            description="Mid-tier methodologist only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=MID_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        ABTestConfig(
            name="midtier_domain",
            description="Mid-tier domain expert only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=MID_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        
        # Communication and Ethics upgrades (often overlooked)
        ABTestConfig(
            name="midtier_communication",
            description="Mid-tier communication only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=MID_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        ABTestConfig(
            name="midtier_ethics",
            description="Mid-tier ethics only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=MID_ETHICS,
        ),
        ABTestConfig(
            name="midtier_support",
            description="Mid-tier communication + ethics (support agents)",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=MID_COMMUNICATION,
            ethics_model=MID_ETHICS,
        ),
        
        # Premium communication/ethics
        ABTestConfig(
            name="premium_communication",
            description="Premium communication only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=PREMIUM_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        ABTestConfig(
            name="premium_ethics",
            description="Premium ethics only",
            orchestrator_model=CHEAP_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=PREMIUM_ETHICS,
        ),
        
        # Orchestrator + one reviewer combinations
        ABTestConfig(
            name="midtier_orch_method",
            description="Mid-tier orchestrator + methodologist",
            orchestrator_model=MID_ORCHESTRATOR,
            methodologist_model=MID_METHODOLOGIST,
            domain_expert_model=CHEAP_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
        ABTestConfig(
            name="midtier_orch_domain",
            description="Mid-tier orchestrator + domain expert",
            orchestrator_model=MID_ORCHESTRATOR,
            methodologist_model=CHEAP_METHODOLOGIST,
            domain_expert_model=MID_DOMAIN,
            communication_model=CHEAP_COMMUNICATION,
            ethics_model=CHEAP_ETHICS,
        ),
    ]


def create_provider_comparison_configs() -> list[ABTestConfig]:
    """
    Create configs testing different model providers in each role.
    
    Tests Claude vs GPT vs Gemini in key roles to see if
    certain providers excel at certain tasks.
    """
    # Best models from each provider (mid-tier cost)
    CLAUDE_MID = "claude-sonnet-4-5"
    GPT_MID = "gpt-5-mini"  
    GEMINI_MID = "gemini/gemini-flash-latest"
    
    # Cheap fallbacks
    CHEAP = "gpt-5-nano"
    CHEAP_GEMINI = "gemini/gemini-flash-lite-latest"
    
    return [
        # All same provider
        ABTestConfig(
            name="all_claude",
            description="All agents use Claude",
            orchestrator_model=CLAUDE_MID,
            methodologist_model=CLAUDE_MID,
            domain_expert_model=CLAUDE_MID,
            communication_model=CLAUDE_MID,
            ethics_model=CLAUDE_MID,
        ),
        ABTestConfig(
            name="all_gpt",
            description="All agents use GPT",
            orchestrator_model=GPT_MID,
            methodologist_model=GPT_MID,
            domain_expert_model=GPT_MID,
            communication_model=GPT_MID,
            ethics_model=GPT_MID,
        ),
        ABTestConfig(
            name="all_gemini",
            description="All agents use Gemini",
            orchestrator_model=GEMINI_MID,
            methodologist_model=GEMINI_MID,
            domain_expert_model=GEMINI_MID,
            communication_model=GEMINI_MID,
            ethics_model=GEMINI_MID,
        ),
        
        # Orchestrator provider tests (with cheap agents)
        ABTestConfig(
            name="claude_orchestrator",
            description="Claude orchestrator, cheap agents",
            orchestrator_model=CLAUDE_MID,
            methodologist_model=CHEAP,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
        ABTestConfig(
            name="gpt_orchestrator",
            description="GPT orchestrator, cheap agents",
            orchestrator_model=GPT_MID,
            methodologist_model=CHEAP,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
        ABTestConfig(
            name="gemini_orchestrator",
            description="Gemini orchestrator, cheap agents",
            orchestrator_model=GEMINI_MID,
            methodologist_model=CHEAP,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
        
        # Methodologist provider tests
        ABTestConfig(
            name="claude_methodologist",
            description="Claude methodologist, cheap rest",
            orchestrator_model="claude-haiku-4-5",
            methodologist_model=CLAUDE_MID,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
        ABTestConfig(
            name="gpt_methodologist",
            description="GPT methodologist, cheap rest",
            orchestrator_model="claude-haiku-4-5",
            methodologist_model=GPT_MID,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
        ABTestConfig(
            name="gemini_methodologist",
            description="Gemini methodologist, cheap rest",
            orchestrator_model="claude-haiku-4-5",
            methodologist_model=GEMINI_MID,
            domain_expert_model=CHEAP_GEMINI,
            communication_model=CHEAP_GEMINI,
            ethics_model=CHEAP,
        ),
    ]


def get_completed_configs(results_file: str) -> set[tuple[str, str]]:
    """
    Get set of (manuscript, config_name) pairs already completed.
    
    Used to skip tests that have already been run.
    """
    import json
    from pathlib import Path
    
    completed = set()
    path = Path(results_file)
    
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            
            for paper in data.get("per_paper_results", []):
                manuscript = paper.get("manuscript", "")
                for config_name in paper.get("results", {}).keys():
                    completed.add((manuscript, config_name))
        except (json.JSONDecodeError, KeyError):
            pass
    
    return completed


def calculate_cost_effectiveness(results: list["ABTestResult"]) -> list[dict]:
    """
    Calculate cost-effectiveness metrics for A/B test results.
    
    Returns list of dicts with:
    - config_name: Name of the configuration
    - score: Quality score (0-10)
    - cost: Cost in USD
    - score_per_dollar: Quality points per dollar spent
    - marginal_gain: Score improvement over baseline
    - marginal_cost: Cost increase over baseline
    - marginal_efficiency: Score gain per dollar over baseline
    """
    if not results:
        return []
    
    # Find baseline (cheapest config or first one)
    baseline = min(
        [r for r in results if not r.error and r.evaluation],
        key=lambda r: r.cost_usd,
        default=None
    )
    
    if not baseline:
        return []
    
    baseline_score = baseline.evaluation.weighted_score
    baseline_cost = baseline.cost_usd
    
    analysis = []
    for result in results:
        if result.error or not result.evaluation:
            continue
            
        score = result.evaluation.weighted_score
        cost = result.cost_usd
        
        # Avoid division by zero
        score_per_dollar = score / cost if cost > 0 else 0
        
        marginal_gain = score - baseline_score
        marginal_cost = cost - baseline_cost
        
        # Marginal efficiency: how much score improvement per additional dollar?
        if marginal_cost > 0:
            marginal_efficiency = marginal_gain / marginal_cost
        elif marginal_cost == 0:
            marginal_efficiency = float('inf') if marginal_gain > 0 else 0
        else:
            # Cost decreased - any score maintained/gained is very efficient
            marginal_efficiency = float('inf') if marginal_gain >= 0 else marginal_gain / marginal_cost
        
        analysis.append({
            "config_name": result.config.name,
            "score": round(score, 2),
            "cost": round(cost, 4),
            "score_per_dollar": round(score_per_dollar, 2),
            "marginal_gain": round(marginal_gain, 2),
            "marginal_cost": round(marginal_cost, 4),
            "marginal_efficiency": round(marginal_efficiency, 2) if marginal_efficiency != float('inf') else "∞",
            "is_baseline": result.config.name == baseline.config.name,
        })
    
    # Sort by marginal efficiency (best value first)
    analysis.sort(
        key=lambda x: x["marginal_efficiency"] if isinstance(x["marginal_efficiency"], (int, float)) else 999,
        reverse=True
    )
    
    return analysis


def format_cost_effectiveness_table(analysis: list[dict]) -> Table:
    """Format cost-effectiveness analysis as a Rich table."""
    table = Table(title="Cost-Effectiveness Analysis")
    
    table.add_column("Config", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Cost", justify="center")
    table.add_column("Score/$", justify="center", style="green")
    table.add_column("Δ Score", justify="center")
    table.add_column("Δ Cost", justify="center")
    table.add_column("Efficiency", justify="center", style="bold")
    
    for row in analysis:
        # Highlight baseline
        style = "dim" if row["is_baseline"] else None
        
        # Format efficiency
        eff = row["marginal_efficiency"]
        if eff == "∞":
            eff_str = "∞"
            eff_style = "bold green"
        elif isinstance(eff, (int, float)):
            if eff > 10:
                eff_str = f"+{eff:.1f}"
                eff_style = "bold green"
            elif eff > 0:
                eff_str = f"+{eff:.1f}"
                eff_style = "green"
            elif eff < 0:
                eff_str = f"{eff:.1f}"
                eff_style = "red"
            else:
                eff_str = "0"
                eff_style = "dim"
        else:
            eff_str = str(eff)
            eff_style = None
        
        table.add_row(
            row["config_name"] + (" (baseline)" if row["is_baseline"] else ""),
            f"{row['score']:.1f}",
            f"${row['cost']:.4f}",
            f"{row['score_per_dollar']:.1f}",
            f"+{row['marginal_gain']:.2f}" if row["marginal_gain"] >= 0 else f"{row['marginal_gain']:.2f}",
            f"+${row['marginal_cost']:.4f}" if row["marginal_cost"] >= 0 else f"-${abs(row['marginal_cost']):.4f}",
            f"[{eff_style}]{eff_str}[/{eff_style}]" if eff_style else eff_str,
            style=style,
        )
    
    return table


class ABTestRunner:
    """Runs A/B tests on review configurations."""
    
    def __init__(
        self,
        judge_model: str = "gpt-5-nano",
        base_config: Optional[dict] = None,
    ):
        self.judge_model = judge_model
        self.base_config = base_config or {}
        self.evaluator = ReviewEvaluator(model=judge_model)
    
    def run_single_config(
        self,
        config: ABTestConfig,
        manuscript_text: str,
        manuscript_path: str,
        pubmed_email: Optional[str] = None,
        verbose: bool = True,
    ) -> ABTestResult:
        """Run a single configuration and evaluate the result."""
        import time
        
        start_time = time.time()
        
        try:
            # Build orchestrator config
            orchestrator_config = self.base_config.copy()
            
            # Override models if specified
            models = orchestrator_config.get("models", {}).copy()
            if config.orchestrator_model:
                models["orchestrator"] = config.orchestrator_model
            if config.methodologist_model:
                models["methodologist"] = config.methodologist_model
            if config.domain_expert_model:
                models["domain_expert"] = config.domain_expert_model
            if config.communication_model:
                models["communication"] = config.communication_model
            if config.ethics_model:
                models["ethics"] = config.ethics_model
            orchestrator_config["models"] = models
            
            # Set debate rounds
            orchestrator_config.setdefault("debate", {})
            orchestrator_config["debate"]["rounds"] = config.debate_rounds
            orchestrator_config["debate"]["temperature"] = config.temperature
            
            # Create orchestrator
            orchestrator = create_orchestrator_from_config(orchestrator_config)
            
            # Handle literature search
            if config.search_literature and config.literature_provider != "pubmed":
                # Use provider system for non-pubmed
                from src.search_providers import get_provider
                from src.literature import format_literature_context
                
                provider = get_provider(config.literature_provider)
                session = provider.search(
                    manuscript_text=manuscript_text,
                    max_results=config.max_literature_results,
                    focus_pubmed=True,
                )
                
                # Convert to literature context
                from src.literature import RelatedPaper
                papers = []
                for r in session.results:
                    papers.append(RelatedPaper(
                        pmid=r.pmid or "",
                        title=r.title,
                        authors=r.authors,
                        abstract=r.abstract,
                        journal=r.journal or "",
                        pub_date=r.pub_date or "",
                        doi=r.doi,
                        relevance_score=r.relevance_score,
                        relevance_reason=r.relevance_reason,
                    ))
                literature_context = format_literature_context(papers)
                
                # Run review without built-in literature search
                # We need to inject the literature context
                review_session = orchestrator.run_review(
                    manuscript_text=manuscript_text,
                    manuscript_path=manuscript_path,
                    search_literature=False,  # We did it ourselves
                )
                review_session.literature_context = literature_context
            else:
                # Use built-in PubMed search
                review_session = orchestrator.run_review(
                    manuscript_text=manuscript_text,
                    manuscript_path=manuscript_path,
                    search_literature=config.search_literature,
                    pubmed_email=pubmed_email,
                    max_literature_results=config.max_literature_results,
                )
            
            duration = time.time() - start_time
            cost = review_session.total_cost
            
            # Evaluate the review
            manuscript_summary = self.evaluator.generate_manuscript_summary(manuscript_text)
            evaluation = self.evaluator.evaluate(
                review_text=review_session.final_review,
                manuscript_summary=manuscript_summary,
            )
            
            return ABTestResult(
                config=config,
                review_session=review_session,
                evaluation=evaluation,
                duration_seconds=duration,
                cost_usd=cost,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ABTestResult(
                config=config,
                error=str(e),
                duration_seconds=duration,
            )
    
    def run_tests(
        self,
        configs: list[ABTestConfig],
        manuscript_path: str,
        pubmed_email: Optional[str] = None,
        verbose: bool = True,
    ) -> ABTestSession:
        """Run A/B tests with multiple configurations."""
        # Load manuscript
        if verbose:
            console.print(Panel(f"[bold]A/B Testing Review Configurations[/bold]\n\nManuscript: {manuscript_path}"))
        
        manuscript = get_manuscript_content(manuscript_path)
        manuscript_text = manuscript["text"]
        
        # Generate summary
        if verbose:
            console.print("[bold blue]Generating manuscript summary...[/bold blue]")
        manuscript_summary = self.evaluator.generate_manuscript_summary(manuscript_text)
        
        session = ABTestSession(
            manuscript_path=manuscript_path,
            manuscript_summary=manuscript_summary,
        )
        
        # Run each configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Testing...", total=len(configs))
            
            for config in configs:
                progress.update(task, description=f"[bold]{config.name}[/bold]...")
                
                if verbose:
                    console.print(f"\n[bold cyan]Testing: {config.name}[/bold cyan]")
                    console.print(f"[dim]{config.description}[/dim]")
                
                result = self.run_single_config(
                    config=config,
                    manuscript_text=manuscript_text,
                    manuscript_path=manuscript_path,
                    pubmed_email=pubmed_email,
                    verbose=verbose,
                )
                
                session.results.append(result)
                
                if verbose:
                    if result.error:
                        console.print(f"[red]  Error: {result.error}[/red]")
                    else:
                        console.print(f"[green]  Score: {result.evaluation.weighted_score:.1f}/10 ({result.evaluation.get_grade()})[/green]")
                        console.print(f"[dim]  Time: {result.duration_seconds:.1f}s, Cost: ${result.cost_usd:.4f}[/dim]")
                
                progress.advance(task)
        
        # Rank results
        rankings = []
        for result in session.results:
            if result.error or not result.evaluation:
                score = 0.0
            else:
                score = result.evaluation.weighted_score
            rankings.append((result.config.name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        session.rankings = rankings
        
        # Display results
        if verbose:
            console.print()
            console.print(session.format_summary_table())
            
            if rankings:
                best = rankings[0]
                console.print(f"\n[bold green]Best Configuration: {best[0]} (score: {best[1]:.1f}/10)[/bold green]")
        
        return session


def run_ab_test(
    manuscript_path: str,
    configs: list[ABTestConfig] | None = None,
    judge_model: str = "gpt-5-nano",
    pubmed_email: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> ABTestSession:
    """
    Convenience function to run A/B tests.
    
    Args:
        manuscript_path: Path to manuscript PDF
        configs: List of configurations to test (default: create_default_configs())
        judge_model: Model for LLM-as-judge
        pubmed_email: Email for PubMed API
        output_path: Optional path to save results
        verbose: Whether to print progress
        
    Returns:
        ABTestSession with results
    """
    if configs is None:
        configs = create_default_configs()
    
    runner = ABTestRunner(judge_model=judge_model)
    session = runner.run_tests(
        configs=configs,
        manuscript_path=manuscript_path,
        pubmed_email=pubmed_email,
        verbose=verbose,
    )
    
    if output_path:
        session.save(output_path)
        if verbose:
            console.print(f"\n[green]✓[/green] Results saved to: {output_path}")
    
    return session

