"""Orchestrator agent that manages the multi-agent review and debate process."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import litellm
from litellm.utils import supports_pdf_input
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Allow LiteLLM to drop unsupported parameters for different models
litellm.drop_params = True

from src.agents import (
    BaseReviewerAgent,
    MethodologistAgent,
    DomainExpertAgent,
    CommunicationAgent,
    EthicsAgent,
)
from src.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_SYNTHESIS_PROMPT,
    LITERATURE_QUERY_PROMPT,
)
from src.literature import (
    search_related_literature,
    format_literature_context,
    filter_and_rank_papers,
    RelatedPaper,
)
from src.costs import CostTracker, reset_global_tracker
from src.document import create_pdf_message_content

console = Console()


@dataclass
class ReviewSession:
    """Stores all data from a review session."""

    manuscript_path: str
    manuscript_text: str
    literature_context: str
    initial_reviews: dict[str, str] = field(default_factory=dict)
    debate_rounds: list[dict[str, str]] = field(default_factory=list)
    final_positions: dict[str, str] = field(default_factory=dict)
    final_review: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cost_tracker: Optional[CostTracker] = None
    pdf_base64: Optional[str] = None  # Base64-encoded PDF for vision models
    use_pdf_vision: bool = False  # Whether PDF vision was used

    @property
    def cost_summary(self) -> dict:
        """Get cost summary."""
        if self.cost_tracker:
            return self.cost_tracker.get_summary()
        return {"total_cost_usd": 0, "total_tokens": 0, "num_calls": 0}

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self.cost_summary.get("total_cost_usd", 0)


# Type aliases for callbacks
StatusCallback = Optional[callable]
OutputCallback = Optional[callable]  # (agent_name: str, output_type: str, content: str) -> None


@dataclass
class Orchestrator:
    """
    Orchestrator agent that coordinates the multi-agent review process.

    Manages:
    - Literature search query generation
    - Initial review collection from all agents
    - Debate rounds between agents
    - Final consensus synthesis
    """

    model: str = "claude-haiku-4-5"
    temperature: float = 0.7
    debate_rounds: int = 2
    agents: list[BaseReviewerAgent] = field(default_factory=list)
    cost_tracker: Optional[CostTracker] = None
    use_pdf_vision: bool = True  # Use native PDF input when available
    status_callback: StatusCallback = None  # Callback for status updates
    output_callback: OutputCallback = None  # Callback for live output

    def __post_init__(self):
        """Initialize the reviewer agents if not provided."""
        if not self.agents:
            self.agents = self._create_default_agents()
    
    def _update_status(self, status: str) -> None:
        """Update status via callback if available."""
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception:
                pass  # Don't let callback errors break the review
    
    def _emit_output(self, agent_name: str, output_type: str, content: str) -> None:
        """Emit output via callback if available."""
        if self.output_callback:
            try:
                self.output_callback(agent_name, output_type, content)
            except Exception:
                pass  # Don't let callback errors break the review

    def _create_default_agents(self) -> list[BaseReviewerAgent]:
        """Create the default set of reviewer agents."""
        return [
            MethodologistAgent(cost_tracker=self.cost_tracker),
            DomainExpertAgent(cost_tracker=self.cost_tracker),
            CommunicationAgent(cost_tracker=self.cost_tracker),
            EthicsAgent(cost_tracker=self.cost_tracker),
        ]

    def _call_llm(self, messages: list[dict[str, str]], operation: str = "orchestrator") -> str:
        """Make a call to the orchestrator's LLM with cost tracking and retry logic."""
        from src.agents.base import retry_with_backoff
        
        # Some models (like gpt-5-nano) don't support custom temperature
        model_lower = self.model.lower()
        supports_temperature = not any(x in model_lower for x in ["nano", "o1", "o3"])
        
        def make_call():
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if supports_temperature:
                kwargs["temperature"] = self.temperature
            return litellm.completion(**kwargs)
        
        # Use retry with backoff for rate limits
        response = retry_with_backoff(make_call)

        # Track cost
        if self.cost_tracker:
            self.cost_tracker.add_entry(
                model=self.model,
                agent_name="Orchestrator",
                operation=operation,
                response=response,
            )

        return response.choices[0].message.content

    def generate_literature_queries(self, manuscript_text: str, pdf_base64: str | None = None) -> list[str]:
        """
        Generate search queries for finding related literature.

        Args:
            manuscript_text: The manuscript text to analyze
            pdf_base64: Optional base64-encoded PDF for vision analysis

        Returns:
            List of search query strings
        """
        prompt_text = f"{LITERATURE_QUERY_PROMPT}\n\n## Manuscript\n{manuscript_text[:8000]}"
        
        # Use PDF vision if available and model supports it
        if pdf_base64 and self.use_pdf_vision:
            import litellm
            clean_pdf = pdf_base64.split(",", 1)[-1]
            if litellm.supports_pdf_input(model=self.model):
                messages = [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "file",
                                "file": {"file_data": f"data:application/pdf;base64,{clean_pdf}"}
                            }
                        ],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ]
        else:
            messages = [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

        response = self._call_llm(messages, operation="literature_queries")

        # Parse the JSON array from the response
        import json
        import re

        # Try to extract JSON array from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    return [q for q in queries if isinstance(q, str)]
            except json.JSONDecodeError:
                pass

        # Fallback: split by newlines and clean up
        lines = response.strip().split("\n")
        queries = []
        for line in lines:
            line = line.strip().strip("-").strip("•").strip("*").strip('"').strip()
            if line and len(line) > 10:
                queries.append(line)

        return queries[:5]  # Return at most 5 queries

    def search_literature(
        self,
        manuscript_text: str,
        max_results: int = 10,
        email: str | None = None,
        score_relevance: bool = True,
        pdf_base64: str | None = None,
    ) -> tuple[list[RelatedPaper], str]:
        """
        Search for related literature based on manuscript content.

        Args:
            manuscript_text: The manuscript text
            max_results: Maximum papers to retrieve
            email: Email for PubMed API
            score_relevance: Whether to score and filter by relevance
            pdf_base64: Optional base64-encoded PDF for vision-based query generation

        Returns:
            Tuple of (list of papers, formatted context string)
        """
        console.print("[bold blue]Generating literature search queries...[/bold blue]")
        queries = self.generate_literature_queries(manuscript_text, pdf_base64=pdf_base64)

        console.print(f"[dim]Generated {len(queries)} queries[/dim]")
        for q in queries:
            console.print(f"  [dim]• {q}[/dim]")

        all_papers: dict[str, RelatedPaper] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching PubMed...", total=len(queries))

            for query in queries:
                try:
                    # Search more papers initially if we're going to filter
                    search_count = (max_results * 3) // len(queries) + 2 if score_relevance else max_results // len(queries) + 1
                    papers = search_related_literature(
                        query=query,
                        max_results=search_count,
                        email=email,
                    )
                    for paper in papers:
                        if paper.pmid and paper.pmid not in all_papers:
                            all_papers[paper.pmid] = paper
                except Exception as e:
                    console.print(f"[yellow]Warning: Search failed for '{query}': {e}[/yellow]")

                progress.advance(task)

        papers_list = list(all_papers.values())
        console.print(f"[dim]Found {len(papers_list)} candidate papers[/dim]")

        # Score and filter by relevance
        if score_relevance and papers_list:
            console.print("[bold blue]Scoring paper relevance...[/bold blue]")
            # Create a brief manuscript summary for scoring
            manuscript_summary = manuscript_text[:2000]

            papers_list = filter_and_rank_papers(
                papers=papers_list,
                manuscript_summary=manuscript_summary,
                top_n=max_results,
                min_score=4.0,  # Only include papers with score >= 4
                model="gpt-5-nano",  # Use cheap model for scoring
            )
            console.print(f"[green]Selected {len(papers_list)} most relevant papers (score >= 4.0)[/green]")

        context = format_literature_context(papers_list, include_scores=score_relevance)

        return papers_list, context

    def collect_initial_reviews(
        self,
        manuscript_text: str,
        literature_context: str,
        pdf_base64: str | None = None,
    ) -> dict[str, str]:
        """
        Collect initial reviews from all agents.

        Args:
            manuscript_text: The manuscript text
            literature_context: Formatted literature context
            pdf_base64: Base64-encoded PDF for vision models (optional)

        Returns:
            Dictionary mapping agent names to their reviews
        """
        reviews = {}
        total_agents = len(self.agents)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting initial reviews...", total=total_agents)

            for i, agent in enumerate(self.agents, 1):
                progress.update(task, description=f"[bold]{agent.name}[/bold] reviewing...")
                self._update_status(f"Step 2/5: {agent.name} reviewing ({i}/{total_agents})...")

                try:
                    review = agent.generate_initial_review(
                        manuscript_text=manuscript_text,
                        literature_context=literature_context,
                        pdf_base64=pdf_base64,
                    )
                    reviews[agent.name] = review
                    self._emit_output(agent.name, "initial_review", review)
                    console.print(f"  [green]✓[/green] {agent.name} completed review")
                except Exception as e:
                    console.print(f"  [red]✗[/red] {agent.name} failed: {e}")
                    reviews[agent.name] = f"Error generating review: {e}"

                progress.advance(task)

        return reviews

    def run_debate_round(self, current_reviews: dict[str, str], round_num: int = 1) -> dict[str, str]:
        """
        Run a single debate round where agents respond to each other.

        Args:
            current_reviews: Current state of all reviews/positions
            round_num: Current debate round number

        Returns:
            Dictionary mapping agent names to their debate responses
        """
        responses = {}

        for agent in self.agents:
            # Get other agents' reviews (excluding this agent)
            other_reviews = {
                name: review
                for name, review in current_reviews.items()
                if name != agent.name
            }

            try:
                response = agent.participate_in_debate(other_reviews)
                responses[agent.name] = response
                self._emit_output(agent.name, f"debate_round_{round_num}", response)
            except Exception as e:
                console.print(f"  [red]✗[/red] {agent.name} debate failed: {e}")
                responses[agent.name] = f"Error in debate: {e}"

        return responses

    def collect_final_positions(self) -> dict[str, str]:
        """
        Collect final positions from all agents after debate.

        Returns:
            Dictionary mapping agent names to their final positions
        """
        positions = {}

        for agent in self.agents:
            try:
                position = agent.get_final_position()
                positions[agent.name] = position
                self._emit_output(agent.name, "final_position", position)
            except Exception as e:
                console.print(f"  [red]✗[/red] {agent.name} final position failed: {e}")
                positions[agent.name] = f"Error: {e}"

        return positions

    def synthesize_final_review(
        self,
        manuscript_text: str,
        initial_reviews: dict[str, str],
        debate_rounds: list[dict[str, str]],
        final_positions: dict[str, str],
    ) -> str:
        """
        Synthesize the final review from all agent inputs.

        Args:
            manuscript_text: The original manuscript
            initial_reviews: Initial reviews from all agents
            debate_rounds: List of debate round responses
            final_positions: Final positions from all agents

        Returns:
            The synthesized final review
        """
        # Build comprehensive context for synthesis
        context_parts = ["## Initial Reviews\n"]

        for name, review in initial_reviews.items():
            context_parts.append(f"### {name}\n{review}\n")

        if debate_rounds:
            context_parts.append("\n## Debate Summary\n")
            for i, round_responses in enumerate(debate_rounds, 1):
                context_parts.append(f"### Round {i}\n")
                for name, response in round_responses.items():
                    context_parts.append(f"**{name}:** {response[:500]}...\n")

        context_parts.append("\n## Final Positions\n")
        for name, position in final_positions.items():
            context_parts.append(f"### {name}\n{position}\n")

        full_context = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""{full_context}

{ORCHESTRATOR_SYNTHESIS_PROMPT}""",
            },
        ]

        return self._call_llm(messages, operation="synthesis")

    def run_review(
        self,
        manuscript_text: str,
        manuscript_path: str = "",
        search_literature: bool = True,
        pubmed_email: str | None = None,
        max_literature_results: int = 10,
        pdf_base64: str | None = None,
    ) -> ReviewSession:
        """
        Run the complete multi-agent review process.

        Args:
            manuscript_text: The manuscript text to review
            manuscript_path: Path to the manuscript file
            search_literature: Whether to search for related literature
            pubmed_email: Email for PubMed API
            max_literature_results: Maximum related papers to retrieve
            pdf_base64: Base64-encoded PDF for vision models (optional)

        Returns:
            ReviewSession containing all review data
        """
        console.print(Panel("[bold]Starting Multi-Agent Peer Review[/bold]", style="blue"))

        # Initialize cost tracker for this session
        self.cost_tracker = reset_global_tracker()

        # Update agents with the cost tracker
        for agent in self.agents:
            agent.cost_tracker = self.cost_tracker

        # Determine if we can use PDF vision
        clean_pdf = None
        if pdf_base64:
            clean_pdf = pdf_base64.split(",", 1)[-1]
        use_vision = (
            self.use_pdf_vision 
            and clean_pdf is not None 
            and supports_pdf_input(model=self.model, custom_llm_provider=None)
        )
        
        if use_vision:
            console.print("[dim]Using native PDF vision for document analysis[/dim]")

        session = ReviewSession(
            manuscript_path=manuscript_path,
            manuscript_text=manuscript_text,
            literature_context="",
            cost_tracker=self.cost_tracker,
            pdf_base64=clean_pdf if use_vision else None,
            use_pdf_vision=use_vision,
        )

        # Step 1: Literature search
        if search_literature:
            self._update_status("Step 1/5: Searching literature...")
            console.print("\n[bold]Step 1: Literature Search[/bold]")
            try:
                _, literature_context = self.search_literature(
                    manuscript_text=manuscript_text,
                    max_results=max_literature_results,
                    email=pubmed_email,
                    pdf_base64=clean_pdf,  # Use PDF vision for query generation
                )
                session.literature_context = literature_context
            except Exception as e:
                console.print(f"[yellow]Literature search skipped: {e}[/yellow]")
                session.literature_context = ""
        else:
            self._update_status("Step 1/5: Literature search skipped")
            console.print("\n[dim]Step 1: Literature search skipped[/dim]")

        # Step 2: Initial reviews
        self._update_status("Step 2/5: Collecting initial reviews...")
        console.print("\n[bold]Step 2: Initial Reviews[/bold]")
        session.initial_reviews = self.collect_initial_reviews(
            manuscript_text=manuscript_text,
            literature_context=session.literature_context,
            pdf_base64=session.pdf_base64,
        )

        # Display initial reviews
        for name, review in session.initial_reviews.items():
            console.print(Panel(review[:500] + "...", title=f"{name} - Initial Review", border_style="dim"))

        # Step 3: Debate rounds
        self._update_status(f"Step 3/5: Debate round 1/{self.debate_rounds}...")
        console.print(f"\n[bold]Step 3: Debate ({self.debate_rounds} rounds)[/bold]")
        current_state = session.initial_reviews.copy()

        for round_num in range(1, self.debate_rounds + 1):
            self._update_status(f"Step 3/5: Debate round {round_num}/{self.debate_rounds}...")
            console.print(f"\n[bold blue]Debate Round {round_num}[/bold blue]")
            round_responses = self.run_debate_round(current_state, round_num=round_num)
            session.debate_rounds.append(round_responses)

            # Update current state with debate responses
            for name, response in round_responses.items():
                current_state[name] = response

            console.print(f"  [green]✓[/green] Round {round_num} complete")

        # Step 4: Final positions
        self._update_status("Step 4/5: Collecting final positions...")
        console.print("\n[bold]Step 4: Final Positions[/bold]")
        session.final_positions = self.collect_final_positions()

        # Step 5: Synthesize final review
        self._update_status("Step 5/5: Synthesizing final review...")
        console.print("\n[bold]Step 5: Synthesizing Final Review[/bold]")
        session.final_review = self.synthesize_final_review(
            manuscript_text=manuscript_text,
            initial_reviews=session.initial_reviews,
            debate_rounds=session.debate_rounds,
            final_positions=session.final_positions,
        )
        self._emit_output("Orchestrator", "final_synthesis", session.final_review)

        console.print(Panel(session.final_review, title="[bold green]Final Review[/bold green]", border_style="green"))

        # Display cost summary
        if self.cost_tracker:
            cost_summary = self.cost_tracker.format_summary()
            console.print(Panel(cost_summary, title="[bold yellow]Cost Summary[/bold yellow]", border_style="yellow"))

        return session


def create_orchestrator_from_config(config: dict[str, Any]) -> Orchestrator:
    """
    Create an orchestrator from a configuration dictionary.

    Args:
        config: Configuration dictionary (from config.yaml)

    Returns:
        Configured Orchestrator instance
    """
    models = config.get("models", {})
    debate_config = config.get("debate", {})

    agents = [
        MethodologistAgent(model=models.get("methodologist", "gpt-5-nano")),
        DomainExpertAgent(model=models.get("domain_expert", "gemini/gemini-flash-lite-latest")),
        CommunicationAgent(model=models.get("communication", "gemini/gemini-flash-lite-latest")),
        EthicsAgent(model=models.get("ethics", "gpt-5-nano")),
    ]

    return Orchestrator(
        model=models.get("orchestrator", "claude-haiku-4-5"),
        temperature=debate_config.get("temperature", 0.7),
        debate_rounds=debate_config.get("rounds", 2),
        agents=agents,
    )
