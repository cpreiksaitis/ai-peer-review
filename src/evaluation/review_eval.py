"""Review quality evaluation using LLM-as-judge."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import litellm

from .criteria import REVIEW_CRITERIA, calculate_weighted_score
from .judge_prompts import (
    REVIEW_JUDGE_SYSTEM,
    MANUSCRIPT_SUMMARY_PROMPT,
    build_review_judge_prompt,
)

# Allow dropping unsupported params
litellm.drop_params = True


@dataclass
class ReviewEvalResult:
    """Result of evaluating a review."""
    
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    overall_assessment: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    exemplary_comments: list[str] = field(default_factory=list)
    problematic_comments: list[str] = field(default_factory=list)
    weighted_score: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scores": self.scores,
            "reasoning": self.reasoning,
            "overall_assessment": self.overall_assessment,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "exemplary_comments": self.exemplary_comments,
            "problematic_comments": self.problematic_comments,
            "weighted_score": self.weighted_score,
            "error": self.error,
        }
    
    def format_report(self) -> str:
        """Format as human-readable report."""
        if self.error:
            return f"## Review Evaluation\n\n**Error:** {self.error}"
        
        lines = [
            "# Review Quality Evaluation",
            "",
            f"**Overall Score:** {self.weighted_score:.1f}/10",
            "",
            "## Criterion Scores",
        ]
        
        for key, score in self.scores.items():
            crit = REVIEW_CRITERIA.get(key, {})
            name = crit.get("name", key)
            reason = self.reasoning.get(key, "")
            lines.append(f"- **{name}:** {score}/10")
            if reason:
                lines.append(f"  - {reason}")
        
        lines.extend([
            "",
            "## Overall Assessment",
            self.overall_assessment,
        ])
        
        if self.strengths:
            lines.extend([
                "",
                "## Strengths",
                *[f"- {s}" for s in self.strengths],
            ])
        
        if self.improvements:
            lines.extend([
                "",
                "## Suggested Improvements",
                *[f"- {i}" for i in self.improvements],
            ])
        
        if self.exemplary_comments:
            lines.extend([
                "",
                "## Exemplary Comments",
                *[f"> {c}" for c in self.exemplary_comments],
            ])
        
        if self.problematic_comments:
            lines.extend([
                "",
                "## Comments Needing Improvement",
                *[f"> {c}" for c in self.problematic_comments],
            ])
        
        return "\n".join(lines)
    
    def get_grade(self) -> str:
        """Get letter grade based on score."""
        score = self.weighted_score
        if score >= 9:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6:
            return "C+"
        elif score >= 5.5:
            return "C"
        elif score >= 5:
            return "C-"
        elif score >= 4:
            return "D"
        else:
            return "F"


class ReviewEvaluator:
    """Evaluates peer review quality using LLM-as-judge."""
    
    def __init__(self, model: str = "gpt-5-nano"):
        self.model = model
    
    def generate_manuscript_summary(self, manuscript_text: str) -> str:
        """Generate a concise summary of the manuscript for evaluation."""
        prompt = MANUSCRIPT_SUMMARY_PROMPT.format(
            manuscript_text=manuscript_text[:10000]
        )
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        return response.choices[0].message.content
    
    def evaluate(
        self,
        review_text: str,
        manuscript_summary: str,
    ) -> ReviewEvalResult:
        """
        Evaluate a peer review.
        
        Args:
            review_text: The review to evaluate
            manuscript_summary: Summary of the manuscript being reviewed
            
        Returns:
            ReviewEvalResult with scores and feedback
        """
        if not review_text or len(review_text.strip()) < 100:
            return ReviewEvalResult(
                error="Review text is too short to evaluate",
            )
        
        # Build evaluation prompt
        prompt = build_review_judge_prompt(
            manuscript_summary=manuscript_summary,
            review_text=review_text,
            criteria=REVIEW_CRITERIA,
        )
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": REVIEW_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return ReviewEvalResult(
                    error=f"Could not parse JSON response: {content[:200]}",
                )
            
            data = json.loads(json_match.group())
            
            scores = data.get("scores", {})
            weighted = calculate_weighted_score(scores, REVIEW_CRITERIA)
            
            return ReviewEvalResult(
                scores=scores,
                reasoning=data.get("reasoning", {}),
                overall_assessment=data.get("overall_assessment", ""),
                strengths=data.get("strengths", []),
                improvements=data.get("improvements", []),
                exemplary_comments=data.get("exemplary_comments", []),
                problematic_comments=data.get("problematic_comments", []),
                weighted_score=weighted,
            )
            
        except json.JSONDecodeError as e:
            return ReviewEvalResult(
                error=f"JSON parsing error: {e}",
            )
        except Exception as e:
            return ReviewEvalResult(
                error=f"Evaluation error: {e}",
            )
    
    def compare_reviews(
        self,
        reviews: dict[str, str],
        manuscript_summary: str,
    ) -> dict[str, ReviewEvalResult]:
        """
        Evaluate and compare multiple reviews.
        
        Args:
            reviews: Dict mapping review name/config to review text
            manuscript_summary: Summary of the manuscript
            
        Returns:
            Dict mapping review name to evaluation result
        """
        evaluations = {}
        
        for name, review_text in reviews.items():
            evaluations[name] = self.evaluate(
                review_text=review_text,
                manuscript_summary=manuscript_summary,
            )
        
        return evaluations
    
    def rank_reviews(
        self,
        evaluations: dict[str, ReviewEvalResult],
    ) -> list[tuple[str, float]]:
        """
        Rank reviews by weighted score.
        
        Args:
            evaluations: Dict of evaluation results
            
        Returns:
            List of (review_name, score) tuples, sorted by score descending
        """
        rankings = []
        
        for name, result in evaluations.items():
            if result.error:
                score = 0.0
            else:
                score = result.weighted_score
            rankings.append((name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

