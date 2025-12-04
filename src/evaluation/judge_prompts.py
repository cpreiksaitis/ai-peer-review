"""LLM-as-judge prompts for evaluation."""

from .criteria import LITERATURE_CRITERIA, REVIEW_CRITERIA

# =============================================================================
# LITERATURE EVALUATION PROMPTS
# =============================================================================

LITERATURE_JUDGE_SYSTEM = """You are an expert academic librarian and research methodologist evaluating the quality of a literature search for peer review purposes.

Your task is to assess how well a set of retrieved papers would support a thorough peer review of a manuscript. You will evaluate the papers against specific criteria and provide scores and reasoning.

Be rigorous but fair. Consider what an expert peer reviewer would need to contextualize and evaluate the manuscript."""


def build_literature_judge_prompt(
    manuscript_summary: str,
    papers: list[dict],
    criteria: dict = None,
) -> str:
    """Build the literature evaluation prompt."""
    criteria = criteria or LITERATURE_CRITERIA
    
    # Format papers for evaluation
    papers_text = ""
    for i, paper in enumerate(papers, 1):
        papers_text += f"""
### Paper {i}
- **Title**: {paper.get('title', 'Unknown')}
- **Authors**: {', '.join(paper.get('authors', [])[:3])}{'...' if len(paper.get('authors', [])) > 3 else ''}
- **PMID**: {paper.get('pmid', 'N/A')}
- **DOI**: {paper.get('doi', 'N/A')}
- **Journal**: {paper.get('journal', 'N/A')}
- **Year**: {paper.get('pub_date', 'N/A')}
- **Abstract**: {paper.get('abstract', 'No abstract available')[:500]}{'...' if len(paper.get('abstract', '')) > 500 else ''}
"""
    
    # Build criteria descriptions
    criteria_text = ""
    for key, crit in criteria.items():
        criteria_text += f"""
### {crit['name']} ({key})
{crit['description']}

Scoring guide:
{crit['scoring_guide']}
"""
    
    return f"""## Manuscript Summary
{manuscript_summary}

## Retrieved Papers
{papers_text}

## Evaluation Criteria
{criteria_text}

## Instructions

Evaluate the retrieved papers against each criterion. For each criterion:
1. Assign a score from 1-10
2. Provide a brief justification (1-2 sentences)

Return your evaluation as a JSON object with this structure:
```json
{{
    "scores": {{
        "relevance": <score>,
        "methodological_match": <score>,
        "recency": <score>,
        "foundational": <score>,
        "diversity": <score>,
        "verifiability": <score>,
        "abstract_quality": <score>
    }},
    "reasoning": {{
        "relevance": "<justification>",
        "methodological_match": "<justification>",
        "recency": "<justification>",
        "foundational": "<justification>",
        "diversity": "<justification>",
        "verifiability": "<justification>",
        "abstract_quality": "<justification>"
    }},
    "overall_assessment": "<2-3 sentence overall assessment>",
    "missing_elements": ["<any important types of papers that are missing>"],
    "strongest_papers": [<indices of the 2-3 most useful papers, 1-indexed>]
}}
```

Return ONLY the JSON object, no additional text."""


# =============================================================================
# REVIEW EVALUATION PROMPTS
# =============================================================================

REVIEW_JUDGE_SYSTEM = """You are an expert journal editor evaluating the quality of a peer review.

Your task is to assess how well a review helps authors improve their manuscript and helps editors make publication decisions. You will evaluate the review against specific criteria and provide scores and reasoning.

Be rigorous but fair. Consider what constitutes an excellent peer review in academic publishing."""


def build_review_judge_prompt(
    manuscript_summary: str,
    review_text: str,
    criteria: dict = None,
) -> str:
    """Build the review evaluation prompt."""
    criteria = criteria or REVIEW_CRITERIA
    
    # Build criteria descriptions
    criteria_text = ""
    for key, crit in criteria.items():
        criteria_text += f"""
### {crit['name']} ({key})
{crit['description']}

Scoring guide:
{crit['scoring_guide']}
"""
    
    return f"""## Manuscript Summary
{manuscript_summary}

## Peer Review to Evaluate
{review_text}

## Evaluation Criteria
{criteria_text}

## Instructions

Evaluate the peer review against each criterion. For each criterion:
1. Assign a score from 1-10
2. Provide a brief justification (1-2 sentences)

Also identify specific strengths and areas for improvement in the review.

Return your evaluation as a JSON object with this structure:
```json
{{
    "scores": {{
        "comprehensiveness": <score>,
        "constructiveness": <score>,
        "specificity": <score>,
        "literature_integration": <score>,
        "balance": <score>,
        "accuracy": <score>,
        "clarity": <score>,
        "recommendation_justification": <score>
    }},
    "reasoning": {{
        "comprehensiveness": "<justification>",
        "constructiveness": "<justification>",
        "specificity": "<justification>",
        "literature_integration": "<justification>",
        "balance": "<justification>",
        "accuracy": "<justification>",
        "clarity": "<justification>",
        "recommendation_justification": "<justification>"
    }},
    "overall_assessment": "<2-3 sentence overall assessment>",
    "strengths": ["<specific strength 1>", "<specific strength 2>", ...],
    "improvements": ["<specific improvement suggestion 1>", "<specific improvement suggestion 2>", ...],
    "exemplary_comments": ["<quote or paraphrase of particularly good comments>"],
    "problematic_comments": ["<quote or paraphrase of comments that need improvement>"]
}}
```

Return ONLY the JSON object, no additional text."""


# =============================================================================
# MANUSCRIPT SUMMARY PROMPT
# =============================================================================

MANUSCRIPT_SUMMARY_PROMPT = """Provide a concise summary of this academic manuscript for evaluation purposes.

Include:
1. Research question/objective (1 sentence)
2. Study design and methods (1-2 sentences)
3. Key population/setting (1 sentence)
4. Main findings (1-2 sentences)
5. Field/domain (e.g., emergency medicine, medical education)

Keep the summary under 200 words.

Manuscript:
{manuscript_text}

Return ONLY the summary, no additional text."""

