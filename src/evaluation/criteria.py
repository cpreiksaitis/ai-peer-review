"""Evaluation criteria for literature search and review quality."""

# Literature Search Evaluation Criteria
LITERATURE_CRITERIA = {
    "relevance": {
        "name": "Topic Relevance",
        "description": "Papers directly address the manuscript's topic, research question, or clinical focus",
        "weight": 0.20,
        "scoring_guide": """
            10: All papers are directly on-topic, addressing the exact research question
            7-9: Most papers are highly relevant, 1-2 may be tangentially related
            4-6: Mix of relevant and tangential papers
            1-3: Most papers are off-topic or only loosely related
        """,
    },
    "methodological_match": {
        "name": "Methodological Similarity",
        "description": "Papers use similar study designs, methods, or analytical approaches",
        "weight": 0.20,
        "scoring_guide": """
            10: Papers use identical or very similar study designs (e.g., RCT vs RCT)
            7-9: Papers use comparable methodologies, allowing meaningful comparison
            4-6: Some methodological overlap, but significant differences
            1-3: Methodologies are incompatible for comparison
        """,
    },
    "recency": {
        "name": "Recency",
        "description": "Includes recent papers from the last 3 years reflecting current knowledge",
        "weight": 0.15,
        "scoring_guide": """
            10: Majority of papers from last 2 years, all within 5 years
            7-9: Good mix of recent papers, most within 5 years
            4-6: Some recent papers but many older
            1-3: Mostly outdated papers (>5 years old)
        """,
    },
    "foundational": {
        "name": "Foundational Coverage",
        "description": "Includes key landmark or highly-cited papers that established the field",
        "weight": 0.15,
        "scoring_guide": """
            10: Includes seminal papers and key references any expert would cite
            7-9: Includes most important foundational work
            4-6: Some foundational papers but missing key references
            1-3: Missing major foundational work in the field
        """,
    },
    "diversity": {
        "name": "Perspective Diversity",
        "description": "Mix of supporting, contradictory, and complementary perspectives",
        "weight": 0.10,
        "scoring_guide": """
            10: Excellent balance of supporting and challenging perspectives
            7-9: Good diversity, includes some contrasting viewpoints
            4-6: Mostly one-sided, limited diversity
            1-3: Echo chamber effect, all papers support same view
        """,
    },
    "verifiability": {
        "name": "Verifiability",
        "description": "Papers have valid PMIDs, DOIs, or URLs that can be verified",
        "weight": 0.10,
        "scoring_guide": """
            10: All papers have valid, verifiable identifiers (PMID/DOI)
            7-9: Most papers verifiable, 1-2 may lack identifiers
            4-6: About half have verifiable identifiers
            1-3: Most papers cannot be verified or don't exist
        """,
    },
    "abstract_quality": {
        "name": "Abstract Quality",
        "description": "Abstracts are substantive, informative, and useful for review",
        "weight": 0.10,
        "scoring_guide": """
            10: All abstracts are complete and provide key methods/results
            7-9: Most abstracts are informative and useful
            4-6: Abstracts present but often incomplete or vague
            1-3: Missing abstracts or abstracts provide little value
        """,
    },
}

# Review Quality Evaluation Criteria
REVIEW_CRITERIA = {
    "comprehensiveness": {
        "name": "Comprehensiveness",
        "description": "Review adequately covers methodology, results, and discussion sections",
        "weight": 0.15,
        "scoring_guide": """
            10: Thoroughly addresses all major sections with detailed analysis
            7-9: Covers most sections well, minor gaps
            4-6: Addresses some sections but significant gaps
            1-3: Superficial or misses major sections entirely
        """,
    },
    "constructiveness": {
        "name": "Constructiveness",
        "description": "Feedback is actionable and oriented toward improvement",
        "weight": 0.15,
        "scoring_guide": """
            10: All critiques include specific, actionable suggestions
            7-9: Most feedback is constructive with clear paths to improvement
            4-6: Mix of constructive and vague/unhelpful feedback
            1-3: Mostly negative without actionable guidance
        """,
    },
    "specificity": {
        "name": "Specificity",
        "description": "Comments reference specific sections, tables, figures, or data",
        "weight": 0.15,
        "scoring_guide": """
            10: All major comments cite specific locations and data
            7-9: Most comments are specific with clear references
            4-6: Some specific comments but many are general
            1-3: Vague, general comments without specific references
        """,
    },
    "literature_integration": {
        "name": "Literature Integration",
        "description": "Appropriately references and compares to related work",
        "weight": 0.10,
        "scoring_guide": """
            10: Excellent integration of relevant literature throughout
            7-9: Good use of literature to contextualize findings
            4-6: Some literature mentioned but not well integrated
            1-3: Little to no reference to existing literature
        """,
    },
    "balance": {
        "name": "Balance",
        "description": "Acknowledges strengths alongside weaknesses",
        "weight": 0.10,
        "scoring_guide": """
            10: Fair assessment highlighting both strengths and weaknesses
            7-9: Good balance with clear acknowledgment of merits
            4-6: Somewhat unbalanced, either too positive or negative
            1-3: One-sided, either purely critical or purely positive
        """,
    },
    "accuracy": {
        "name": "Technical Accuracy",
        "description": "Methodological and technical critiques are correct",
        "weight": 0.15,
        "scoring_guide": """
            10: All technical critiques are accurate and well-founded
            7-9: Most critiques are technically sound
            4-6: Some accurate points but also some errors
            1-3: Contains significant technical errors or misconceptions
        """,
    },
    "clarity": {
        "name": "Clarity",
        "description": "Review is well-organized and clearly written",
        "weight": 0.10,
        "scoring_guide": """
            10: Exceptionally clear, logical structure, easy to follow
            7-9: Well-organized and readable
            4-6: Somewhat disorganized or unclear in places
            1-3: Poorly organized, difficult to understand
        """,
    },
    "recommendation_justification": {
        "name": "Recommendation Justification",
        "description": "Final decision is well-supported by the review comments",
        "weight": 0.10,
        "scoring_guide": """
            10: Decision clearly follows from evidence presented
            7-9: Decision is well-justified with minor gaps
            4-6: Decision partially supported but some disconnects
            1-3: Decision seems arbitrary or contradicts review content
        """,
    },
}


def get_criteria_weights(criteria_dict: dict) -> dict[str, float]:
    """Extract just the weights from a criteria dictionary."""
    return {key: val["weight"] for key, val in criteria_dict.items()}


def calculate_weighted_score(scores: dict[str, float], criteria_dict: dict) -> float:
    """Calculate weighted average score from individual scores."""
    total_weight = sum(c["weight"] for c in criteria_dict.values())
    weighted_sum = sum(
        scores.get(key, 0) * criteria_dict[key]["weight"]
        for key in criteria_dict
    )
    return weighted_sum / total_weight if total_weight > 0 else 0

