"""
Prompt template for professional news summarization.
Supports Short, Medium, and Long summary targets with specific guidelines.
"""

PROFESSIONAL_SUMMARIZATION_PROMPT = """You are a professional news summarization model.

Your task is to generate a high-quality abstractive summary of the given news article.

Instructions:
1. Do NOT copy sentences directly from the article.
2. Rewrite in your own words (fully abstractive).
3. Focus only on the most important information.
4. Remove unnecessary details, repetition, and minor facts.
5. Maintain factual accuracy.
6. Keep tone neutral and professional.
7. Follow the requested summary length strictly.

Length Guidelines:
{length_guidelines}

Output Format:
Summary Type: {summary_type}

Article:
{input_article}

Summary:
"""

LENGTH_GUIDELINES = {
    "short": "- SHORT: 1 sentence (15–25 words)",
    "medium": "- MEDIUM: 2–3 sentences (40–70 words)",
    "long": "- LONG: 1 well-structured paragraph (100–150 words)"
}

def format_summarization_prompt(article_text: str, summary_type: str = "medium") -> str:
    """
    Formats the professional summarization prompt template.
    
    Args:
        article_text: The news article to summarize
        summary_type: Target length ('short', 'medium', or 'long')
        
    Returns:
        Formatted prompt string ready for LLM input
    """
    summary_type = summary_type.lower()
    if summary_type not in LENGTH_GUIDELINES:
        summary_type = "medium"
        
    guidelines = LENGTH_GUIDELINES[summary_type]
    
    return PROFESSIONAL_SUMMARIZATION_PROMPT.format(
        length_guidelines=guidelines,
        summary_type=summary_type.upper(),
        input_article=article_text
    )

