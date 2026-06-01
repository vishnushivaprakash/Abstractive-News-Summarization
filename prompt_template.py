"""
Prompt template for professional news summarization.
Supports Short, Medium, and Long summary targets with specific guidelines.
"""

PROFESSIONAL_SUMMARIZATION_PROMPT = """You are an abstractive news summarization model.

Based on the selected length type, generate the summary as follows:
- If SHORT → Summarize the content into approximately 1/3 of the original length.
- If MEDIUM → Summarize the content into approximately 2/3 of the original length.
- If LONG → Elaborate the content to double its original length (e.g., if input is 50 lines, output MUST be 100 lines) with additional context and detailed explanation while maintaining factual accuracy.

Rules:
1. Rewrite in your own words (no direct copying).
2. Maintain a formal news tone.
3. Keep important information and maintain factual accuracy.
4. Ensure coherence and clarity.
5. Focus only on the core message while adhering to the requested format.

Length Guidelines:
{length_guidelines}

Output Format:
Summary Type: {summary_type}

Article:
{input_article}

Summary:
"""

LENGTH_GUIDELINES = {
    "short": "- SHORT: Summarize to approximately 1/3 of original length",
    "medium": "- MEDIUM: Summarize to approximately 2/3 of original length",
    "long": "- LONG: Elaborate content to be EXACTLY DOUBLE the original length (e.g. 50 lines to 100 lines) with additional context"
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

