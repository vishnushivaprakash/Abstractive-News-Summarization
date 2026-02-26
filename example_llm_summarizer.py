"""
Example usage of the professional LLM-based summarizer with multi-length support.

This example shows how the prompt template adapts to different length targets.
"""

from prompt_template import format_summarization_prompt, PROFESSIONAL_SUMMARIZATION_PROMPT
from summarizer import LLMSummarizer

# Example 1: View the base prompt template
print("=" * 60)
print("BASE PROMPT TEMPLATE (Before formatting):")
print("=" * 60)
print(PROFESSIONAL_SUMMARIZATION_PROMPT)
print()

# Sample article for demonstration
sample_article = """
A historic blizzard has brought much of the Northeast to a standstill, dumping up to 3 feet of snow in some areas. 
State governments have declared states of emergency, closing schools, halting public transportation, and urging residents 
to remain indoors. Power outages are affecting thousands of homes, particularly in coastal regions where high winds 
have downed power lines and trees. Emergency responders are working around the clock to clear major highways and assist 
stranded motorists. Meteorologists predict that the storm will continue through tomorrow morning, bringing an additional 
6 to 12 inches of snow before moving offshore.
"""

# Example 2: Show prompts for different lengths
for length in ["Short", "Medium", "Long"]:
    print("=" * 60)
    print(f"FORMATTED PROMPT ({length.upper()}):")
    print("=" * 60)
    formatted_prompt = format_summarization_prompt(sample_article, summary_type=length)
    # Print first few lines of the prompt to see the length guideline
    lines = formatted_prompt.split('\n')
    print('\n'.join(lines[:22])) # Show header and instructions
    print("...")
    print(lines[-2]) # Show Summary: line
    print()

# Example 3: Using LLMSummarizer with a mock function
def mock_llm_call(prompt):
    """Mock LLM that returns a simple summary for demonstration."""
    if "SHORT" in prompt:
        return "A historic blizzard has paralyzed the Northeast, leading to states of emergency and power outages."
    elif "MEDIUM" in prompt:
        return "A massive blizzard in the Northeast has caused states of emergency, power outages, and travel halts. Emergency responders are working to clear roads as the storm is expected to continue until tomorrow."
    else:
        return "A historic winter storm has brought the Northeast to a standstill, dumping up to three feet of snow and causing widespread emergencies. Thousands are without power as high winds topple lines, while emergency crews work desperately to clear highways. Meteorologists warn of further snowfall before the storm moves offshore tomorrow."

print("=" * 60)
print("DEMONSTRATING LLMSummarizer WITH DIFFERENT LENGTHS:")
print("=" * 60)
summarizer = LLMSummarizer(api_call_fn=mock_llm_call)

for length in ["Short", "Medium", "Long"]:
    summary = summarizer.summarize(sample_article, length_option=length)
    print(f"\n[{length.upper()} SUMMARY]:")
    print(summary)
print()

