import torch
from transformers import pipeline
from prompt_template import format_summarization_prompt

class NewsSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the summarizer with the BART-Large-CNN pretrained model.

        WHY BART-Large-CNN?
        ────────────────────
        BART (Bidirectional and Auto-Regressive Transformer) was pre-trained by
        Facebook AI and fine-tuned on the CNN/DailyMail news dataset, which contains
        multi-sentence human-written summaries. This makes it ideal for generating:

            Short  → 1 sentence,   max 20 words  (~25 tokens)
            Medium → 2–3 sentences, ~50 words    (~65 tokens)
            Long   → 4–6 sentences, ~100 words   (~130 tokens)

        Unlike Pegasus-XSum (which forces a single extreme sentence), BART-CNN
        naturally produces multi-sentence, formally-toned, abstractive summaries
        that rephrase the source without copying it verbatim.
        """
        self.device = 0 if torch.cuda.is_available() else -1

        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=self.device,
            tokenizer=model_name
        )

        # Length configurations aligned to user-defined summary targets:
        #   Short  → 1 sentence,  max 20 words  (~25 tokens)
        #   Medium → 2–3 sentences, ~50 words   (~65 tokens)
        #   Long   → 4–6 sentences, ~100 words  (~130 tokens)
        self.length_configs = {
            "Short":  {"max_length": 30,  "min_length": 10},
            "Medium": {"max_length": 70,  "min_length": 35},
            "Long":   {"max_length": 140, "min_length": 80}
        }

    def summarize(self, text, length_option="Medium"):
        """
        Generates a multi-sentence abstractive summary using BART-Large-CNN.

        The model rewrites the core content of the article into a concise,
        formally-toned summary — never copying sentences directly.

        Summary length is controlled by the length_option parameter:
            - "Short"  → 1 sentence,   ~20 words
            - "Medium" → 2–3 sentences, ~50 words
            - "Long"   → 4–6 sentences, ~100 words
        """
        config = self.length_configs.get(length_option, self.length_configs["Medium"])

        summary = self.summarizer(
            text,
            max_length=config["max_length"],
            min_length=config["min_length"],
            do_sample=False,
            num_beams=6,           # 6 beams for high-quality abstractive output
            truncation=True,
            early_stopping=True    # Stop once all beams produce an EOS token
        )

        return summary[0]["summary_text"]


class LLMSummarizer:
    """
    LLM-based summarizer using prompt engineering.
    This class is designed to work with generative language models (GPT, Claude, etc.)
    that accept text prompts and generate summaries.
    
    Usage example with OpenAI:
        import openai
        summarizer = LLMSummarizer(api_call_fn=lambda prompt: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content)
    """
    
    def __init__(self, api_call_fn=None):
        """
        Initialize the LLM summarizer.
        
        Args:
            api_call_fn: A function that takes a prompt string and returns the generated text.
                        If None, you'll need to provide this when calling summarize().
        """
        self.api_call_fn = api_call_fn
    
    def summarize(self, text, api_call_fn=None, length_option="Medium"):
        """
        Generates an abstractive summary using an LLM with the professional prompt template.
        
        Args:
            text: The news article to summarize
            api_call_fn: Optional function to call the LLM API. If provided, overrides
                        the instance-level api_call_fn.
            length_option: Target length ('Short', 'Medium', or 'Long').
        
        Returns:
            Generated summary string
        """
        # Format the prompt using the template
        prompt = format_summarization_prompt(text, summary_type=length_option)
        
        # Use provided api_call_fn or instance-level one
        call_fn = api_call_fn or self.api_call_fn
        
        if call_fn is None:
            raise ValueError(
                "No API call function provided. Either set api_call_fn in __init__ "
                "or pass it as a parameter to summarize()."
            )
        
        # Call the LLM API
        summary = call_fn(prompt)
        
        # Clean up the response (remove the prompt or prefix if it's included)
        summary = summary.strip()
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        
        return summary
    
    def get_prompt(self, text, length_option="Medium"):
        """
        Get the formatted prompt for the given article text.
        Useful for debugging or manual API calls.
        
        Args:
            text: The news article to summarize
            length_option: Target length ('Short', 'Medium', or 'Long')
            
        Returns:
            Formatted prompt string
        """
        return format_summarization_prompt(text, summary_type=length_option)
