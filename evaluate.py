from rouge_score import rouge_scorer
from summarizer import NewsSummarizer

def evaluate_model():
    print("Initializing Model...")
    summarizer = NewsSummarizer()
    
    # A sample long article serving as input
    reference_article = """
    A historic blizzard has brought much of the Northeast to a standstill, dumping up to 3 feet of snow in some areas. 
    State governments have declared states of emergency, closing schools, halting public transportation, and urging residents 
    to remain indoors. Power outages are affecting thousands of homes, particularly in coastal regions where high winds 
    have downed power lines and trees. Emergency responders are working around the clock to clear major highways and assist 
    stranded motorists. Meteorologists predict that the storm will continue through tomorrow morning, bringing an additional 
    6 to 12 inches of snow before moving offshore. The economic impact is estimated to be in the hundreds of millions as 
    businesses remain closed and supply chains are disrupted. Local officials are asking citizens to check on elderly neighbors 
    and avoid unnecessary travel until conditions improve.
    """
    
    # The gold standard summary (human written reference)
    human_summary = "A massive blizzard in the Northeast has caused states of emergency, widespread power outages, and travel bans. Emergency crews are working to clear roads and assist residents. The storm is expected to continue with severe economic impacts."
    
    print("\nGenerating Model Summary (Medium Length)...")
    model_summary = summarizer.summarize(reference_article, length_option="Medium")
    
    print(f"Generated Model Summary:\n{model_summary}\n")
    print(f"Reference Human Summary:\n{human_summary}\n")
    
    # Initialize ROUGE scorer
    # ROUGE-1: Unigram overlap
    # ROUGE-2: Bigram overlap
    # ROUGE-L: Longest Common Subsequence overlap
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(human_summary, model_summary)
    
    print("--- ROUGE Evaluation ---")
    for metric, score in scores.items():
        print(f"{metric} -> Precision: {score.precision:.2f}, Recall: {score.recall:.2f}, F1 Measure: {score.fmeasure:.2f}")

if __name__ == "__main__":
    evaluate_model()
