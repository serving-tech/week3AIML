# Task 3: NLP with spaCy – NER and Sentiment Analysis

# Step 1: Install spaCy and download the English model
!pip install -U spacy
!python -m spacy download en_core_web_sm

# Step 2: Import libraries
import spacy

# Load small English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon-style product reviews
reviews = [
    "I absolutely love the Apple AirPods. The sound is crystal clear and the battery lasts all day!",
    "This Samsung Galaxy phone is terrible. It lags a lot and the screen cracked in a week.",
    "The JBL speaker is amazing! Great bass and compact size.",
    "I had high hopes for the Lenovo laptop, but it overheats and the fan is loud.",
    "Google Nest is perfect for my home. Easy to set up and works flawlessly."
]

# Step 3: Define sentiment keywords
positive_keywords = {"love", "great", "amazing", "perfect", "flawlessly", "easy", "clear"}
negative_keywords = {"terrible", "lags", "cracked", "overheats", "loud", "bad", "slow"}

# Step 4: Process and analyze each review
for i, review in enumerate(reviews):
    print(f"\n🔍 Review {i+1}: {review}")
    doc = nlp(review)
    
    # Named Entity Recognition (NER)
    print("📦 Extracted Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    # Simple Sentiment Analysis (rule-based)
    tokens = {token.text.lower() for token in doc}
    pos_matches = tokens & positive_keywords
    neg_matches = tokens & negative_keywords

    sentiment = "Positive" if len(pos_matches) > len(neg_matches) else \
                "Negative" if len(neg_matches) > len(pos_matches) else "Neutral"
    
    print(f"💬 Sentiment: {sentiment}")
    print(f"Matched Positive Words: {pos_matches}")
    print(f"Matched Negative Words: {neg_matches}")
