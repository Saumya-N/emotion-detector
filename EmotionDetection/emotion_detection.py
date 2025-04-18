from transformers import pipeline
# Initialize the Hugging Face emotion classifier
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

def emotion_detector(text_to_analyse):
    # Run the classifier and flatten nested lists if needed
    raw_result = classifier(text_to_analyse)
    emotion_list = raw_result[0] if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], list) else raw_result

    # Build a label->score map
    scores = {item['label'].lower(): item['score'] for item in emotion_list}

    # Extract individual emotions with defaults
    anger = scores.get('anger', 0.0)
    love = scores.get('love', 0.0)
    fear = scores.get('fear', 0.0)
    joy = scores.get('joy', 0.0)
    sadness = scores.get('sadness', 0.0)
    surprise = scores.get('surprise', 0.0)

    # Compose result dict
    emotions = {
        'sadness': sadness,
        'joy': joy,
        'love': love,
        'anger': anger,
        'fear': fear,
        'surprise': surprise
    }

    # Determine dominant emotion
    emotions['dominant_emotion'] = max(emotions, key=emotions.get)
    return emotions
