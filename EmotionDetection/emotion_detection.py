import json
import requests


def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(url, json=myobj, headers=header)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # # Parsing the JSON response from the API
    formatted_response = json.loads(response.text)

    # Extract the emotion scores correctly
    emotion_data = formatted_response['emotionPredictions'][0]['emotion']

    anger = emotion_data['anger']
    disgust = emotion_data['disgust']
    fear = emotion_data['fear']
    joy = emotion_data['joy']
    sadness = emotion_data['sadness']

    emotions = {
        'anger': anger,
        'disgust': disgust,
        'fear': fear,
        'joy': joy,
        'sadness': sadness
    }

    dominant_emotion = max(emotions, key=emotions.get)

    emotions['dominant_emotion'] = dominant_emotion
    # Returning a dictionary containing sentiment analysis results
    return emotions
