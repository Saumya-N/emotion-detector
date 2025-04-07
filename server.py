''' Executing this function initiates the application of emotion
    detection to be executed over the Flask channel and deployed on
    localhost:5000.
'''
from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def emot_detector():
    ''' This code receives the text from the HTML interface and
        runs emotion detection over it using emotion_detector()
        function. The output returned shows the label and its confidence
        score for the provided text.
    '''
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')
    # Pass the text to the emotion_detector function and store the response
    response = emotion_detector(text_to_analyze)
    # Return the response obtained
    # Check for invalid input
    if response['dominant_emotion'] is None:
        return "Invalid text! Please try again!"
    # Format the response string
    result = (f"For the given statement, the system response is "
              f"'anger': {response['anger']}, "
              f"'disgust': {response['disgust']}, "
              f"'fear': {response['fear']}, "
              f"'joy': {response['joy']} and "
              f"'sadness': {response['sadness']}. "
              f"The dominant emotion is {response['dominant_emotion']}.")

    return result

@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application
        page over the Flask channel
    '''
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
