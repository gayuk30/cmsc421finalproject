from flask import Flask, request, render_template, jsonify
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import json

app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def index():
    return render_template('index.html')

def extract_transcript_segment(transcript_list, start_time, end_time):
    if end_time <= start_time:
        raise ValueError("End time must be after start time.")

    segment = []
    for item in transcript_list:
        # Validate that each item has the 'start' and 'text' keys
        if 'start' in item and 'text' in item:
            if start_time <= item['start'] < end_time:
                segment.append(item['text'])
        else:
            raise KeyError("Transcript item missing required 'start' or 'text' keys")

    return ' '.join(segment) if segment else "No text found in the specified segment."

def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript_list:
            raise ValueError("Failed to fetch or parse transcript.")
        return transcript_list
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

# Update your route handling to better capture and display exceptions
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        url = request.form['url']
        api_key = request.form['api_key']
        whole_video = 'whole_video' in request.form and request.form['whole_video'] == 'on'
        openai.api_key = api_key

        video_id = get_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        transcript_list = fetch_transcript(video_id)
        if whole_video:
            transcript_segment = ' '.join([item['text'] for item in transcript_list])
        else:
            start_time_hms = request.form.get('start_time')
            end_time_hms = request.form.get('end_time')
            start_seconds = convert_hms_to_seconds(start_time_hms)
            end_seconds = convert_hms_to_seconds(end_time_hms)
            transcript_segment = extract_transcript_segment(transcript_list, start_seconds, end_seconds)

        if not transcript_segment:
            raise ValueError("No transcript data found")

        summary = summarize_text(transcript_segment)
        questions = generate_questions(transcript_segment)
        filtered_questions = filter_questions(questions)
        keywords = extract_keywords(transcript_segment)
        return render_template('summary.html', summary=summary, questions=filtered_questions, video_id=video_id, keywords=keywords)

    except Exception as e:
        return render_template('error.html', message=str(e))


def convert_hms_to_seconds(hms):
    try:
        h, m, s = map(int, hms.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError("Invalid time format. Use hh:mm:ss.")



def get_video_id(url):
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    video_id = parse_qs(query).get('v')
    return video_id[0] if video_id else None



def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Provide a comprehensive summary of this transcript and give me the accuracy rate of the summary."},
                {"role": "user", "content": text}
            ],
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

def extract_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    fdist = FreqDist(filtered_words)
    return [word for word, _ in fdist.most_common(5)]

def find_resources(text):
    keywords = extract_keywords(text)
    search_results = get_search_results(' '.join(keywords))
    relevance_labels = classify_relevance(search_results, keywords)
    X_train, _, y_train, _ = train_test_split(search_results, relevance_labels, test_size=0.2, random_state=42)
    clf, vectorizer = train_classifier(X_train, y_train)
    X_test = vectorizer.transform(search_results)
    predicted_labels = clf.predict(X_test)
    relevant_results = [result for result, label in zip(search_results, predicted_labels) if label == 1]
    return relevant_results

def get_search_results(word):
    try:
        search_results = search(word, num=5, stop=5, pause=2.0)
        return list(search_results)
    except StopIteration:
        return []

def classify_relevance(search_results, keywords):
    relevance = []
    for result in search_results:
        title = result.split(' - ')[0].lower()
        is_relevant = any(keyword in title for keyword in keywords)
        relevance.append(1 if is_relevant else 0)
    return relevance

def train_classifier(X, y):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_train, y)
    return clf, vectorizer

def generate_questions(text):
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        prompt = f"generate question: {sent.text.strip()}"
        result = question_generator(prompt, max_length=50)
        questions.extend([res['generated_text'].strip() for res in result if res['generated_text'].strip().endswith('?')])
    return questions

def filter_questions(questions):
    filtered = []
    for question in questions:
        if len(question) > 10 and question.count('?') == 1:
            filtered.append(question)
    return filtered

if __name__ == '__main__':
    app.run(debug=True)
