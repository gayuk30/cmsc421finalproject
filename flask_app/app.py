# Libraries imported

from flask import Flask, request, render_template, jsonify  # Importing Flask for Front-end
import openai  # Used for accessing OpenAI's API for AI-driven text generation.
from youtube_transcript_api import YouTubeTranscriptApi  # To fetch transcripts for YouTube videos.
import spacy  # Natural language processing library.
from transformers import pipeline  # Hugging Face's library for state-of-the-art machine learning models.
import nltk  # Natural Language Toolkit for text processing and analysis.
from nltk.tokenize import word_tokenize  # For tokenizing strings into words.
from nltk.corpus import stopwords  # To handle common words that are typically ignored.
from nltk.probability import FreqDist  # For frequency distribution of words.
from sklearn.feature_extraction.text import CountVectorizer  # Converts text to a matrix of token counts.
from sklearn.linear_model import LogisticRegression  # Machine learning algorithm for classification.
from sklearn.model_selection import train_test_split  # Splits data into random train and test subsets.
import requests  # To make HTTP requests.
from bs4 import BeautifulSoup  # For parsing HTML and XML documents.
from googlesearch import search  # To perform Google searches.
import json  # For handling JSON data.

app = Flask(__name__)  # Initializing the Flask application

# Load NLP models
nlp = spacy.load("en_core_web_sm")  # Load a SpaCy model for English processing.
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")  # Load a pre-trained model for question generation.

# Download NLTK resources
nltk.download('punkt')  # Punkt Tokenizer Model for English.
nltk.download('stopwords')  # Common words that generally have little lexical content.

@app.route('/')
def index():
    return render_template('index.html')  # Route to serve the home page.

def extract_transcript_segment(transcript_list, start_time, end_time):
    # Extracts a specific segment of the transcript between start_time and end_time.
    if end_time <= start_time:
        raise ValueError("End time must be after start time.")

    segment = []
    for item in transcript_list:
        if 'start' in item and 'text' in item and start_time <= item['start'] < end_time:
            segment.append(item['text'])

    return ' '.join(segment) if segment else "No text found in the specified segment."

def fetch_transcript(video_id):
    # Fetches the YouTube video transcript.
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list if transcript_list else None
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

@app.route('/summarize', methods=['POST'])
def summarize():
    # Handles the summarization of the video transcript.
    try:
        url = request.form['url']
        api_key = request.form['api_key']
        openai.api_key = api_key  # Set the API key for the OpenAI service.

        video_id = get_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        transcript_list = fetch_transcript(video_id)
        transcript_segment = ' '.join([item['text'] for item in transcript_list])  # Summarize the whole video if no timestamps provided.

        summary = summarize_text(transcript_segment)
        questions = generate_questions(transcript_segment)
        filtered_questions = filter_questions(questions)
        keywords = extract_keywords(transcript_segment)

        return render_template('summary.html', summary=summary, questions=filtered_questions, keywords=keywords, video_id=video_id)

    except Exception as e:
        return render_template('error.html', message=str(e))

def convert_hms_to_seconds(hms):
    # Converts hh:mm:ss time format to seconds.
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

def get_video_id(url):
    # Extracts video ID from YouTube URL.
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    video_id = parse_qs(query).get('v')
    return video_id[0] if video_id else None

def summarize_text(text):
    # Uses OpenAI's GPT model to summarize the text.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Provide a comprehensive summary of this transcript."},
                  {"role": "user", "content": text}],
        max_tokens=600
    )
    return response['choices'][0]['message']['content']

def extract_keywords(text):
    # Tokenizes the text into words and filters out common stopwords and non-alphanumeric characters,
    # then returns the five most common words.
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    fdist = FreqDist(filtered_words)
    return [word for word, _ in fdist.most_common(5)]

def find_resources(text):
    # Extracts keywords from the text, searches for related web pages, and evaluates their relevance
    # based on the presence of keywords in the titles. Returns the most relevant results.
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
    # Performs a Google search for the given word and retrieves the top 5 results,
    # returning them as a list. Handles exceptions if the search does not yield results.
    try:
        search_results = search(word, num=5, stop=5, pause=2.0)
        return list(search_results)
    except StopIteration:
        return []

def classify_relevance(search_results, keywords):
    # Evaluates the relevance of search results based on the occurrence of keywords in the title.
    # Returns a list of binary values indicating relevance.
    relevance = []
    for result in search_results:
        title = result.split(' - ')[0].lower()
        is_relevant = any(keyword in title for keyword in keywords)
        relevance.append(1 if is_relevant else 0)
    return relevance

def train_classifier(X, y):
    # Trains a logistic regression classifier using the given data.
    # Returns both the trained classifier and the vectorizer used for feature extraction.
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_train, y)
    return clf, vectorizer

def generate_questions(text):
    # Generates questions based on the provided text using a pre-trained machine learning model.
    # Returns a list of generated questions.
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        prompt = f"generate question: {sent.text.strip()}"
        result = question_generator(prompt, max_length=50)
        questions.extend([res['generated_text'].strip() for res in result if res['generated_text'].strip().endswith('?')])
    return questions

def filter_questions(questions):
    # Filters out questions to ensure each is of a minimum length and ends with a question mark.
    # Returns the filtered list of questions.
    filtered = []
    for question in questions:
        if len(question) > 10 and question.count('?') == 1:
            filtered.append(question)
    return filtered

if __name__ == '__main__':
    app.run(debug=True)  # Starts the Flask application.
