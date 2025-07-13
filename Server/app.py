# Import Flask
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import numpy as np
import joblib
import imdb
import os
import requests
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load('en_core_web_sm')
# Constants for Movie Metadata
BASE_URL = "https://api.themoviedb.org/3"
API_KEY = "fb6cd9a842dd77355df496b80e19bf61"  # Replace with your TMDB API Key
# api key = d4a204101d212bad9e11959d67149dae
TMDB_API_URL = 'https://api.themoviedb.org/3/search/movie'
# Load Data
MoviesData = joblib.load('Movies_Datase.pkl')
X = joblib.load('Movies_Learned_Features.pkl')

my_ratings = np.zeros((9724, 1))
my_movies = []
my_added_movies = []
# Load and preprocess the chatbot data
df = pd.read_csv('main_data.csv')
df.fillna('', inplace=True)


def computeCost(X, y, theta):
    m = y.size
    s = np.dot(X, theta) - y
    j = (1 / (2 * m)) * (np.dot(np.transpose(s), s))
    print(j)
    return j


def gradientDescent(X, y, theta, alpha, num_iters):
    m = float(y.shape[0])
    theta = theta.copy()
    for i in range(num_iters):
        theta = theta - (alpha / m) * \
            (np.dot(np.transpose((np.dot(X, theta) - y)), X))
    return theta


def checkAndAdd(movie, rating):
    try:
        if isinstance(int(rating), str):
            pass
    except ValueError:
        return 3
    if 0 <= int(rating) <= 5:
        movie = movie.lower()
        movie = movie + ' '
        if movie not in MoviesData['title'].unique():
            return 1
        else:
            index = MoviesData[MoviesData['title'] == movie].index.values[0]
            my_ratings[index] = rating
            movieid = MoviesData.loc[MoviesData['title'] == movie, 'movieid']
            if movie in my_added_movies:
                return 2
            my_movies.append(movieid)
            my_added_movies.append(movie)
            return 0
    else:
        return -1


def url_clean(url):
    base, ext = os.path.splitext(url)
    i = url.count('@')
    s2 = url.split('@')[0]
    url = s2 + '@' * i + ext
    return url

# Fetch Movie Metadata including Trailer


def fetch_movie_details(movie_title):
    try:
        # Search for the movie by title
        search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_title}"
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_results = search_response.json().get('results', [])

        if not search_results:
            return None  # No movie found

        # Use the first result for the movie
        movie_id = search_results[0]['id']

        # Fetch movie details including credits and reviews
        details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits,videos,reviews"
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        details = details_response.json()

        # Extract required details
        movie_data = {
            "title": details.get('title', 'N/A'),
            "summary": details.get('overview', 'No summary available'),
            "poster_url": f"https://image.tmdb.org/t/p/w500{details.get('poster_path', '')}",
            "rating": details.get('vote_average', 'N/A'),
            "genres": [genre['name'] for genre in details.get('genres', [])],
            "release_date": details.get('release_date', 'N/A'),
            # Top 5 cast members
            "cast": [cast['name'] for cast in details.get('credits', {}).get('cast', [])[:5]],
        }

        # Extract trailer URL if available
        videos = details.get('videos', {}).get('results', [])
        if videos:
            # Look for the first YouTube trailer (if available)
            trailer = next(
                (video for video in videos if video['site'] == 'YouTube'), None)
            if trailer:
                movie_data["trailer_url"] = f"https://www.youtube.com/embed/{trailer['key']}"
            else:
                movie_data["trailer_url"] = None
        else:
            movie_data["trailer_url"] = None

        # Get Reviews
        reviews = details.get('reviews', {}).get('results', [])
        movie_data["reviews"] = [review['content'] for review in reviews]

        return movie_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie details: {e}")
        return None


# Load the pre-trained model from Hugging Face (for sentiment analysis)
sentiment_pipeline = pipeline(
    "sentiment-analysis",  model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', truncation=True, max_length=512)


def analyze_sentiment(reviews):
    sentiment_results = {"positive": 0, "neutral": 0, "negative": 0}
    review_sentiments = []

    for review in reviews:
        # Perform sentiment analysis using BERT
        sentiment = sentiment_pipeline(review)[0]
        label = sentiment['label']
        score = sentiment['score']

        # Classify sentiment based on BERT output
        if label == 'POSITIVE':
            sentiment_label = "Positive"
            sentiment_results["positive"] += 1
        elif label == 'NEGATIVE':
            sentiment_label = "Negative"
            sentiment_results["negative"] += 1
        else:
            sentiment_label = "Neutral"
            sentiment_results["neutral"] += 1

        # Add the sentiment label and review text to the list
        review_sentiments.append({
            "review": review,
            "sentiment": sentiment_label,
            "score": score  # Add the confidence score from the model
        })

    total_reviews = len(reviews)
    if total_reviews > 0:
        # Calculate percentages
        sentiment_results["positive_percent"] = (
            sentiment_results["positive"] / total_reviews) * 100
        sentiment_results["negative_percent"] = (
            sentiment_results["negative"] / total_reviews) * 100
        sentiment_results["neutral_percent"] = (
            sentiment_results["neutral"] / total_reviews) * 100
    else:
        sentiment_results["positive_percent"] = sentiment_results["negative_percent"] = sentiment_results["neutral_percent"] = 0

    return review_sentiments, sentiment_results


# Initialize the BART model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def generate_summary(movie_summary):
    # Check if the summary is not empty
    if movie_summary:
        # Generate the summary using the BART model
        summary = summarizer(movie_summary, max_length=70,
                             min_length=10, do_sample=False)
        return summary[0]['summary_text']
    return "No summary available to summarize."


# Combine relevant columns for vectorization
df['combined_features'] = df['actor_name'] + ' ' + \
    df['director_name'] + ' ' + df['genres'] + ' ' + df['movie_title']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])


# List of known genres
known_genres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'Sci-Fi', 'TV Movie', 'Thriller', 'War', 'Western'
]

# Function to get movie details from TMDb


def get_movie_details(title):
    params = {
        'api_key': API_KEY,
        'query': title
    }
    response = requests.get(TMDB_API_URL, params=params)
    data = response.json()
    if data['results']:
        movie = data['results'][0]
        return {
            'title': movie['title'],
            'poster_path': 'https://image.tmdb.org/t/p/w500' + movie['poster_path'] if movie['poster_path'] else '',
            'rating': movie['vote_average']
        }
    else:
        return None

# Improved recommendation function with TMDb data


def get_recommendations(query, actor=None, director=None, genre=None):
    # Filter based on actor, director, and genre if provided
    df_filtered = df.copy()
    if actor:
        df_filtered = df_filtered[df_filtered['actor_name'].str.contains(
            actor, case=False, na=False)]
    if director:
        df_filtered = df_filtered[df_filtered['director_name'].str.contains(
            director, case=False, na=False)]
    if genre:
        df_filtered = df_filtered[df_filtered['genres'].str.contains(
            genre, case=False, na=False)]

    # Use cosine similarity for better recommendations
    if not df_filtered.empty:
        query_tfidf = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_tfidf, vectorizer.transform(
            df_filtered['combined_features'])).flatten()
        top_indices = cosine_sim.argsort()[-5:][::-1]
        recommendations = []
        for index in top_indices:
            movie_title = df_filtered.iloc[index]['movie_title']
            movie_details = get_movie_details(movie_title)
            if movie_details:
                recommendations.append(movie_details)
        return recommendations
    else:
        return []

# Extract genre from user input


def extract_genre(user_input):
    doc = nlp(user_input.lower())
    for token in doc:
        if token.text.capitalize() in known_genres:
            return token.text.capitalize()
    return None


# Flask App Setup
app = Flask(__name__)


@app.route('/movieDetails/<movie_title>', methods=['GET'])
def movie_details(movie_title):
    movie_data = fetch_movie_details(movie_title)
    if not movie_data:
        return render_template('error.html', message="Movie details not found!")

    # Perform sentiment analysis on the reviews
    review_sentiments, sentiment_results = analyze_sentiment(
        movie_data.get("reviews", []))

    # Generate a summary of the movie's plot using BART
    generated_summary = generate_summary(movie_data.get("summary", ""))

    return render_template('movie_details.html',
                           movie=movie_data,
                           sentiment=sentiment_results,
                           review_sentiments=review_sentiments,
                           generated_summary=generated_summary)  # Add this line


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/addMovie/', methods=['GET', 'POST'])
def addMovie():
    val = request.form.get('movie_name')
    rating = request.form.get('rating')
    flag = checkAndAdd(val, rating)
    if flag == 1:
        processed_text = "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"
        return render_template('home.html', processed_text=processed_text)
    elif flag == -1:
        processed_text = "Please enter rating between 1-5. This application follows a five-star rating system"
        return render_template('home.html', processed_text=processed_text)
    elif flag == 2:
        processed_text = "The movie has already been added by you"
        return render_template('home.html', processed_text=processed_text)
    elif flag == 3:
        processed_text = "Invalid Input! Please enter a number between 0-5 in the rating field"
        return render_template('home.html', processed_text=processed_text)
    else:
        processed_text = "Successfully added movie to your rated movies"
        movie_text = ", you've rated " + rating + " stars to movie: " + val
        return render_template('home.html', processed_text=processed_text, movie_text=movie_text, my_added_movies=my_added_movies)


@app.route('/reset/', methods=['GET', 'POST'])
def reset():
    global my_ratings
    global my_movies
    global my_added_movies

    my_ratings = np.zeros((9724, 1))
    my_movies = []
    my_added_movies = []
    processed_text = 'Successfully reset'
    return render_template('home.html', processed_text=processed_text)


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        if len(my_added_movies) == 0:
            processed_text = "Yikes! You've to add some movies before predicting anything"
            return render_template('home.html', processed_text=processed_text)

        out_arr = my_ratings[np.nonzero(my_ratings)]
        out_arr = out_arr.reshape(-1, 1)
        idx = np.where(my_ratings)[0]
        X_1 = [X[x] for x in idx]
        X_1 = np.array(X_1)
        y = out_arr.flatten()
        theta = gradientDescent(X_1, y, np.zeros((100)), 0.001, 4000)

        p = X @ theta.T
        p = p.flatten()

        predictedData = MoviesData.copy()
        predictedData['Prediction'] = p
        sorted_data = predictedData.sort_values(
            by=['Prediction'], ascending=False)
        sorted_data = sorted_data[~sorted_data.title.isin(
            my_added_movies)].iloc[:40]

        recommendations = []
        for _, row in sorted_data.iterrows():
            movie_title = row['title']
            try:
                metadata = fetch_movie_details(movie_title)
                poster_url = metadata.get('poster_url', 'N/A')
            except Exception as e:
                print(f"Error fetching metadata for {movie_title}: {e}")
                poster_url = "N/A"
            recommendations.append(
                [movie_title, row['Prediction'], poster_url])

        return render_template('result.html', my_list=recommendations)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    session = request.form.get('session', 'start')
    user_genre = request.form.get('genre', '')
    user_actor = request.form.get('actor', '')
    user_director = request.form.get('director', '')

    if session == 'start':
        response = "Hi! What kind of movie would you like to watch? (e.g., Action, Comedy, Drama)"
        next_session = 'movie_genre'
    elif session == 'movie_genre':
        genre = extract_genre(user_input)
        if genre:
            response = f"Great choice! You've picked {genre}. Do you have any favorite actors or directors in mind?"
            next_session = 'actor_director'
            user_genre = genre  # Save the genre for later use
        else:
            response = "I couldn't recognize that genre. Could you please specify the genre again?"
            next_session = 'movie_genre'
    elif session == 'actor_director':
        doc = nlp(user_input.lower())
        actor, director = None, None
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                if 'actor' in user_input or 'actress' in user_input:
                    actor = ent.text
                elif 'director' in user_input:
                    director = ent.text
        user_actor = actor if actor else user_actor
        user_director = director if director else user_director

        query = f"{user_genre} {user_actor if user_actor else ''} {user_director if user_director else ''}".strip()
        recommendations = get_recommendations(
            query, user_actor, user_director, user_genre)

        if recommendations:
            response = "Here are some movie recommendations for you:"
            for movie in recommendations:
                response += f"<br><b>Title:</b> {movie['title']} <br><b>Rating:</b> {movie['rating']}<br>"
                if movie['poster_path']:
                    response += f"<img src='{movie['poster_path']}' alt='{movie['title']} poster' width='100px'><br>"
            next_session = 'end'
        else:
            response = "Sorry, I couldn't find any recommendations based on your input. Can you specify more details?"
            next_session = 'actor_director'
    else:
        response = "I'm not sure how to help with that. Let's start again. What kind of movie would you like to watch?"
        next_session = 'start'

    return jsonify({'response': response, 'next_session': next_session, 'genre': user_genre, 'actor': user_actor, 'director': user_director})


if __name__ == '__main__':
    app.run()
