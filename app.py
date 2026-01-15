"""
Movie Recommendation System Demo Application
A Streamlit web app demonstrating three recommendation algorithms on MovieLens-20M
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import requests
from io import BytesIO
from PIL import Image
import sqlite3
import bcrypt
from datetime import datetime

# Recommenders library imports
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k
from recommenders.models.sar import SAR
from recommenders.utils.timer import Timer

# Cornac imports for BiVAE model
try:
    import cornac
    from cornac.models import BiVAECF
    CORNAC_AVAILABLE = True
except ImportError:
    CORNAC_AVAILABLE = False
    print("Cornac not available. BiVAE predictions for new users will be disabled.")

# BiVAE predictions availability flag
BIVAE_AVAILABLE = True  # Will be set to False if predictions file not found

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TOP_K = 10
MOVIELENS_DATA_SIZE = '100k'  # Start with 100k for faster demo, can change to 20m
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_TIMESTAMP = "Timestamp"
COL_PREDICTION = "prediction"
TRAIN_RATIO = 0.75
RANDOM_SEED = 42

# Database configuration
DB_PATH = "movie_ratings.db"
STARTING_USER_ID = 10001  # New users start from this ID

# TMDb API configuration (optional - for movie posters)
TMDB_API_KEY = "8d50cb61259ff8679ebb46e0e4da34a6" # Set your TMDb API key here if available
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/185x278/FF6B6B/FFFFFF?text=No+Poster"

# Pre-computed predictions configuration
BIVAE_PREDICTIONS_PATH = "./all_predictions_bivae.parquet1"  # Path to saved BiVAE predictions (parquet)
BIVAE_MODEL_PATH = "./bivae_model.pkl"  # Path to saved BiVAE model for new user predictions
ALS_PREDICTIONS_PATH = "./all_prediction_als.parquet"  # Path to saved ALS predictions (parquet directory)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Database Functions
# ============================================================================

def init_database():
    """Initialize SQLite database with users and ratings tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if not exists
    cursor.execute('SELECT user_id FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        admin_password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            'INSERT INTO users (user_id, username, password_hash, is_admin) VALUES (?, ?, ?, ?)',
            (1, 'admin', admin_password_hash, 1)
        )
    
    # Create ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            UNIQUE(user_id, movie_id)
        )
    ''')
    
    # Create A/B testing tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ab_impressions (
            impression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            movie_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ab_clicks (
            click_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            movie_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()


def get_next_user_id():
    """Get the next available user ID (starting from STARTING_USER_ID)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(user_id) FROM users')
    max_id = cursor.fetchone()[0]
    conn.close()
    
    if max_id is None:
        return STARTING_USER_ID
    else:
        return max(max_id + 1, STARTING_USER_ID)


def create_user(username, password, is_admin=False):
    """Create a new user with hashed password. Returns (success, user_id or error_message)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists"
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Get next user ID
        user_id = get_next_user_id()
        
        # Insert new user
        cursor.execute(
            'INSERT INTO users (user_id, username, password_hash, is_admin) VALUES (?, ?, ?, ?)',
            (user_id, username, password_hash, 1 if is_admin else 0)
        )
        conn.commit()
        conn.close()
        
        return True, user_id
    except Exception as e:
        return False, str(e)


def authenticate_user(username, password):
    """Authenticate user. Returns (success, user_id or None, is_admin)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, password_hash, is_admin FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[1]):
            return True, result[0], bool(result[2])
        else:
            return False, None, False
    except Exception as e:
        return False, None, False


def add_rating(user_id, movie_id, rating):
    """Add or update a rating for a user. Returns success boolean."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        
        # Insert or replace rating
        cursor.execute('''
            INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, movie_id, rating, timestamp))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False


def delete_rating(user_id, movie_id):
    """Delete a rating for a user. Returns success boolean."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM ratings WHERE user_id = ? AND movie_id = ?', (user_id, movie_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False


def get_user_ratings(user_id):
    """Get all ratings for a user as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    query = f'''
        SELECT user_id as {COL_USER}, movie_id as {COL_ITEM}, 
               rating as {COL_RATING}, timestamp as {COL_TIMESTAMP}
        FROM ratings
        WHERE user_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df


def get_user_rating_count(user_id):
    """Get the number of ratings for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM ratings WHERE user_id = ?', (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ============================================================================
# A/B Testing Functions
# ============================================================================

def get_assigned_algorithm(user_id):
    """Get the assigned algorithm for a user based on user_id % 4."""
    algorithms = ["Popularity-Based", "Item-KNN (SAR)", "ALS (Matrix Factorization)", "BiVAE (Deep Learning)"]
    return algorithms[user_id % 4]


def track_impression(user_id, algorithm, movie_id, rank):
    """Track that a movie recommendation was shown to a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        cursor.execute(
            'INSERT INTO ab_impressions (user_id, algorithm, movie_id, rank, timestamp) VALUES (?, ?, ?, ?, ?)',
            (user_id, algorithm, movie_id, rank, timestamp)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False


def track_click(user_id, algorithm, movie_id, rank):
    """Track that a user clicked on a movie recommendation."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        cursor.execute(
            'INSERT INTO ab_clicks (user_id, algorithm, movie_id, rank, timestamp) VALUES (?, ?, ?, ?, ?)',
            (user_id, algorithm, movie_id, rank, timestamp)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False


# Initialize database on app start
init_database()


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'needs_cold_start' not in st.session_state:
        st.session_state.needs_cold_start = False


def logout():
    """Logout current user."""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.is_admin = False
    st.session_state.needs_cold_start = False


# ============================================================================
# Authentication Pages
# ============================================================================

def show_login_page():
    """Display login page."""
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", max_chars=50)
            password = st.text_input("Password", type="password", max_chars=100)
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, user_id, is_admin = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.is_admin = is_admin
                        
                        # Check if user needs cold start (less than 5 ratings) - skip for admin
                        if not is_admin:
                            rating_count = get_user_rating_count(user_id)
                            if rating_count < 5:
                                st.session_state.needs_cold_start = True
                        
                        st.success(f"Welcome back, {username}!" + (" 👑 (Admin)" if is_admin else ""))
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        st.markdown("---")
        st.markdown("### Don't have an account?")
        if st.button("Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()


def show_signup_page():
    """Display signup page."""
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📝 Create New Account")
        with st.form("signup_form"):
            username = st.text_input("Username", max_chars=50, 
                                   help="Choose a unique username (3-50 characters)")
            password = st.text_input("Password", type="password", max_chars=100,
                                   help="Choose a strong password (min 6 characters)")
            password_confirm = st.text_input("Confirm Password", type="password", max_chars=100)
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                # Validation
                if not username or not password or not password_confirm:
                    st.error("Please fill in all fields")
                elif len(username) < 3:
                    st.error("Username must be at least 3 characters long")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                else:
                    success, result = create_user(username, password)
                    if success:
                        st.success(f"Account created successfully! Your user ID is {result}")
                        st.info("You'll need to rate at least 5 movies before getting recommendations.")
                        time.sleep(2)
                        
                        # Auto login
                        st.session_state.logged_in = True
                        st.session_state.user_id = result
                        st.session_state.username = username
                        st.session_state.is_admin = False
                        st.session_state.needs_cold_start = True
                        st.rerun()
                    else:
                        st.error(f"Signup failed: {result}")
        
        st.markdown("---")
        st.markdown("### Already have an account?")
        if st.button("Back to Login", use_container_width=True):
            if 'show_signup' in st.session_state:
                del st.session_state.show_signup
            st.rerun()


def show_cold_start_page(movie_df):
    """Display cold start rating page for new users."""
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown(f"### 🌟 Welcome, {st.session_state.username}!")
    st.markdown("#### Please rate at least 5 movies to get personalized recommendations")
    
    # Get current user ratings
    user_ratings_df = get_user_ratings(st.session_state.user_id)
    current_rating_count = len(user_ratings_df)
    
    # Progress indicator
    progress = min(current_rating_count / 5, 1.0)
    st.progress(progress)
    st.markdown(f"**{current_rating_count} / 5 movies rated**")
    
    if current_rating_count >= 5:
        st.success("✅ Great! You've rated enough movies. You can now access recommendations!")
        if st.button("Go to Recommendations", type="primary"):
            st.session_state.needs_cold_start = False
            st.rerun()
        st.markdown("---")
        st.markdown("#### Or continue rating more movies below:")
    
    # Movie search and rating section
    st.markdown("---")
    st.markdown("### 🔍 Search and Rate Movies")
    
    # Prepare movie list for search
    if movie_df is not None and not movie_df.empty:
        # Create searchable list with title, year, genres
        movie_list = []
        movie_id_map = {}
        
        for idx, row in movie_df.iterrows():
            movie_id = row.name if COL_ITEM not in movie_df.columns else row[COL_ITEM]
            title = row.get('title_clean', row.get('title', f'Movie {movie_id}'))
            year = row.get('year', '')
            genres = row.get('genres', '')
            
            # Format display string
            display_str = f"{title}"
            if year and year != 'Unknown':
                display_str += f" ({year})"
            if genres and genres != 'Unknown':
                genre_list = str(genres).split('|')[:2]
                display_str += f" - {', '.join(genre_list)}"
            
            movie_list.append(display_str)
            movie_id_map[display_str] = movie_id
        
        # Search box
        selected_movie_display = st.selectbox(
            "Search for a movie:",
            options=[""] + sorted(movie_list),
            index=0,
            help="Start typing to search for movies"
        )
        
        if selected_movie_display:
            movie_id = movie_id_map[selected_movie_display]
            
            # Get movie details
            if COL_ITEM in movie_df.columns:
                movie_row = movie_df[movie_df[COL_ITEM] == movie_id].iloc[0]
            else:
                movie_row = movie_df.loc[movie_id]
            
            # Get movie title and year for fetching details
            title_clean = movie_row.get('title_clean', movie_row.get('title', ''))
            year = movie_row.get('year', 'Unknown')
            
            # Fetch movie details (cast, crew, overview) from TMDb
            with st.spinner('Fetching movie details...'):
                actors, directors, overview = get_movie_details(title_clean, year)
            
            # Display movie info with same layout as recommendations
            poster_col, col1, col2 = st.columns([1, 3, 2], vertical_alignment="top")
            
            with poster_col:
                poster_url = get_movie_poster(title_clean, year if year != 'Unknown' else None)
                try:
                    st.image(poster_url, use_container_width=True)
                except:
                    st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)
            
            with col1:
                # Full title - large font with negative margin to align with poster
                full_title = movie_row.get('title', title_clean)
                imdb_url = get_imdb_search_url(title_clean)
                st.markdown(f"<h2 style='margin-top: -15px;'><a href='{imdb_url}' target='_blank' style='text-decoration: none; color: inherit;'>{full_title}</a></h2>", unsafe_allow_html=True)
                
                # Genres with badges
                genres = movie_row.get('genres', '')
                if pd.notna(genres) and genres not in ['Unknown', '']:
                    genre_list = str(genres).split('|')
                    genre_badges = []
                    for g in genre_list[:4]:  # Show up to 4 genres
                        genre_name = g.strip()
                        emoji = get_genre_emoji(genre_name)
                        genre_badges.append(f"{emoji} {genre_name}")
                    st.markdown(f"{' · '.join(genre_badges)}")
                else:
                    st.markdown("🎭 No genres available")
                
                # Display directors
                if directors:
                    st.markdown(f"🎬 Director: {', '.join(directors)}")
                
                # Display starring actors
                if actors:
                    st.markdown(f"⭐ Starring: {', '.join(actors)}")
            
            with col2:
                # Statistics in metrics - right aligned
                st.markdown("<style>div[data-testid='metric-container'] {text-align: right;}</style>", unsafe_allow_html=True)
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    avg_rating = movie_row.get('avg_rating', 0)
                    if avg_rating > 0:
                        st.metric("Avg Rating", f"{avg_rating:.2f}/5")
                    else:
                        st.metric("Avg Rating", "N/A")
                
                with stat_col2:
                    num_ratings = movie_row.get('num_ratings', 0)
                    if num_ratings > 0:
                        st.metric("# Ratings", f"{int(num_ratings):,}")
                    else:
                        st.metric("# Ratings", "N/A")
            
            # Display overview/description spanning full width below the columns
            if overview:
                st.markdown(f"{overview}")
            
            st.markdown("---")
            
            # Rating input
            st.markdown("### Your Rating")
            
            # Check if user already rated this movie
            existing_rating = None
            if not user_ratings_df.empty:
                existing_df = user_ratings_df[user_ratings_df[COL_ITEM] == movie_id]
                if not existing_df.empty:
                    existing_rating = existing_df.iloc[0][COL_RATING]
            
            rating = st.slider(
                "Rate this movie:",
                min_value=0.5,
                max_value=5.0,
                value=float(existing_rating) if existing_rating is not None else 3.0,
                step=0.5,
                format="%.1f ⭐"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Submit Rating", type="primary", use_container_width=True):
                    if add_rating(st.session_state.user_id, movie_id, rating):
                        st.success(f"✅ Rated '{movie_row.get('title_clean', movie_row.get('title', ''))}' with {rating} stars!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to save rating. Please try again.")
            
            with col2:
                if existing_rating is not None:
                    if st.button("Delete Rating", use_container_width=True):
                        if delete_rating(st.session_state.user_id, movie_id):
                            st.success("✅ Rating deleted!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to delete rating.")
    
    # Show current ratings
    if current_rating_count > 0:
        st.markdown("---")
        st.markdown("### 📝 Your Current Ratings")
        
        # Merge with movie data
        if movie_df is not None and not movie_df.empty:
            if COL_ITEM in movie_df.columns:
                display_df = user_ratings_df.merge(movie_df, on=COL_ITEM, how='left')
            else:
                display_df = user_ratings_df.merge(movie_df, left_on=COL_ITEM, right_index=True, how='left')
        else:
            display_df = user_ratings_df.copy()
            display_df['title'] = display_df[COL_ITEM].apply(lambda x: f"Movie {x}")
        
        # Display ratings
        for idx, row in display_df.sort_values(COL_RATING, ascending=False).iterrows():
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                st.markdown(f"### {'⭐' * int(row[COL_RATING])}")
                st.markdown(f"**{row[COL_RATING]}**/5")
            
            with col2:
                title = row.get('title_clean', row.get('title', f"Movie {row[COL_ITEM]}"))
                st.markdown(f"**{title}**")
                
                genres = row.get('genres', '')
                if pd.notna(genres) and genres not in ['', 'Unknown']:
                    genre_list = str(genres).split('|')[:2]
                    genre_display = []
                    for g in genre_list:
                        emoji = get_genre_emoji(g.strip())
                        genre_display.append(f"{emoji} {g.strip()}")
                    st.markdown(f"{' · '.join(genre_display)}")
            
            with col3:
                year = row.get('year', '')
                if pd.notna(year) and year != 'Unknown':
                    st.markdown(f"📅 {year}")
            
            st.markdown("---")


def get_movie_poster(title, year):
    """Fetch movie poster from TMDb API or return placeholder."""
    try:
        if TMDB_API_KEY:
            # Search for movie on TMDb
            search_url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'query': title,
                'year': year if year != 'Unknown' else None
            }
            response = requests.get(search_url, params=params, timeout=2)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results and results[0].get('poster_path'):
                    poster_url = f"{TMDB_IMAGE_BASE_URL}{results[0]['poster_path']}"
                    return poster_url
        
        # Return placeholder if no API key or poster not found
        return PLACEHOLDER_IMAGE_URL
    except:
        return PLACEHOLDER_IMAGE_URL


def get_imdb_search_url(title):
    """Generate IMDB search URL for a movie title."""
    import urllib.parse
    # Handle None or NaN values
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return "https://www.imdb.com"
    # Clean the title and encode for URL
    clean_title = str(title).strip()
    encoded_title = urllib.parse.quote(clean_title)
    return f"https://www.imdb.com/find/?q={encoded_title}&s=tt&ttype=ft"


@st.cache_data(ttl=3600)
def get_genre_emoji(genre):
    """Return appropriate emoji for each genre."""
    genre_emojis = {
        'Action': '💥',
        'Adventure': '🗺️',
        'Animation': '🎨',
        'Children': '👶',
        'Comedy': '😂',
        'Crime': '🔫',
        'Documentary': '📹',
        'Drama': '🎭',
        'Fantasy': '🧙',
        'Film-Noir': '🕵️',
        'Horror': '👻',
        'Musical': '🎵',
        'Mystery': '🔍',
        'Romance': '❤️',
        'Sci-Fi': '🚀',
        'Thriller': '😱',
        'War': '⚔️',
        'Western': '🤠',
        'IMAX': '🎬'
    }
    return genre_emojis.get(genre, '🎭')


@st.cache_data(ttl=3600)
def get_movie_details(title, year):
    """Fetch movie details including cast, crew, and overview from TMDb API."""
    try:
        if not TMDB_API_KEY:
            return None, None, None
        
        # Search for movie on TMDb
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'year': year if year != 'Unknown' else None
        }
        response = requests.get(search_url, params=params, timeout=3)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                movie_id = results[0]['id']
                overview = results[0].get('overview', None)
                
                # Get movie credits (cast and crew)
                credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
                credits_params = {'api_key': TMDB_API_KEY}
                credits_response = requests.get(credits_url, params=credits_params, timeout=3)
                
                if credits_response.status_code == 200:
                    credits = credits_response.json()
                    
                    # Extract top cast (actors/actresses)
                    cast = credits.get('cast', [])[:5]  # Top 5 actors
                    actors = [actor['name'] for actor in cast]
                    
                    # Extract directors
                    crew = credits.get('crew', [])
                    directors = [person['name'] for person in crew if person['job'] == 'Director']
                    
                    return actors, directors, overview
        
        return None, None, None
    except:
        return None, None, None


def prefetch_movie_details(movies_list):
    """Prefetch movie details for all movies in parallel using threading."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    details_dict = {}
    
    if not TMDB_API_KEY:
        return details_dict
    
    def fetch_single_movie(movie_info):
        title, year, movie_id = movie_info
        actors, directors, overview = get_movie_details(title, year)
        return movie_id, (actors, directors, overview)
    
    # Create list of movies to fetch
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_single_movie, movie_info): movie_info 
            for movie_info in movies_list
        }
        
        for future in as_completed(futures):
            try:
                movie_id, details = future.result()
                details_dict[movie_id] = details
            except:
                pass
    
    return details_dict


class PopularityRecommender:
    """Popularity-based recommender system."""
    
    def __init__(self, col_user=COL_USER, col_item=COL_ITEM, col_rating=COL_RATING):
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.popular_items = None
        
    def fit(self, train_df):
        """Calculate item popularity based on rating count and average rating."""
        item_stats = train_df.groupby(self.col_item).agg({
            self.col_rating: ['count', 'mean']
        })
        item_stats.columns = ['count', 'avg_rating']
        
        # Normalize and combine metrics
        item_stats['count_norm'] = (item_stats['count'] - item_stats['count'].min()) / \
                                    (item_stats['count'].max() - item_stats['count'].min())
        item_stats['rating_norm'] = (item_stats['avg_rating'] - item_stats['avg_rating'].min()) / \
                                     (item_stats['avg_rating'].max() - item_stats['avg_rating'].min())
        
        # Popularity score: 70% count, 30% rating
        item_stats['popularity_score'] = 0.7 * item_stats['count_norm'] + 0.3 * item_stats['rating_norm']
        
        self.popular_items = item_stats.sort_values('popularity_score', ascending=False)
        return self
    
    def recommend_k_items(self, test_df, top_k=TOP_K, remove_seen=True):
        """Recommend top-k popular items for each user."""
        recommendations = []
        
        for user_id in test_df[self.col_user].unique():
            if remove_seen:
                seen_items = test_df[test_df[self.col_user] == user_id][self.col_item].values
                available_items = self.popular_items[~self.popular_items.index.isin(seen_items)]
            else:
                available_items = self.popular_items
            
            top_items = available_items.head(top_k)
            
            for rank, (item_id, row) in enumerate(top_items.iterrows(), 1):
                recommendations.append({
                    self.col_user: user_id,
                    self.col_item: item_id,
                    COL_PREDICTION: row['popularity_score']
                })
        
        return pd.DataFrame(recommendations)
    
    def recommend_for_user(self, user_id, top_k=TOP_K):
        """Recommend top-k items for a specific user."""
        top_items = self.popular_items.head(top_k)
        recommendations = []
        
        for rank, (item_id, row) in enumerate(top_items.iterrows(), 1):
            recommendations.append({
                self.col_item: item_id,
                'score': row['popularity_score'],
                'rank': rank
            })
        
        return pd.DataFrame(recommendations)


@st.cache_data
def load_data():
    """Load MovieLens data with movie metadata."""
    with st.spinner('Loading MovieLens dataset...'):
        df = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP]
        )
        
        # Keep only top 10,000 users with the most ratings for faster inference
        TOP_N_USERS = 10000
        user_counts = df.groupby(COL_USER).size().reset_index(name='count')
        top_users = user_counts.nlargest(TOP_N_USERS, 'count')[COL_USER]
        df = df[df[COL_USER].isin(top_users)]
        
        # Load movie metadata with titles and genres
        try:
            # Load movie metadata including genres
            movie_df = movielens.load_item_df(
                size=MOVIELENS_DATA_SIZE,
                movie_col=COL_ITEM,
                title_col='title',
                genres_col='genres'
            )
            
            # Reset index to make MovieId a regular column for easier manipulation
            if movie_df.index.name == COL_ITEM:
                movie_df = movie_df.reset_index()
            
            # Ensure all required columns exist and are strings
            if 'title' not in movie_df.columns:
                movie_df['title'] = 'Unknown'
            else:
                # Convert to string to handle any non-string values
                movie_df['title'] = movie_df['title'].astype(str)
            
            if 'genres' not in movie_df.columns:
                movie_df['genres'] = 'Unknown'
            else:
                # Convert to string and replace empty or (no genres listed) with Unknown
                movie_df['genres'] = movie_df['genres'].astype(str)
                movie_df['genres'] = movie_df['genres'].replace(['', '(no genres listed)', 'nan'], 'Unknown')
            
            # Parse release year from title (format: "Movie Title (Year)")
            # Extract year - try to get 4-digit year in parentheses at the end
            movie_df['year'] = movie_df['title'].str.extract(r'\((\d{4})\)')[0]
            
            # Clean title by removing year
            movie_df['title_clean'] = movie_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
            
            # If title_clean is empty, use original title
            mask = (movie_df['title_clean'] == '') | (movie_df['title_clean'].isna())
            movie_df.loc[mask, 'title_clean'] = movie_df.loc[mask, 'title']
            
            # Fill NaN years with Unknown
            movie_df['year'] = movie_df['year'].fillna('Unknown')
            
            # Set MovieId as index for easier merging later
            movie_df = movie_df.set_index(COL_ITEM)
            
            # Calculate movie statistics
            movie_stats = df.groupby(COL_ITEM).agg({
                COL_RATING: ['mean', 'count']
            })
            movie_stats.columns = ['avg_rating', 'num_ratings']
            
            # Merge metadata with statistics
            movie_df = movie_df.join(movie_stats, how='left')
            movie_df['avg_rating'] = movie_df['avg_rating'].fillna(0)
            movie_df['num_ratings'] = movie_df['num_ratings'].fillna(0).astype(int)
            
        except Exception as e:
            st.error(f"Error loading movie metadata: {str(e)}")
            # Fallback: create basic movie dataframe
            unique_movies = df[COL_ITEM].unique()
            movie_stats = df.groupby(COL_ITEM).agg({
                COL_RATING: ['mean', 'count']
            })
            movie_stats.columns = ['avg_rating', 'num_ratings']
            
            movie_df = pd.DataFrame({
                COL_ITEM: unique_movies,
                'title': [f'Movie {mid}' for mid in unique_movies],
                'title_clean': [f'Movie {mid}' for mid in unique_movies],
                'genres': ['Unknown'] * len(unique_movies),
                'year': ['Unknown'] * len(unique_movies)
            }).set_index(COL_ITEM)
            
            movie_df = movie_df.join(movie_stats, how='left')
            movie_df['avg_rating'] = movie_df['avg_rating'].fillna(0)
            movie_df['num_ratings'] = movie_df['num_ratings'].fillna(0).astype(int)
        
    return df, movie_df


@st.cache_data
def split_data(df):
    """Split data into train and test sets."""
    train_df, test_df = python_random_split(df, ratio=TRAIN_RATIO, seed=RANDOM_SEED)
    return train_df, test_df


@st.cache_resource
def train_popularity_model(train_df):
    """Train popularity-based model."""
    model = PopularityRecommender()
    model.fit(train_df)
    return model


@st.cache_resource
def train_sar_model(train_df):
    """Train SAR (Item-KNN) model."""
    model = SAR(
        col_user=COL_USER,
        col_item=COL_ITEM,
        col_rating=COL_RATING,
        col_timestamp=COL_TIMESTAMP,
        similarity_type="jaccard",
        time_decay_coefficient=30,
        timedecay_formula=True,
        normalize=True
    )
    model.fit(train_df)
    return model


@st.cache_data
def load_als_predictions():
    """Load pre-computed ALS predictions from parquet file.
    
    Returns a DataFrame with columns: UserId, MovieId, prediction
    """
    try:
        all_predictions = pd.read_parquet(ALS_PREDICTIONS_PATH, engine="fastparquet")
        return all_predictions
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load ALS predictions: {str(e)}")
        return None


def get_als_recommendations_for_user(als_predictions, user_id, available_movie_ids, top_k=TOP_K):
    """Get top-K recommendations for a specific user from ALS predictions.
    
    Args:
        als_predictions: DataFrame with all predictions (UserId, MovieId, prediction)
        user_id: User ID to get recommendations for
        available_movie_ids: List or set of movie IDs that exist in the dataset
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with columns: MovieId, score, rank
    """
    if als_predictions is None:
        return None
    
    # Filter predictions for the specific user
    user_preds = als_predictions[als_predictions[COL_USER] == user_id].copy()
    
    if len(user_preds) == 0:
        return None
    
    # Filter to only include movies that exist in the dataset
    user_preds = user_preds[user_preds[COL_ITEM].isin(available_movie_ids)]
    
    if len(user_preds) == 0:
        return None
    
    # Sort by prediction score and get top-K
    user_preds = user_preds.nlargest(top_k, COL_PREDICTION)
    
    # Format output
    recs = user_preds[[COL_ITEM, COL_PREDICTION]].copy()
    recs = recs.rename(columns={COL_PREDICTION: 'score'})
    recs['rank'] = range(1, len(recs) + 1)
    recs = recs.reset_index(drop=True)
    
    return recs


@st.cache_data
def load_bivae_predictions():
    """Load pre-computed BiVAE predictions from parquet file.
    
    Returns a DataFrame with columns: UserId, MovieId, prediction
    """
    try:
        all_predictions = pd.read_parquet(BIVAE_PREDICTIONS_PATH, engine="fastparquet")
        
        # Rename columns to match app convention if needed
        column_mapping = {}
        if 'userID' in all_predictions.columns:
            column_mapping['userID'] = COL_USER
        if 'itemID' in all_predictions.columns:
            column_mapping['itemID'] = COL_ITEM
        if column_mapping:
            all_predictions = all_predictions.rename(columns=column_mapping)
        
        # Ensure prediction column exists
        if 'prediction' in all_predictions.columns and COL_PREDICTION != 'prediction':
            all_predictions = all_predictions.rename(columns={'prediction': COL_PREDICTION})
        
        return all_predictions
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load BiVAE predictions: {str(e)}")
        return None


def get_bivae_recommendations_for_user(bivae_predictions, user_id, top_k=TOP_K):
    """Get top-K recommendations for a specific user from BiVAE predictions.
    
    Args:
        bivae_predictions: DataFrame with all predictions (UserId, MovieId, prediction)
        user_id: User ID to get recommendations for
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with columns: MovieId, score, rank
    """
    if bivae_predictions is None:
        return None
    
    # Filter predictions for the specific user
    user_preds = bivae_predictions[bivae_predictions[COL_USER] == user_id].copy()
    
    if len(user_preds) == 0:
        return None
    
    # Sort by prediction score and get top-K
    user_preds = user_preds.nlargest(top_k, COL_PREDICTION)
    
    # Format output
    recs = user_preds[[COL_ITEM, COL_PREDICTION]].copy()
    recs = recs.rename(columns={COL_PREDICTION: 'score'})
    recs['rank'] = range(1, len(recs) + 1)
    recs = recs.reset_index(drop=True)
    
    return recs


@st.cache_resource
def load_bivae_model():
    """Load pre-trained BiVAE model from pickle file.
    
    Returns:
        Tuple of (model, movie_id_to_idx) or (None, None) if unavailable
    """
    if not CORNAC_AVAILABLE:
        st.info("BiVAE model requires cornac library (not available)")
        return None, None
    
    try:
        import pickle
        
        # Load the pickled model
        with open(BIVAE_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Create movie_id to index mapping from the model's item mapping
        # Cornac models store mappings in iid_map (item id map)
        movie_id_to_idx = {}
        if hasattr(model, 'iid_map'):
            # iid_map is {original_id: internal_idx}
            movie_id_to_idx = dict(model.iid_map)
        
        return model, movie_id_to_idx
    except FileNotFoundError:
        st.info(f"BiVAE model file not found at {BIVAE_MODEL_PATH}")
        return None, None
    except Exception as e:
        st.warning(f"Failed to load BiVAE model: {str(e)}")
        return None, None


def get_bivae_predictions_for_new_user(model, movie_id_to_idx, user_ratings_df, all_movie_ids, top_k=TOP_K):
    """Generate BiVAE predictions for a new user using their ratings.
    
    Args:
        model: Loaded BiVAE model
        movie_id_to_idx: Dict mapping MovieId to internal index
        user_ratings_df: DataFrame with columns [MovieId, Rating]
        all_movie_ids: List of all available movie IDs
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with columns: MovieId, score, rank
    """
    if model is None or movie_id_to_idx is None:
        return None
    
    try:
        import numpy as np
        
        # Convert user ratings to model's internal indices
        rated_items = []
        ratings = []
        for _, row in user_ratings_df.iterrows():
            movie_id = row[COL_ITEM]
            if movie_id in movie_id_to_idx:
                rated_items.append(movie_id_to_idx[movie_id])
                ratings.append(row[COL_RATING])
        
        if len(rated_items) == 0:
            return None
        
        # For BiVAE, we need to score all items
        # Get indices for all candidate items
        candidate_items = []
        candidate_movie_ids = []
        for movie_id in all_movie_ids:
            if movie_id in movie_id_to_idx:
                candidate_items.append(movie_id_to_idx[movie_id])
                candidate_movie_ids.append(movie_id)
        
        # BiVAE models typically have a score method that takes user preferences
        # For a new user, we can use the model's rate method with the user's rating pattern
        # This is a simplified approach - create a sparse rating vector
        scores = []
        
        # For each candidate item, score it based on the user's ratings
        # This uses the model's internal scoring mechanism
        for item_idx in candidate_items:
            # Skip already rated items
            if item_idx in rated_items:
                scores.append(-np.inf)
            else:
                # Use model's score method
                # BiVAE.score(user_idx, item_idx) but we don't have a user_idx for new users
                # Instead, we can use the model's mu_theta (decoder output) directly
                # For simplicity, use a heuristic: average similarity to rated items
                try:
                    # Get item embedding/representation
                    if hasattr(model, 'beta'):
                        # beta is the item factor matrix in BiVAE
                        item_vec = model.beta[item_idx]
                        
                        # Compute similarity to rated items
                        similarities = []
                        for rated_idx, rating in zip(rated_items, ratings):
                            rated_vec = model.beta[rated_idx]
                            sim = np.dot(item_vec, rated_vec) / (np.linalg.norm(item_vec) * np.linalg.norm(rated_vec) + 1e-10)
                            similarities.append(sim * rating)
                        
                        # Weighted average
                        score = np.mean(similarities) if similarities else 0.0
                        scores.append(score)
                    else:
                        scores.append(0.0)
                except:
                    scores.append(0.0)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            COL_ITEM: candidate_movie_ids,
            'score': scores
        })
        
        # Filter out already rated movies
        rated_movie_ids = set(user_ratings_df[COL_ITEM].values)
        predictions_df = predictions_df[~predictions_df[COL_ITEM].isin(rated_movie_ids)]
        
        # Sort by score and get top-K
        predictions_df = predictions_df.nlargest(top_k, 'score')
        predictions_df['rank'] = range(1, len(predictions_df) + 1)
        predictions_df = predictions_df.reset_index(drop=True)
        
        return predictions_df
    except Exception as e:
        st.error(f"Failed to generate BiVAE predictions for new user: {str(e)}")
        return None


def evaluate_model(test_df, predictions, model_name):
    """Evaluate model and return metrics."""
    map_score = map_at_k(
        test_df, predictions,
        col_user=COL_USER, col_item=COL_ITEM,
        col_rating=COL_RATING, col_prediction=COL_PREDICTION,
        k=TOP_K
    )
    
    ndcg_score = ndcg_at_k(
        test_df, predictions,
        col_user=COL_USER, col_item=COL_ITEM,
        col_rating=COL_RATING, col_prediction=COL_PREDICTION,
        k=TOP_K
    )
    
    return {
        'model': model_name,
        'map': map_score,
        'ndcg': ndcg_score
    }


def main():
    """Main application."""
    
    # Initialize session state
    init_session_state()
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        # Show login or signup page
        if 'show_signup' in st.session_state and st.session_state.show_signup:
            show_signup_page()
        else:
            show_login_page()
        return
    
    # Check if user needs cold start (less than 5 ratings)
    if st.session_state.needs_cold_start:
        # Load movie data for cold start
        df, movie_df = load_data()
        show_cold_start_page(movie_df)
        return
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, {st.session_state.username}! 👋")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Logout button in sidebar
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation pages - add admin pages if user is admin
    if st.session_state.is_admin:
        page = st.sidebar.radio(
            "Navigate to:",
            ["🏠 Home", "🎯 Get Recommendations", "🌟 Rate Movies", "� View All Users", "📋 All Ratings", "🧪 A/B Test Results", "ℹ️ About"]
        )
    else:
        page = st.sidebar.radio(
            "Navigate to:",
            ["🏠 Home", "🎯 Get Recommendations", "🌟 Rate Movies", "ℹ️ About"]
        )
    
    # Load data
    df, movie_df = load_data()
    train_df, test_df = split_data(df)
    
    # Load ALS predictions from parquet (cached, only loads once)
    als_predictions = load_als_predictions()
    if als_predictions is not None:
        st.sidebar.success("✅ ALS predictions loaded")
    else:
        st.sidebar.info(f"ℹ️ ALS predictions not found at {ALS_PREDICTIONS_PATH}")
    
    # Load BiVAE model for predictions (cached, only loads once)
    bivae_model, bivae_movie_id_to_idx = load_bivae_model()
    if bivae_model is not None:
        st.sidebar.success("✅ BiVAE model loaded")
    else:
        st.sidebar.info(f"ℹ️ BiVAE model not found at {BIVAE_MODEL_PATH}")
    
    # Dataset info in sidebar (MovieLens-20M statistics)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.metric("Users", "138,493")
    st.sidebar.metric("Movies", "27,278")
    st.sidebar.metric("Ratings", "20,000,263")
    st.sidebar.metric("Train Set", "15,000,197")
    st.sidebar.metric("Test Set", "5,000,066")
    
    # User info in sidebar
    user_rating_count = get_user_rating_count(st.session_state.user_id)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Your Stats")
    if st.session_state.is_admin:
        st.sidebar.markdown("**Role:** 👑 Administrator")
    st.sidebar.metric("Your Ratings", f"{user_rating_count}")
    
    # Admin mode for testing original dataset users (only show if admin)
    demo_user_id = None
    if st.session_state.is_admin:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧪 Admin Mode")
        demo_mode = st.sidebar.checkbox("Test with dataset user", value=False)
        if demo_mode:
            # Get available user IDs from the dataset
            available_users = sorted(df[COL_USER].unique())
            demo_user_id = st.sidebar.number_input(
                "Enter User ID from dataset:",
                min_value=int(available_users[0]),
                max_value=int(available_users[-1]),
                value=int(available_users[0]),
                step=1,
                help=f"Choose from {len(available_users)} available users"
            )
            st.sidebar.info(f"Testing as User ID: {demo_user_id}")
    
    # Pages
    if page == "🏠 Home":
        show_home_page(df, movie_df)
    
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(train_df, test_df, df, movie_df, als_predictions, bivae_model, bivae_movie_id_to_idx, demo_user_id)
    
    elif page == "🌟 Rate Movies":
        show_cold_start_page(movie_df)
    
    elif page == " All Ratings":
        if st.session_state.is_admin:
            show_all_ratings_page(df, movie_df)
        else:
            st.error("Access denied. Admin privileges required.")
    
    elif page == "🧪 A/B Test Results":
        if st.session_state.is_admin:
            show_ab_test_results()
        else:
            st.error("Access denied. Admin privileges required.")
    
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(df, movie_df):
    """Display home page."""
    
    # Top rated movies
    st.markdown("### 🌟 Top Rated Movies (minimum 50 ratings)")
    top_movies = df.groupby(COL_ITEM).agg({
        COL_RATING: ['mean', 'count']
    }).reset_index()
    top_movies.columns = [COL_ITEM, 'rating_avg', 'rating_count']
    top_movies = top_movies[top_movies['rating_count'] >= 50].sort_values('rating_avg', ascending=False).head(10)
    
    # Merge with movie titles
    if movie_df is not None and not movie_df.empty:
        if COL_ITEM in movie_df.columns:
            top_movies = top_movies.merge(movie_df, on=COL_ITEM, how='left', suffixes=('', '_meta'))
        else:
            top_movies = top_movies.merge(movie_df, left_on=COL_ITEM, right_index=True, how='left', suffixes=('', '_meta'))
        
        # Display with enhanced information
        for rank, (idx, row) in enumerate(top_movies.iterrows(), 1):
            col1, col2, col3 = st.columns([1, 6, 2])
            
            with col1:
                st.markdown(f"### #{rank}")
            
            with col2:
                title = row.get('title_clean', row.get('title', f"Movie {row[COL_ITEM]}"))
                imdb_url = get_imdb_search_url(title)
                st.markdown(f"## [{title}]({imdb_url})")
                
                # Genres
                genres = row.get('genres', '')
                if pd.notna(genres) and genres not in ['', 'Unknown']:
                    genre_list = str(genres).split('|')[:3]
                    genre_display = []
                    for g in genre_list:
                        emoji = get_genre_emoji(g.strip())
                        genre_display.append(f"{emoji} {g.strip()}")
                    st.markdown(f"#### {' · '.join(genre_display)}")
                
                # Year
                year = row.get('year', '')
                if pd.notna(year):
                    st.markdown(f"#### 📅 {year}")
            
            with col3:
                st.metric("Rating", f"{row['rating_avg']:.2f}/5.0")
                st.caption(f"{int(row['rating_count']):,} ratings")
            
            st.markdown("---")
    else:
        for idx, row in top_movies.iterrows():
            st.markdown(f"**Movie {row[COL_ITEM]}** - ⭐ {row['rating_avg']:.2f} ({int(row['rating_count'])} ratings)")


def show_recommendations_page(train_df, test_df, df, movie_df, als_predictions=None, bivae_model=None, bivae_movie_id_to_idx=None, demo_user_id=None):
    """Display recommendations page."""
    
    st.markdown("## 🎯 Get Personalized Recommendations")
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        st.warning("Please login to get personalized recommendations")
        return
    
    # Use demo user ID if provided, otherwise use logged-in user
    active_user_id = demo_user_id if demo_user_id is not None else st.session_state.user_id
    active_username = f"Dataset User {demo_user_id}" if demo_user_id is not None else st.session_state.username
    
    # Get user's database ratings and merge with MovieLens training data
    if demo_user_id is not None:
        # For demo mode, get ratings from the original dataset
        user_db_ratings = df[df[COL_USER] == demo_user_id]
    else:
        # For logged-in users, get ratings from database
        user_db_ratings = get_user_ratings(st.session_state.user_id)
    
    # Combine user's database ratings with MovieLens training data for model training
    combined_train_df = pd.concat([train_df, user_db_ratings], ignore_index=True)
    # Remove duplicates (keep the latest rating for each user-item pair)
    combined_train_df = combined_train_df.drop_duplicates(subset=[COL_USER, COL_ITEM], keep='last')
    
    # Use combined data as user's full history
    user_full_df = pd.concat([df, user_db_ratings], ignore_index=True)
    user_full_df = user_full_df.drop_duplicates(subset=[COL_USER, COL_ITEM], keep='last')
    
    st.markdown(f"### Welcome, {active_username}!")
    st.info(f"You have {len(user_db_ratings)} ratings in your history.")
    
    # Determine if user is new (not in original dataset) or existing
    is_new_user = active_user_id >= STARTING_USER_ID
    
    # A/B Testing: Auto-assign algorithm based on user ID (only for non-admin, non-demo users)
    if demo_user_id is None and not st.session_state.is_admin:
        assigned_algorithm = get_assigned_algorithm(active_user_id)
        # st.info(f"🧪 **A/B Test Mode**: You are assigned to **{assigned_algorithm}**")
        algorithm = assigned_algorithm
        show_algorithm_selector = False
    else:
        # Admin or demo mode: allow manual selection
        show_algorithm_selector = True
        available_algorithms = ["Popularity-Based", "Item-KNN (SAR)"]
        
        # ALS is available if predictions are loaded (for existing users only)
        if als_predictions is not None:
            available_algorithms.append("ALS (Matrix Factorization)")
        
        # BiVAE is available if model is loaded
        if bivae_model is not None:
            available_algorithms.append("BiVAE (Deep Learning)")
        
        algorithm = st.selectbox(
            "Choose Algorithm:",
            available_algorithms
        )
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("Generate Recommendations", type="primary"):
        # Clear previous impressions tracking when generating new recommendations
        if 'impressions_tracked' in st.session_state:
            del st.session_state['impressions_tracked']
        with st.spinner(f'Training {algorithm} model and generating recommendations...'):
            
            if algorithm == "Popularity-Based":
                model = train_popularity_model(combined_train_df)
                recs = model.recommend_for_user(active_user_id, top_k=num_recs)
                
            elif algorithm == "Item-KNN (SAR)":
                model = train_sar_model(combined_train_df)
                # Get user's history for context
                user_history = combined_train_df[combined_train_df[COL_USER] == active_user_id]
                
                if len(user_history) > 0:
                    recs_df = model.recommend_k_items(
                        pd.DataFrame({COL_USER: [active_user_id]}),
                        top_k=num_recs,
                        remove_seen=True
                    )
                    recs = recs_df.rename(columns={COL_PREDICTION: 'score'})
                    recs['rank'] = range(1, len(recs) + 1)
                else:
                    st.warning("You have no ratings. Please rate some movies first.")
                    return
                    
            elif algorithm == "ALS (Matrix Factorization)":
                # Get user's history for context
                user_history = combined_train_df[combined_train_df[COL_USER] == active_user_id]
                
                if len(user_history) == 0:
                    st.warning("You have no ratings. Please rate some movies first.")
                    return
                
                # Check if user is an existing user (in the original dataset)
                is_new_user = active_user_id >= STARTING_USER_ID
                
                if not is_new_user:
                    # Old user: use pre-computed predictions from parquet file
                    if als_predictions is not None:
                        # Get available movie IDs from the dataset
                        available_movie_ids = movie_df.index.tolist()
                        recs = get_als_recommendations_for_user(als_predictions, active_user_id, available_movie_ids, top_k=num_recs)
                        if recs is not None and len(recs) > 0:
                            # st.info("Using pre-computed ALS predictions for existing user.")
                            pass
                        else:
                            # Fallback to SAR if no predictions found
                            sar_model = train_sar_model(combined_train_df)
                            recs_df = sar_model.recommend_k_items(
                                pd.DataFrame({COL_USER: [active_user_id]}),
                                top_k=num_recs,
                                remove_seen=True
                            )
                            if recs_df is not None and len(recs_df) > 0:
                                recs = recs_df[[COL_ITEM, COL_PREDICTION]].copy()
                                recs = recs.rename(columns={COL_PREDICTION: 'score'})
                                recs['rank'] = range(1, len(recs) + 1)
                            else:
                                recs = None
                    else:
                        # Parquet not available, fallback to SAR
                        st.warning("ALS predictions file not available. Falling back to Item-KNN (SAR).")
                        sar_model = train_sar_model(combined_train_df)
                        recs_df = sar_model.recommend_k_items(
                            pd.DataFrame({COL_USER: [active_user_id]}),
                            top_k=num_recs,
                            remove_seen=True
                        )
                        if recs_df is not None and len(recs_df) > 0:
                            recs = recs_df[[COL_ITEM, COL_PREDICTION]].copy()
                            recs = recs.rename(columns={COL_PREDICTION: 'score'})
                            recs['rank'] = range(1, len(recs) + 1)
                        else:
                            recs = None
                else:
                    # New user: fallback to SAR (ALS cannot handle new users without retraining)
                    # st.info("New user detected. Using Item-KNN (SAR) for recommendations.")
                    sar_model = train_sar_model(combined_train_df)
                    recs_df = sar_model.recommend_k_items(
                        pd.DataFrame({COL_USER: [active_user_id]}),
                        top_k=num_recs,
                        remove_seen=True
                    )
                    if recs_df is not None and len(recs_df) > 0:
                        recs = recs_df[[COL_ITEM, COL_PREDICTION]].copy()
                        recs = recs.rename(columns={COL_PREDICTION: 'score'})
                        recs['rank'] = range(1, len(recs) + 1)
                    else:
                        recs = None
            
            else:  # BiVAE (Deep Learning)
                # Get user's history for context
                user_history = combined_train_df[combined_train_df[COL_USER] == active_user_id]
                
                if len(user_history) == 0:
                    st.warning("You have no ratings. Please rate some movies first.")
                    return
                
                # Use BiVAE model for all users
                if bivae_model is not None and bivae_movie_id_to_idx is not None:
                    with st.spinner('Generating BiVAE predictions...'):
                        # Get all available movie IDs (MovieId is the index of movie_df)
                        all_movie_ids = movie_df.index.tolist()
                        
                        # Generate predictions using the model
                        recs = get_bivae_predictions_for_new_user(
                            bivae_model,
                            bivae_movie_id_to_idx,
                            user_history[[COL_ITEM, COL_RATING]],
                            all_movie_ids,
                            top_k=num_recs
                        )
                    
                    if recs is None or len(recs) == 0:
                        st.warning("Could not generate BiVAE predictions. Using popularity-based recommendations.")
                        pop_model = train_popularity_model(combined_train_df)
                        recs = pop_model.recommend_for_user(active_user_id, top_k=num_recs)
                else:
                    st.warning("BiVAE model not available. Using popularity-based recommendations.")
                    pop_model = train_popularity_model(combined_train_df)
                    recs = pop_model.recommend_for_user(active_user_id, top_k=num_recs)
        
        # Store recommendations in session state
        st.session_state['current_recs'] = recs
        st.session_state['current_algorithm'] = algorithm
        st.session_state['current_user_id'] = active_user_id
        st.session_state['demo_user_id'] = demo_user_id
    
    # Display recommendations if available in session state
    if 'current_recs' in st.session_state and st.session_state['current_recs'] is not None:
        recs = st.session_state['current_recs']
        algorithm = st.session_state['current_algorithm']
        active_user_id = st.session_state['current_user_id']
        stored_demo_user_id = st.session_state.get('demo_user_id', None)
        
        st.markdown(f"### 🎬 Top {len(recs)} Recommendations")
        
        # Track impressions for A/B testing (only for regular users, not admin/demo) - only once
        if stored_demo_user_id is None and not st.session_state.is_admin and 'impressions_tracked' not in st.session_state:
            for idx, row in recs.iterrows():
                track_impression(active_user_id, algorithm, row[COL_ITEM], int(row['rank']))
            st.session_state['impressions_tracked'] = True
        
        # Merge with movie titles - use inner join to only keep movies that exist in the dataset
        if movie_df is not None and not movie_df.empty:
            if COL_ITEM in movie_df.columns:
                recs_with_titles = recs.merge(movie_df, on=COL_ITEM, how='inner')
            else:
                recs_with_titles = recs.merge(movie_df, left_on=COL_ITEM, right_index=True, how='inner')
        else:
            recs_with_titles = recs.copy()
            recs_with_titles['title'] = recs_with_titles[COL_ITEM].apply(lambda x: f"Movie {x}")
        
        # Filter out recommendations with NaN titles
        title_col = 'title_clean' if 'title_clean' in recs_with_titles.columns else 'title'
        recs_with_titles = recs_with_titles[
            recs_with_titles[title_col].notna() & 
            (recs_with_titles[title_col] != '') &
            (recs_with_titles[title_col].astype(str).str.strip() != '')
        ].reset_index(drop=True)
        
        # Re-rank after filtering
        recs_with_titles['rank'] = range(1, len(recs_with_titles) + 1)
        
        # Prefetch all movie details in parallel for better performance
        with st.spinner('Fetching movie details...'):
            movies_to_fetch = [
                (
                    row.get('title_clean', row.get('title', f'Movie {row[COL_ITEM]}')),
                    row.get('year', 'Unknown'),
                    row[COL_ITEM]
                )
                for idx, row in recs_with_titles.iterrows()
            ]
            movie_details_cache = prefetch_movie_details(movies_to_fetch)
        
        # Display recommendations with detailed info in cards
        for idx, row in recs_with_titles.iterrows():
            # Create a card for each movie
            rank = int(row['rank'])
            title_clean = row.get('title_clean', row.get('title', f'Movie {row[COL_ITEM]}'))
            avg_rating = row.get('avg_rating', 0)
            year = row.get('year', 'Unknown')
            
            # Create columns with poster on the left
            poster_col, col1, col2 = st.columns([1, 3, 2], vertical_alignment="top")
            
            # Display movie poster
            with poster_col:
                poster_url = get_movie_poster(title_clean, year)
                try:
                    st.image(poster_url, use_container_width=True)
                except:
                    st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)
            
            with col1:
                # Full title with rank - large font with negative margin to align with poster
                full_title = row.get('title', title_clean)
                imdb_url = get_imdb_search_url(title_clean)
                movie_id = row[COL_ITEM]
                click_key = f"click_{movie_id}_{rank}"
                
                # Display title as clickable link
                st.markdown(f"<h2 style='margin-top: -15px;'><a href='{imdb_url}' target='_blank' style='text-decoration: none; color: inherit;'>{full_title}</a></h2>", unsafe_allow_html=True)
                
                # Genres with badges - smaller font
                genres = row.get('genres', 'Unknown')
                if pd.notna(genres) and genres not in ['Unknown', '']:
                    genre_list = str(genres).split('|')
                    # Create colored genre badges with specific emojis
                    genre_badges = []
                    for g in genre_list[:4]:  # Show up to 4 genres
                        genre_name = g.strip()
                        emoji = get_genre_emoji(genre_name)
                        genre_badges.append(f"{emoji} {genre_name}")
                    st.markdown(f"{' · '.join(genre_badges)}")
                else:
                    st.markdown("🎭 No genres available")
                
                # Get cast, crew, and overview information from prefetched cache
                movie_id = row[COL_ITEM]
                actors, directors, overview = movie_details_cache.get(movie_id, (None, None, None))
                
                # Display directors - smaller font
                if directors:
                    st.markdown(f"🎬 Director: {', '.join(directors)}")
                
                # Display starring actors - smaller font
                if actors:
                    st.markdown(f"⭐ Starring: {', '.join(actors)}")
            
            with col2:
                # Statistics in metrics - right aligned
                st.markdown("<style>div[data-testid='metric-container'] {text-align: right;}</style>", unsafe_allow_html=True)
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    if avg_rating > 0:
                        st.metric("Avg Rating", f"{avg_rating:.2f}/5")
                    else:
                        st.metric("Avg Rating", "N/A")
                
                with stat_col2:
                    num_ratings_val = row.get('num_ratings', 0)
                    if num_ratings_val > 0:
                        st.metric("# Ratings", f"{int(num_ratings_val):,}")
                    else:
                        st.metric("# Ratings", "N/A")
            
            # Display overview/description spanning full width below the columns
            if overview:
                st.markdown(f"{overview}")
            
            # Add IMDb button for click tracking (only for non-admin, non-demo users)
            if stored_demo_user_id is None and not st.session_state.is_admin:
                if st.button(f"🎬 View on IMDb", key=click_key, use_container_width=False):
                    track_click(active_user_id, algorithm, movie_id, rank)
                    # Open IMDb in new tab using JavaScript
                    st.markdown(f'<script>window.open("{imdb_url}", "_blank");</script>', unsafe_allow_html=True)
            
            st.markdown("---")
        
        st.markdown("---")
        
        # Show user's rating history with ability to add/edit ratings
        st.markdown("### 📝 Your Rating History")
        
        # Display user's current ratings
        user_ratings = user_db_ratings.sort_values(COL_RATING, ascending=False)
        
        if len(user_ratings) > 0:
            if movie_df is not None and not movie_df.empty:
                if COL_ITEM in movie_df.columns:
                    user_ratings_display = user_ratings.merge(movie_df, on=COL_ITEM, how='left')
                else:
                    user_ratings_display = user_ratings.merge(movie_df, left_on=COL_ITEM, right_index=True, how='left')
            else:
                user_ratings_display = user_ratings.copy()
                user_ratings_display['title'] = user_ratings_display[COL_ITEM].apply(lambda x: f"Movie {x}")
            
            # Display as a nice table with option to delete
            for idx, row in user_ratings_display.iterrows():
                col1, col2, col3, col4 = st.columns([1, 5, 2, 1])
                
                with col1:
                    rating = row[COL_RATING]
                    st.markdown(f"### {'⭐' * int(rating)}")
                    st.markdown(f"**{rating}**/5")
                
                with col2:
                    title = row.get('title_clean', row.get('title', f"Movie {row[COL_ITEM]}"))
                    imdb_url = get_imdb_search_url(title)
                    st.markdown(f"**[{title}]({imdb_url})**")
                    
                    genres = row.get('genres', '')
                    if pd.notna(genres) and genres not in ['', 'Unknown']:
                        genre_list = str(genres).split('|')[:2]
                        genre_display = []
                        for g in genre_list:
                            emoji = get_genre_emoji(g.strip())
                            genre_display.append(f"{emoji} {g.strip()}")
                        st.markdown(f"### {' · '.join(genre_display)}")
                
                with col3:
                    year = row.get('year', 'N/A')
                    if pd.notna(year) and year != 'Unknown':
                        st.markdown(f"### 📅 {year}")
                    
                    avg_rating = row.get('avg_rating', 0)
                    if avg_rating > 0:
                        st.markdown(f"### 📊 Avg: {avg_rating:.2f}/5")
                
                with col4:
                    if st.button("🗑️", key=f"delete_{row[COL_ITEM]}", help="Delete this rating"):
                        if delete_rating(st.session_state.user_id, row[COL_ITEM]):
                            st.success("Rating deleted!")
                            time.sleep(0.5)
                            st.rerun()
                
                st.markdown("---")
        else:
            st.info("You haven't rated any movies yet. Go to the cold start page to rate some movies!")
        
        # Add new rating section
        st.markdown("### ➕ Add New Rating")
        if st.button("Rate More Movies"):
            st.session_state.needs_cold_start = True
            st.rerun()


def show_about_page():
    """Display about page."""
    
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🎬 Movie Recommendation System Demo
    
    This interactive web application demonstrates three different recommendation algorithms 
    on the MovieLens dataset. It's designed for educational purposes and benchmarking.
    
    ### 🧠 Algorithms Implemented
    
    #### 1. Popularity-Based Recommender
    - **Type**: Non-personalized, content-agnostic
    - **Approach**: Ranks items by popularity score (rating count + average rating)
    - **Pros**: Fast, simple, good for cold-start
    - **Cons**: No personalization, filter bubble
    
    #### 2. Item-KNN (SAR - Smart Adaptive Recommendations)
    - **Type**: Item-based collaborative filtering
    - **Approach**: Finds similar items using Jaccard similarity
    - **Pros**: Interpretable, good personalization, efficient
    - **Cons**: Requires user history, limited by item catalog
    
    #### 3. ALS (Alternating Least Squares)
    - **Type**: Matrix factorization
    - **Approach**: Learns latent factors for users and items
    - **Pros**: Best personalization, handles sparsity well
    - **Cons**: Computationally expensive for training
    - **Status**: Available (uses pre-computed predictions)
    
    #### 4. BiVAE (Bilateral Variational Autoencoder)
    - **Type**: Deep learning / Variational autoencoder
    - **Approach**: Learns latent representations using neural networks for both users and items
    - **Pros**: State-of-the-art accuracy, handles complex patterns, symmetric modeling
    - **Cons**: Requires pre-training, computationally intensive
    - **Status**: Available (requires cornac library)
    - **Reference**: Truong et al. "Bilateral Variational Autoencoder for Collaborative Filtering" (WSDM 2021)
    
    ### 📊 Evaluation Metrics
    
    **MAP@K (Mean Average Precision at K)**
    - Measures the precision of recommendations considering the ranking order
    - Emphasizes getting relevant items at the top of the list
    - Range: 0 (worst) to 1 (best)
    
    **NDCG@K (Normalized Discounted Cumulative Gain at K)**
    - Measures ranking quality with position-based discounting
    - Items at higher positions contribute more to the score
    - Range: 0 (worst) to 1 (best)
    
    ### 🛠️ Technology Stack
    
    - **Framework**: Streamlit
    - **ML Library**: Microsoft Recommenders
    - **Data**: MovieLens (100k/1M/10M/20M)
    - **Visualization**: Plotly
    
    ### 📚 Dataset
    
    **MovieLens**
    - Source: GroupLens Research (University of Minnesota)
    - Contains movie ratings from users
    - Multiple sizes available (100k, 1M, 10M, 20M ratings)
    - Includes movie titles and genres
    
    ### 🚀 How to Use
    
    1. **Home**: Explore dataset statistics and top-rated movies
    2. **Get Recommendations**: Select a user and algorithm to get personalized recommendations
    3. **Rate Movies**: Rate movies to get better recommendations
    4. **About**: Learn more about the algorithms and metrics (you are here!)
    
    ### 📖 References
    
    - [Microsoft Recommenders GitHub](https://github.com/microsoft/recommenders)
    - [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
    - [Collaborative Filtering (Wikipedia)](https://en.wikipedia.org/wiki/Collaborative_filtering)
    
    ### 👨‍💻 Development
    
    This application was created for educational purposes to demonstrate different 
    recommendation algorithms and their trade-offs in performance, accuracy, and personalization.
    
    ---
    
    **Note**: For production use with the 20M dataset, consider using a more powerful 
    server and implementing additional optimizations like caching and batch processing.
    """)


def show_all_ratings_page(df, movie_df):
    """Display all ratings page (admin only)."""
    st.markdown("## 📋 All Ratings")
    st.markdown("### View ratings from all users")
    
    # Tabs for different rating sources
    tab1, tab2 = st.tabs(["📝 Registered User Ratings", "📊 Dataset Ratings"])
    
    with tab1:
        st.markdown("### Ratings from Registered Users")
        
        # Get all ratings from database
        conn = sqlite3.connect(DB_PATH)
        all_ratings_df = pd.read_sql_query(
            f'''
            SELECT r.user_id, u.username, r.movie_id, r.rating, r.timestamp
            FROM ratings r
            JOIN users u ON r.user_id = u.user_id
            ORDER BY r.timestamp DESC
            ''',
            conn
        )
        conn.close()
        
        if len(all_ratings_df) > 0:
            # Merge with movie info
            all_ratings_df = all_ratings_df.merge(
                movie_df[['title_clean', 'year', 'genres']].reset_index(),
                left_on='movie_id',
                right_on=COL_ITEM,
                how='left'
            )
            
            # Format timestamp
            all_ratings_df['timestamp'] = pd.to_datetime(all_ratings_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
            
            # User filter
            all_users = sorted(all_ratings_df['username'].unique())
            selected_user = st.selectbox("Filter by user:", ["All"] + all_users)
            
            if selected_user != "All":
                filtered_df = all_ratings_df[all_ratings_df['username'] == selected_user]
            else:
                filtered_df = all_ratings_df
            
            # Display ratings
            st.dataframe(
                filtered_df[['username', 'title_clean', 'year', 'genres', 'rating', 'timestamp']],
                column_config={
                    "username": "User",
                    "title_clean": "Movie",
                    "year": "Year",
                    "genres": "Genres",
                    "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
                    "timestamp": "Date"
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.metric("Total Ratings", len(filtered_df))
        else:
            st.info("No ratings from registered users yet.")
    
    with tab2:
        st.markdown("### Sample Dataset Ratings")
        st.markdown("Showing 1000 random ratings from the MovieLens dataset")
        
        # Sample dataset ratings
        sample_df = df.sample(min(1000, len(df)))
        
        # Merge with movie info
        sample_df = sample_df.merge(
            movie_df[['title_clean', 'year', 'genres']].reset_index(),
            left_on=COL_ITEM,
            right_on=COL_ITEM,
            how='left'
        )
        
        # Format timestamp
        sample_df['timestamp_formatted'] = pd.to_datetime(sample_df[COL_TIMESTAMP], unit='s').dt.strftime('%Y-%m-%d')
        
        # User filter
        user_filter = st.number_input("Filter by User ID (0 for all):", min_value=0, value=0, step=1)
        
        if user_filter > 0:
            filtered_sample = sample_df[sample_df[COL_USER] == user_filter]
        else:
            filtered_sample = sample_df
        
        # Display ratings
        st.dataframe(
            filtered_sample[[COL_USER, 'title_clean', 'year', 'genres', COL_RATING, 'timestamp_formatted']].head(100),
            column_config={
                COL_USER: "User ID",
                "title_clean": "Movie",
                "year": "Year",
                "genres": "Genres",
                COL_RATING: st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
                "timestamp_formatted": "Date"
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.info(f"Showing {min(100, len(filtered_sample))} of {len(filtered_sample)} ratings")


def show_ab_test_results():
    """Display A/B test results page (admin only)."""
    st.markdown("## 🧪 A/B Test Results")
    st.markdown("### Click-Through Rate (CTR) Analysis by Algorithm")
    
    # Get impressions and clicks data
    conn = sqlite3.connect(DB_PATH)
    
    impressions_df = pd.read_sql_query(
        'SELECT algorithm, movie_id, rank, COUNT(*) as impression_count FROM ab_impressions GROUP BY algorithm, movie_id, rank',
        conn
    )
    
    clicks_df = pd.read_sql_query(
        'SELECT algorithm, movie_id, rank, COUNT(*) as click_count FROM ab_clicks GROUP BY algorithm, movie_id, rank',
        conn
    )
    
    # Get overall stats by algorithm
    overall_impressions = pd.read_sql_query(
        'SELECT algorithm, COUNT(*) as total_impressions FROM ab_impressions GROUP BY algorithm',
        conn
    )
    
    overall_clicks = pd.read_sql_query(
        'SELECT algorithm, COUNT(*) as total_clicks FROM ab_clicks GROUP BY algorithm',
        conn
    )
    
    # Get user assignments
    user_assignments = pd.read_sql_query(
        'SELECT DISTINCT user_id, algorithm FROM ab_impressions ORDER BY user_id',
        conn
    )
    
    conn.close()
    
    # Calculate CTR by algorithm
    if len(overall_impressions) > 0:
        ctr_df = overall_impressions.merge(overall_clicks, on='algorithm', how='left')
        ctr_df['total_clicks'] = ctr_df['total_clicks'].fillna(0).astype(int)
        ctr_df['ctr'] = (ctr_df['total_clicks'] / ctr_df['total_impressions'] * 100).round(2)
        ctr_df = ctr_df.sort_values('ctr', ascending=False)
    else:
        ctr_df = pd.DataFrame()
        st.info("No A/B test data available yet. Users need to interact with recommendations first.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall CTR", "📈 Detailed Metrics", "🎯 Position Analysis", "👥 User Assignments"])
    
    with tab1:
        st.markdown("### Overall Click-Through Rate by Algorithm")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3, col4], ctr_df.iterrows())):
            with col:
                st.metric(
                    row['algorithm'],
                    f"{row['ctr']}%",
                    f"{int(row['total_clicks'])} / {int(row['total_impressions'])}"
                )
        
        # Bar chart
        if len(ctr_df) > 0:
            fig = px.bar(
                ctr_df,
                x='algorithm',
                y='ctr',
                title='Click-Through Rate by Algorithm',
                labels={'algorithm': 'Algorithm', 'ctr': 'CTR (%)'},
                color='ctr',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Detailed Metrics by Algorithm")
        
        # Display detailed table
        st.dataframe(
            ctr_df,
            column_config={
                "algorithm": "Algorithm",
                "total_impressions": st.column_config.NumberColumn("Total Impressions", format="%d"),
                "total_clicks": st.column_config.NumberColumn("Total Clicks", format="%d"),
                "ctr": st.column_config.NumberColumn("CTR (%)", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Show top performing movies by algorithm
        st.markdown("### Top Clicked Movies by Algorithm")
        
        selected_algo = st.selectbox("Select Algorithm:", ctr_df['algorithm'].tolist())
        
        # Get click data for selected algorithm
        conn = sqlite3.connect(DB_PATH)
        movie_clicks = pd.read_sql_query(
            f'SELECT movie_id, COUNT(*) as clicks FROM ab_clicks WHERE algorithm = ? GROUP BY movie_id ORDER BY clicks DESC LIMIT 10',
            conn,
            params=(selected_algo,)
        )
        conn.close()
        
        if len(movie_clicks) > 0:
            st.dataframe(
                movie_clicks,
                column_config={
                    "movie_id": "Movie ID",
                    "clicks": st.column_config.NumberColumn("Clicks", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"No clicks recorded for {selected_algo} yet.")
    
    with tab3:
        st.markdown("### CTR by Rank Position")
        
        # Merge impressions and clicks by algorithm and rank
        if len(impressions_df) > 0 and len(clicks_df) > 0:
            rank_analysis = impressions_df.groupby(['algorithm', 'rank']).agg({
                'impression_count': 'sum'
            }).reset_index()
            
            rank_clicks = clicks_df.groupby(['algorithm', 'rank']).agg({
                'click_count': 'sum'
            }).reset_index()
            
            rank_ctr = rank_analysis.merge(rank_clicks, on=['algorithm', 'rank'], how='left')
            rank_ctr['click_count'] = rank_ctr['click_count'].fillna(0).astype(int)
            rank_ctr['ctr'] = (rank_ctr['click_count'] / rank_ctr['impression_count'] * 100).round(2)
            
            # Line chart showing CTR by position for each algorithm
            fig = px.line(
                rank_ctr,
                x='rank',
                y='ctr',
                color='algorithm',
                title='CTR by Rank Position',
                labels={'rank': 'Rank Position', 'ctr': 'CTR (%)', 'algorithm': 'Algorithm'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            pivot_table = rank_ctr.pivot(index='algorithm', columns='rank', values='ctr').fillna(0)
            
            fig_heatmap = px.imshow(
                pivot_table,
                labels=dict(x="Rank Position", y="Algorithm", color="CTR (%)"),
                x=pivot_table.columns,
                y=pivot_table.index,
                title="CTR Heatmap: Algorithm vs Rank Position",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Not enough data for position analysis yet.")
    
    with tab4:
        st.markdown("### User Algorithm Assignments")
        
        if len(user_assignments) > 0:
            # Count users per algorithm
            algo_counts = user_assignments.groupby('algorithm').size().reset_index(name='user_count')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    algo_counts,
                    column_config={
                        "algorithm": "Algorithm",
                        "user_count": st.column_config.NumberColumn("# Users", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                fig = px.pie(
                    algo_counts,
                    values='user_count',
                    names='algorithm',
                    title='User Distribution Across Algorithms'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show individual user assignments
            st.markdown("### Individual User Assignments")
            st.dataframe(
                user_assignments,
                column_config={
                    "user_id": "User ID",
                    "algorithm": "Assigned Algorithm"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No user assignments recorded yet.")


if __name__ == "__main__":
    main()
