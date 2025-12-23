"""
Movie Recommendation System Demo Application
A Streamlit web app demonstrating three recommendation algorithms on MovieLens-20M
"""

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

# Recommenders library imports
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k
from recommenders.models.sar import SAR
from recommenders.utils.timer import Timer

# Try to import Spark and ALS
try:
    from recommenders.utils.spark_utils import start_or_get_spark
    from recommenders.datasets.spark_splitters import spark_random_split
    from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation
    from recommenders.models.als import ALSWrap
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    start_or_get_spark = None
    spark_random_split = None
    SparkRatingEvaluation = None
    ALSWrap = None
    SparkSession = None

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
MOVIELENS_DATA_SIZE = '20m'  # Start with 100k for faster demo, can change to 20m
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_TIMESTAMP = "Timestamp"
COL_PREDICTION = "prediction"
TRAIN_RATIO = 0.75
RANDOM_SEED = 42

# TMDb API configuration (optional - for movie posters)
TMDB_API_KEY = "8d50cb61259ff8679ebb46e0e4da34a6" # Set your TMDb API key here if available
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/185x278/FF6B6B/FFFFFF?text=No+Poster"

# BiVAE predictions configuration
BIVAE_PREDICTIONS_PATH = "./saved_models/bivae_predictions.pkl"  # Path to saved BiVAE predictions

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
    """Fetch movie details including cast and crew from TMDb API."""
    try:
        if not TMDB_API_KEY:
            return None, None
        
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
                    
                    return actors, directors
        
        return None, None
    except:
        return None, None


def prefetch_movie_details(movies_list):
    """Prefetch movie details for all movies in parallel using threading."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    details_dict = {}
    
    if not TMDB_API_KEY:
        return details_dict
    
    def fetch_single_movie(movie_info):
        title, year, movie_id = movie_info
        actors, directors = get_movie_details(title, year)
        return movie_id, (actors, directors)
    
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
        
        st.info(f"📊 Filtered to top {TOP_N_USERS:,} users | Total ratings: {len(df):,} | Unique items: {df[COL_ITEM].nunique():,}")
        
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


@st.cache_resource
def get_spark_session():
    """Initialize and return Spark session."""
    if not SPARK_AVAILABLE:
        return None
    try:
        spark = start_or_get_spark("MovieRecommender", memory="4g")
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark: {str(e)}")
        return None


@st.cache_data
def load_bivae_predictions():
    """Load pre-computed BiVAE predictions from pickle file.
    
    Returns a DataFrame with columns: UserId, MovieId, prediction
    """
    try:
        with open(BIVAE_PREDICTIONS_PATH, 'rb') as f:
            all_predictions = pickle.load(f)
        
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
def train_als_model(_train_df):
    """Train ALS model using Spark. Cached to avoid retraining."""
    if not SPARK_AVAILABLE:
        return None
    
    spark = get_spark_session()
    if spark is None:
        return None
    
    try:
        # Convert pandas to Spark DataFrame
        train_spark = spark.createDataFrame(_train_df)
        
        # Initialize ALS model
        model = ALSWrap(
            col_user=COL_USER,
            col_item=COL_ITEM,
            col_rating=COL_RATING,
            rank=10,
            maxIter=15,
            regParam=0.05,
            coldStartStrategy='drop'
        )
        
        # Train the model
        with st.spinner('Training ALS model (this will be cached for future use)...'):
            model.fit(train_spark)
        
        return model
    except Exception as e:
        st.error(f"Failed to train ALS model: {str(e)}")
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
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Compare three recommendation algorithms on MovieLens dataset")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Home", "🎯 Get Recommendations", "📊 Model Evaluation", "ℹ️ About"]
    )
    
    # Load data
    df, movie_df = load_data()
    train_df, test_df = split_data(df)
    
    # Pretrain ALS model if Spark is available (cached, only runs once)
    if SPARK_AVAILABLE:
        als_model = train_als_model(train_df)
        if als_model is not None:
            st.sidebar.success("✅ ALS model ready")
    else:
        als_model = None
    
    # Load BiVAE predictions if available (cached, only loads once)
    bivae_predictions = load_bivae_predictions()
    if bivae_predictions is not None:
        st.sidebar.success("✅ BiVAE predictions loaded")
    else:
        st.sidebar.info(f"ℹ️ BiVAE predictions not found at {BIVAE_PREDICTIONS_PATH}")
    
    # Dataset info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.metric("Users", f"{df[COL_USER].nunique():,}")
    st.sidebar.metric("Movies", f"{df[COL_ITEM].nunique():,}")
    st.sidebar.metric("Ratings", f"{len(df):,}")
    st.sidebar.metric("Train Set", f"{len(train_df):,}")
    st.sidebar.metric("Test Set", f"{len(test_df):,}")
    
    # Pages
    if page == "🏠 Home":
        show_home_page(df, movie_df)
    
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(train_df, test_df, df, movie_df, als_model, bivae_predictions)
    
    elif page == "📊 Model Evaluation":
        show_evaluation_page(train_df, test_df, als_model, bivae_predictions)
    
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(df, movie_df):
    """Display home page."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Available Algorithms")
        st.markdown("""
        1. **Popularity-Based** 🔥
           - Recommends globally popular items
           - Fast and simple baseline
           - No personalization
        
        2. **Item-KNN (SAR)** 🔗
           - Item-based collaborative filtering
           - Finds similar items using Jaccard similarity
           - Good balance of speed and quality
        
        3. **ALS (Matrix Factorization)** ⚡
           - Matrix factorization
           - Learns latent factors
           - Best personalization (requires PySpark)
        
        4. **BiVAE (Deep Learning)** 🧠
           - Bilateral Variational Autoencoder
           - Deep learning based collaborative filtering
           - Excellent personalization (requires cornac)
        """)
    
    with col2:
        st.markdown("### 📈 Evaluation Metrics")
        st.markdown("""
        **MAP@K** (Mean Average Precision)
        - Measures precision considering order
        - Higher is better
        - Range: 0 to 1
        
        **NDCG@K** (Normalized Discounted Cumulative Gain)
        - Measures ranking quality
        - Position-based discounting
        - Higher is better
        - Range: 0 to 1
        """)
    
    st.markdown("---")
    
    # Display some statistics
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{df[COL_USER].nunique():,}")
    with col2:
        st.metric("Total Movies", f"{df[COL_ITEM].nunique():,}")
    with col3:
        st.metric("Total Ratings", f"{len(df):,}")
    with col4:
        density = len(df) / (df[COL_USER].nunique() * df[COL_ITEM].nunique())
        st.metric("Sparsity", f"{(1-density)*100:.2f}%")
    
    # Rating distribution
    st.markdown("### 📊 Rating Distribution")
    fig = px.histogram(df, x=COL_RATING, nbins=10,
                       title="Distribution of Ratings",
                       labels={COL_RATING: "Rating"},
                       color_discrete_sequence=['#FF6B6B'])
    st.plotly_chart(fig, use_container_width=True)
    
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
                st.markdown(f"**{title}**")
                
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


def show_recommendations_page(train_df, test_df, df, movie_df, als_model=None, bivae_predictions=None):
    """Display recommendations page."""
    
    st.markdown("## 🎯 Get Personalized Recommendations")
    
    # Algorithm selection
    available_algorithms = ["Popularity-Based", "Item-KNN (SAR)"]
    if SPARK_AVAILABLE:
        available_algorithms.append("ALS (Matrix Factorization)")
    if bivae_predictions is not None:
        available_algorithms.append("BiVAE (Deep Learning)")
    
    algorithm = st.selectbox(
        "Choose Algorithm:",
        available_algorithms
    )
    
    # Show warning if ALS is not available
    if not SPARK_AVAILABLE:
        st.info("ℹ️ ALS algorithm requires PySpark. Install it with: `pip install pyspark`")
    
    # User selection
    users = sorted(df[COL_USER].unique())
    selected_user = st.selectbox(
        "Select User ID:",
        users,
        index=0
    )
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner(f'Training {algorithm} model and generating recommendations...'):
            
            if algorithm == "Popularity-Based":
                model = train_popularity_model(train_df)
                recs = model.recommend_for_user(selected_user, top_k=num_recs)
                
            elif algorithm == "Item-KNN (SAR)":
                model = train_sar_model(train_df)
                # Get user's history for context
                user_history = train_df[train_df[COL_USER] == selected_user]
                
                if len(user_history) > 0:
                    recs_df = model.recommend_k_items(
                        pd.DataFrame({COL_USER: [selected_user]}),
                        top_k=num_recs,
                        remove_seen=True
                    )
                    recs = recs_df.rename(columns={COL_PREDICTION: 'score'})
                    recs['rank'] = range(1, len(recs) + 1)
                else:
                    st.warning(f"User {selected_user} has no ratings in training set. Showing popular items instead.")
                    pop_model = train_popularity_model(train_df)
                    recs = pop_model.recommend_for_user(selected_user, top_k=num_recs)
                    
            elif algorithm == "ALS (Matrix Factorization)":
                if als_model is None:
                    st.error("ALS model is not available. Please ensure PySpark is installed and the model trained successfully.")
                    st.stop()
                
                # Get user's history for context
                user_history = train_df[train_df[COL_USER] == selected_user]
                
                if len(user_history) > 0:
                    spark = get_spark_session()
                    if spark is None:
                        st.error("Cannot get Spark session.")
                        st.stop()
                    
                    # Create test dataframe for single user
                    test_user_df = spark.createDataFrame(
                        pd.DataFrame({COL_USER: [selected_user]})
                    )
                    
                    # Get recommendations using pretrained model
                    recs_spark = als_model.recommend_k_items(
                        test_user_df,
                        top_k=num_recs,
                        remove_seen=True
                    )
                    
                    # Convert to pandas
                    recs = recs_spark.toPandas()
                    recs = recs.rename(columns={COL_PREDICTION: 'score'})
                    recs['rank'] = range(1, len(recs) + 1)
                else:
                    st.warning(f"User {selected_user} has no ratings in training set. Showing popular items instead.")
                    pop_model = train_popularity_model(train_df)
                    recs = pop_model.recommend_for_user(selected_user, top_k=num_recs)
            
            else:  # BiVAE (Deep Learning)
                if bivae_predictions is None:
                    st.error("BiVAE predictions are not available. Please ensure the model is trained and loaded.")
                    st.stop()
                
                # Get user's history for context
                user_history = train_df[train_df[COL_USER] == selected_user]
                
                if len(user_history) > 0:
                    # Get recommendations from pre-computed BiVAE predictions
                    recs = get_bivae_recommendations_for_user(bivae_predictions, selected_user, top_k=num_recs)
                    
                    if recs is None or len(recs) == 0:
                        st.warning(f"User {selected_user} not found in BiVAE predictions. Showing popular items instead.")
                        pop_model = train_popularity_model(train_df)
                        recs = pop_model.recommend_for_user(selected_user, top_k=num_recs)
                else:
                    st.warning(f"User {selected_user} has no ratings in training set. Showing popular items instead.")
                    pop_model = train_popularity_model(train_df)
                    recs = pop_model.recommend_for_user(selected_user, top_k=num_recs)
        
        # Display recommendations
        st.markdown(f"### 🎬 Top {num_recs} Recommendations for User {selected_user}")
        
        # Merge with movie titles
        if movie_df is not None and not movie_df.empty:
            if COL_ITEM in movie_df.columns:
                recs_with_titles = recs.merge(movie_df, on=COL_ITEM, how='left')
            else:
                recs_with_titles = recs.merge(movie_df, left_on=COL_ITEM, right_index=True, how='left')
        else:
            recs_with_titles = recs.copy()
            recs_with_titles['title'] = recs_with_titles[COL_ITEM].apply(lambda x: f"Movie {x}")
        
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
            
            # Main container
            st.markdown(f"#### #{rank}")
            
            # Create columns with poster on the left
            poster_col, col1, col2, col3 = st.columns([1, 3, 2, 1])
            
            # Display movie poster
            with poster_col:
                poster_url = get_movie_poster(title_clean, year)
                try:
                    st.image(poster_url, use_container_width=True)
                except:
                    st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)
            
            with col1:
                # Full title with year - large font
                full_title = row.get('title', title_clean)
                if year != 'Unknown' and str(year) != 'nan':
                    st.markdown(f"## {full_title}")
                else:
                    st.markdown(f"## {full_title}")
                
                # Genres with badges - medium font
                genres = row.get('genres', 'Unknown')
                if pd.notna(genres) and genres not in ['Unknown', '']:
                    genre_list = str(genres).split('|')
                    # Create colored genre badges with specific emojis
                    genre_badges = []
                    for g in genre_list[:4]:  # Show up to 4 genres
                        genre_name = g.strip()
                        emoji = get_genre_emoji(genre_name)
                        genre_badges.append(f"{emoji} {genre_name}")
                    st.markdown(f"### {' · '.join(genre_badges)}")
                else:
                    st.markdown("### 🎭 No genres available")
                
                # Get cast and crew information from prefetched cache
                movie_id = row[COL_ITEM]
                actors, directors = movie_details_cache.get(movie_id, (None, None))
                
                # Display directors - medium font
                if directors:
                    st.markdown(f"### 🎬 Director: {', '.join(directors)}")
                
                # Display starring actors - medium font
                if actors:
                    st.markdown(f"### ⭐ Starring: {', '.join(actors)}")
            
            with col2:
                # Statistics in metrics
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    if avg_rating > 0:
                        st.metric("Avg Rating", f"⭐ {avg_rating:.1f}/5")
                    else:
                        st.metric("Avg Rating", "N/A")
                
                with stat_col2:
                    num_ratings = row.get('num_ratings', 0)
                    if num_ratings > 0:
                        st.metric("# Ratings", f"{int(num_ratings):,}")
                    else:
                        st.metric("# Ratings", "N/A")
                
                # Year - medium font
                if year != 'Unknown' and str(year) != 'nan':
                    st.markdown(f"### 📅 Released: {year}")
                else:
                    st.markdown("### 📅 Year: Unknown")
            
            with col3:
                # Recommendation score
                score = row.get('score', 0)
                st.metric("Rec Score", f"{score:.3f}")
                st.caption(f"ID: {row[COL_ITEM]}")
            
            st.markdown("---")
        
        st.markdown("---")
        
        # Show user's rating history
        st.markdown("### 📝 User's Rating History (Top 10)")
        user_ratings = df[df[COL_USER] == selected_user].sort_values(COL_RATING, ascending=False).head(10)
        
        if len(user_ratings) > 0:
            if movie_df is not None and not movie_df.empty:
                if COL_ITEM in movie_df.columns:
                    user_ratings_display = user_ratings.merge(movie_df, on=COL_ITEM, how='left')
                else:
                    user_ratings_display = user_ratings.merge(movie_df, left_on=COL_ITEM, right_index=True, how='left')
            else:
                user_ratings_display = user_ratings.copy()
                user_ratings_display['title'] = user_ratings_display[COL_ITEM].apply(lambda x: f"Movie {x}")
            
            # Display as a nice table
            for idx, row in user_ratings_display.iterrows():
                col1, col2, col3 = st.columns([1, 5, 2])
                
                with col1:
                    rating = row[COL_RATING]
                    st.markdown(f"### {'⭐' * int(rating)}")
                    st.markdown(f"**{rating}**/5")
                
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
                        st.markdown(f"### {' · '.join(genre_display)}")
                
                with col3:
                    year = row.get('year', 'N/A')
                    if pd.notna(year):
                        st.markdown(f"### 📅 {year}")
                    
                    avg_rating = row.get('avg_rating', 0)
                    if avg_rating > 0:
                        st.markdown(f"### 📊 Avg: {avg_rating:.1f}/5")
                
                st.markdown("---")
        else:
            st.info(f"User {selected_user} has no ratings in the dataset.")


def show_evaluation_page(train_df, test_df, als_model=None, bivae_predictions=None):
    """Display evaluation page."""
    
    st.markdown("## 📊 Model Evaluation & Comparison")
    st.markdown("Compare the performance of different recommendation algorithms using ranking metrics.")
    
    if st.button("Run Evaluation", type="primary"):
        
        results = []
        
        # Evaluate Popularity-based
        with st.spinner('Evaluating Popularity-based model...'):
            pop_model = train_popularity_model(train_df)
            
            with Timer() as pred_time:
                pop_predictions = pop_model.recommend_k_items(test_df, top_k=TOP_K, remove_seen=True)
            
            pop_results = evaluate_model(test_df, pop_predictions, "Popularity")
            pop_results['pred_time'] = pred_time.interval
            results.append(pop_results)
            
            st.success(f"✅ Popularity: MAP@{TOP_K}={pop_results['map']:.4f}, NDCG@{TOP_K}={pop_results['ndcg']:.4f}")
        
        # Evaluate SAR
        with st.spinner('Evaluating Item-KNN (SAR) model...'):
            sar_model = train_sar_model(train_df)
            
            with Timer() as pred_time:
                sar_predictions = sar_model.recommend_k_items(test_df, top_k=TOP_K, remove_seen=True)
            
            sar_results = evaluate_model(test_df, sar_predictions, "Item-KNN (SAR)")
            sar_results['pred_time'] = pred_time.interval
            results.append(sar_results)
            
            st.success(f"✅ Item-KNN: MAP@{TOP_K}={sar_results['map']:.4f}, NDCG@{TOP_K}={sar_results['ndcg']:.4f}")
        
        # Evaluate ALS if available
        if SPARK_AVAILABLE and als_model is not None:
            with st.spinner('Evaluating ALS model...'):
                spark = get_spark_session()
                if spark is not None:
                    try:
                        with Timer() as pred_time:
                            test_spark = spark.createDataFrame(test_df)
                            als_predictions_spark = als_model.recommend_k_items(
                                test_spark,
                                top_k=TOP_K,
                                remove_seen=True
                            )
                            als_predictions = als_predictions_spark.toPandas()
                        
                        als_results = evaluate_model(test_df, als_predictions, "ALS")
                        als_results['pred_time'] = pred_time.interval
                        results.append(als_results)
                        
                        st.success(f"✅ ALS: MAP@{TOP_K}={als_results['map']:.4f}, NDCG@{TOP_K}={als_results['ndcg']:.4f}")
                    except Exception as e:
                        st.warning(f"⚠️ ALS evaluation failed: {str(e)}")
                else:
                    st.warning("⚠️ Spark session could not be initialized. Skipping ALS evaluation.")
        elif SPARK_AVAILABLE and als_model is None:
            st.warning("⚠️ ALS model not available. Skipping ALS evaluation.")
        else:
            st.info("ℹ️ ALS evaluation skipped (PySpark not installed)")
        
        # Evaluate BiVAE if available
        if bivae_predictions is not None:
            with st.spinner('Evaluating BiVAE model...'):
                try:
                    with Timer() as pred_time:
                        # BiVAE predictions are already computed, just filter for top-K
                        bivae_topk = bivae_predictions.groupby(COL_USER).apply(
                            lambda x: x.nlargest(TOP_K, COL_PREDICTION)
                        ).reset_index(drop=True)
                    
                    bivae_results = evaluate_model(test_df, bivae_topk, "BiVAE")
                    bivae_results['pred_time'] = pred_time.interval
                    results.append(bivae_results)
                    
                    st.success(f"✅ BiVAE: MAP@{TOP_K}={bivae_results['map']:.4f}, NDCG@{TOP_K}={bivae_results['ndcg']:.4f}")
                except Exception as e:
                    st.warning(f"⚠️ BiVAE evaluation failed: {str(e)}")
        else:
            st.info("ℹ️ BiVAE evaluation skipped (predictions not loaded)")
        
        # Results table
        st.markdown("### 📋 Results Summary")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('map', ascending=False).reset_index(drop=True)
        
        # Format for display
        display_df = results_df.copy()
        display_df['MAP@10'] = display_df['map'].apply(lambda x: f"{x:.4f}")
        display_df['NDCG@10'] = display_df['ndcg'].apply(lambda x: f"{x:.4f}")
        display_df['Prediction Time (s)'] = display_df['pred_time'].apply(lambda x: f"{x:.2f}")
        display_df = display_df[['model', 'MAP@10', 'NDCG@10', 'Prediction Time (s)']]
        display_df.columns = ['Algorithm', 'MAP@10', 'NDCG@10', 'Prediction Time (s)']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Winner announcement
        best_model = results_df.iloc[0]['model']
        st.success(f"🏆 Best performing model: **{best_model}**")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        # Color palette for bars
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3'][:len(results_df)]
        
        with col1:
            # MAP comparison
            fig_map = go.Figure(data=[
                go.Bar(
                    x=results_df['model'],
                    y=results_df['map'],
                    marker_color=colors,
                    text=results_df['map'].apply(lambda x: f'{x:.4f}'),
                    textposition='outside'
                )
            ])
            fig_map.update_layout(
                title=f"MAP@{TOP_K} Comparison",
                xaxis_title="Algorithm",
                yaxis_title=f"MAP@{TOP_K}",
                height=400
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            # NDCG comparison
            fig_ndcg = go.Figure(data=[
                go.Bar(
                    x=results_df['model'],
                    y=results_df['ndcg'],
                    marker_color=colors,
                    text=results_df['ndcg'].apply(lambda x: f'{x:.4f}'),
                    textposition='outside'
                )
            ])
            fig_ndcg.update_layout(
                title=f"NDCG@{TOP_K} Comparison",
                xaxis_title="Algorithm",
                yaxis_title=f"NDCG@{TOP_K}",
                height=400
            )
            st.plotly_chart(fig_ndcg, use_container_width=True)
        
        # Performance time comparison
        fig_time = go.Figure(data=[
            go.Bar(
                x=results_df['model'],
                y=results_df['pred_time'],
                marker_color='#45B7D1',
                text=results_df['pred_time'].apply(lambda x: f'{x:.2f}s'),
                textposition='outside'
            )
        ])
        fig_time.update_layout(
            title="Prediction Time Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Time (seconds)",
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)


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
    - **Cons**: Computationally expensive, requires PySpark
    - **Status**: Available (requires PySpark installation)
    
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
    3. **Model Evaluation**: Compare algorithms using MAP@K and NDCG@K metrics
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


if __name__ == "__main__":
    main()
