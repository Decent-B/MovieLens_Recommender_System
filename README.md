# Movie Recommendation System Demo App 🎬

An interactive web application demonstrating three recommendation algorithms on the MovieLens dataset.

## Features

- **3 Recommendation Algorithms**:
  - Popularity-Based (non-personalized baseline)
  - Item-KNN using SAR (Smart Adaptive Recommendations)
  - ALS available in notebook version (requires PySpark)

- **Interactive UI**:
  - 🏠 Home: Dataset statistics and exploration
  - 🎯 Get Recommendations: Personalized movie recommendations for any user
  - 📊 Model Evaluation: Compare algorithms with MAP@K and NDCG@K metrics
  - ℹ️ About: Learn about algorithms and metrics

- **Evaluation Metrics**:
  - MAP@K (Mean Average Precision)
  - NDCG@K (Normalized Discounted Cumulative Gain)

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Explore the Dataset** (Home page):
   - View dataset statistics
   - See rating distributions
   - Discover top-rated movies

2. **Get Recommendations**:
   - Select a user ID
   - Choose an algorithm
   - Get personalized movie recommendations
   - View user's rating history

3. **Evaluate Models**:
   - Run evaluation on all algorithms
   - Compare MAP@K and NDCG@K scores
   - View performance benchmarks
   - See visualization charts

## Dataset

The app uses the **MovieLens dataset** from GroupLens Research:
- Default: 100k ratings (for fast demo)
- Can be configured for 1M, 10M, or 20M ratings
- Contains user ratings, movie IDs, and movie titles

To change dataset size, edit the `MOVIELENS_DATA_SIZE` variable in `app.py`:
```python
MOVIELENS_DATA_SIZE = '100k'  # Options: '100k', '1m', '10m', '20m'
```

## Algorithms Explained

### 1. Popularity-Based
- Recommends globally popular movies
- Score based on: 70% rating count + 30% average rating
- Fast but not personalized
- Good for cold-start scenarios

### 2. Item-KNN (SAR)
- Item-based collaborative filtering
- Uses Jaccard similarity to find similar items
- Personalized based on user history
- Good balance of speed and quality

### 3. ALS (Notebook Only)
- Matrix factorization approach
- Learns latent factors for users and items
- Best personalization but requires PySpark
- See `movie_recommendation_demo.ipynb` for implementation

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── movie_recommendation_demo.ipynb # Jupyter notebook with ALS
├── als_movielens.ipynb            # Original ALS example
└── README.md                      # This file
```

## Configuration

Edit these constants in `app.py` to customize:

```python
TOP_K = 10                          # Number of recommendations
MOVIELENS_DATA_SIZE = '100k'       # Dataset size
TRAIN_RATIO = 0.75                 # Train/test split ratio
RANDOM_SEED = 42                   # Random seed for reproducibility
```

## Performance Notes

- **100k dataset**: Runs smoothly on any machine
- **1M dataset**: Still fast, recommended for demos
- **10M/20M datasets**: Slower, recommended for servers with more RAM
- Models are cached using `@st.cache_resource` for faster subsequent runs

## Requirements

- Python 3.8+
- 4GB+ RAM (for 100k dataset)
- 16GB+ RAM recommended (for 20M dataset)

## Technologies

- **Streamlit**: Web framework
- **Microsoft Recommenders**: ML library
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## References

- [Microsoft Recommenders](https://github.com/microsoft/recommenders)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is for educational purposes. MovieLens data is provided by GroupLens Research.

## Troubleshooting

**Issue**: App is slow
- Solution: Use smaller dataset (100k or 1m)
- Clear Streamlit cache: Press 'C' in the running app

**Issue**: Memory error
- Solution: Reduce dataset size or increase available RAM

**Issue**: Module not found
- Solution: `pip install -r requirements.txt`

## Next Steps

- Try different dataset sizes
- Experiment with algorithm parameters
- Compare results with the notebook version (includes ALS)
- Deploy to Streamlit Cloud for public access

---

**Enjoy exploring movie recommendations! 🍿**
