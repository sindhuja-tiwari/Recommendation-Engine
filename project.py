import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.movie_features = None
        
    def load_data(self, movies_path='movies.csv', ratings_path='ratings.csv'):
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"✓ Loaded {len(self.movies_df)} movies")
        print(f"✓ Loaded {len(self.ratings_df)} ratings")
        print(f"✓ User-Item Matrix: {self.user_item_matrix.shape}")
        
    def preprocess_data(self):
        genres = self.movies_df['genres'].str.get_dummies('|')
        self.movie_features = pd.concat([
            self.movies_df[['movieId', 'title']], 
            genres
        ], axis=1)
        
        print(f"✓ Created {len(genres.columns)} genre features")
    def collaborative_filtering_user_based(self, user_id, n_recommendations=10):
        print(f"\n🔍 Running User-Based Collaborative Filtering for User {user_id}...")
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found!")
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate similarity with all other users
        similarities = []
        for other_user_id in self.user_item_matrix.index:
            if other_user_id == user_id:
                continue
                
            other_ratings = self.user_item_matrix.loc[other_user_id]
            
            # Find common rated movies
            common_movies = (user_ratings > 0) & (other_ratings > 0)
            
            if common_movies.sum() < 2:  # Need at least 2 common movies
                continue
                
            # Calculate Pearson correlation
            user_common = user_ratings[common_movies]
            other_common = other_ratings[common_movies]
            
            correlation = np.corrcoef(user_common, other_common)[0, 1]
            
            if not np.isnan(correlation) and correlation > 0:
                similarities.append((other_user_id, correlation))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = similarities[:50]
        
        print(f"✓ Found {len(top_similar_users)} similar users")
        
        # Predict ratings for unrated movies
        unrated_movies = user_ratings[user_ratings == 0].index
        predictions = {}
        
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_id, similarity in top_similar_users:
                rating = self.user_item_matrix.loc[similar_user_id, movie_id]
                
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predictions[movie_id] = weighted_sum / similarity_sum
        
        # Get top recommendations
        top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_movies, 'Collaborative-User')
    
    def collaborative_filtering_item_based(self, user_id, n_recommendations=10):
        print(f"\n🎬 Running Item-Based Collaborative Filtering for User {user_id}...")
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found!")
            return []
        
        # Transpose matrix for item-item similarity
        item_user_matrix = self.user_item_matrix.T
        
        # Calculate item-item similarity
        item_similarity = cosine_similarity(item_user_matrix)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
        
        # Get user's highly rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        liked_movies = user_ratings[user_ratings >= 4].index
        
        print(f"✓ Found {len(liked_movies)} highly rated movies")
        
        # Find similar movies
        movie_scores = {}
        
        for movie_id in liked_movies:
            similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:51]
            
            for sim_movie_id, similarity in similar_movies.items():
                if user_ratings[sim_movie_id] == 0:
                    if sim_movie_id not in movie_scores:
                        movie_scores[sim_movie_id] = 0
                    movie_scores[sim_movie_id] += similarity * user_ratings[movie_id]
        
        top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_movies, 'Collaborative-Item')
    
    # ==================== CONTENT-BASED FILTERING ====================
    
    def content_based_filtering(self, user_id, n_recommendations=10):
        """
        Content-Based Filtering using movie genres
        Recommends movies with similar content to user's preferences
        """
        print(f"\n📚 Running Content-Based Filtering for User {user_id}...")
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found!")
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        liked_movies = user_ratings[user_ratings >= 4].index.tolist()
        
        if len(liked_movies) == 0:
            print("❌ No highly rated movies found for this user")
            return []
        
        print(f"✓ Analyzing {len(liked_movies)} liked movies")
        
        # Get genre features for liked movies
        liked_features = self.movie_features[
            self.movie_features['movieId'].isin(liked_movies)
        ].drop(['movieId', 'title'], axis=1)
        
        # Create user genre profile
        user_profile = liked_features.mean()
        
        # Get unrated movies
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        unrated_features = self.movie_features[
            self.movie_features['movieId'].isin(unrated_movies)
        ]
        
        # Calculate similarity
        scores = {}
        for idx, row in unrated_features.iterrows():
            movie_id = row['movieId']
            movie_genres = row.drop(['movieId', 'title'])
            
            # Cosine similarity
            similarity = np.dot(user_profile, movie_genres) / (
                np.linalg.norm(user_profile) * np.linalg.norm(movie_genres) + 1e-10
            )
            scores[movie_id] = similarity
        
        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_movies, 'Content-Based')
    def matrix_factorization_svd(self, user_id, n_recommendations=10, n_factors=50):
        print(f"\n🧮 Running Matrix Factorization (SVD) for User {user_id}...")
        
        from scipy.sparse.linalg import svds
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found!")
            return []
        
        # Convert to numpy array
        R = self.user_item_matrix.values
        
        # Normalize by user mean
        user_ratings_mean = np.mean(R, axis=1)
        R_normalized = R - user_ratings_mean.reshape(-1, 1)
        
        print(f"Performing SVD with {n_factors} factors...")
        
        # Perform SVD
        U, sigma, Vt = svds(R_normalized, k=n_factors)
        sigma = np.diag(sigma)
        
        # Predict ratings
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        
        # Get predictions for the user
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_predictions = predicted_ratings[user_idx]
        
        # Get unrated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_mask = user_ratings == 0
        
        movie_ids = self.user_item_matrix.columns[unrated_mask]
        scores = user_predictions[unrated_mask]
        
        recommendations = list(zip(movie_ids, scores))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✓ Generated predictions using {n_factors} latent factors")
        
        return self._format_recommendations(recommendations[:n_recommendations], 'Matrix-Factorization')
    def hybrid_recommendations(self, user_id, n_recommendations=10):
        print(f"\n🎯 Running Hybrid Recommendations for User {user_id}...")
        
        # Get recommendations from each method
        collab_recs = self.collaborative_filtering_user_based(user_id, n_recommendations * 2)
        content_recs = self.content_based_filtering(user_id, n_recommendations * 2)
        svd_recs = self.matrix_factorization_svd(user_id, n_recommendations * 2)
        
        # Combine scores with weights
        combined_scores = {}
        
        # Collaborative (50%)
        for movie_id, score, _, _, _ in collab_recs:
            combined_scores[movie_id] = score * 0.5
        
        # Content-based (30%)
        for movie_id, score, _, _, _ in content_recs:
            if movie_id in combined_scores:
                combined_scores[movie_id] += score * 0.3
            else:
                combined_scores[movie_id] = score * 0.3
        
        # SVD (20%)
        for movie_id, score, _, _, _ in svd_recs:
            if movie_id in combined_scores:
                combined_scores[movie_id] += score * 0.2
            else:
                combined_scores[movie_id] = score * 0.2
        
        top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        print(f"✓ Combined {len(collab_recs)} + {len(content_recs)} + {len(svd_recs)} recommendations")
        
        return self._format_recommendations(top_movies, 'Hybrid')
    def _format_recommendations(self, recommendations, method):
        """Format recommendations with movie details"""
        result = []
        for movie_id, score in recommendations:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                result.append((
                    movie_id,
                    score,
                    movie_info.iloc[0]['title'],
                    movie_info.iloc[0]['genres'],
                    method
                ))
        return result
    
    def get_user_profile(self, user_id):
        """Get user's rating history"""
        if user_id not in self.user_item_matrix.index:
            return None
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        profile = []
        for movie_id, rating in rated_movies.items():
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                profile.append({
                    'movieId': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'rating': rating
                })
        
        return sorted(profile, key=lambda x: x['rating'], reverse=True)
    
    def evaluate_model(self, test_size=0.2, algorithm='collaborative'):
        """Evaluate recommendation quality using RMSE and MAE"""
        print(f"\n📊 Evaluating {algorithm} model...")
        
        # Split data
        train_data, test_data = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"Train set: {len(train_data)} ratings")
        print(f"Test set: {len(test_data)} ratings")
        
        # Build model on training data
        train_matrix = train_data.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Simple prediction using user mean (baseline)
        predictions = []
        actuals = []
        
        for _, row in test_data.head(1000).iterrows():  # Evaluate on subset
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            if user_id in train_matrix.index and movie_id in train_matrix.columns:
                pred_rating = train_matrix.loc[user_id].replace(0, np.nan).mean()
                if not np.isnan(pred_rating):
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"\n✓ RMSE: {rmse:.4f}")
        print(f"✓ MAE: {mae:.4f}")
        
        return rmse, mae
    
    def print_recommendations(self, recommendations, user_id):
        """Pretty print recommendations"""
        print(f"\n{'='*80}")
        print(f"TOP RECOMMENDATIONS FOR USER {user_id}")
        print(f"{'='*80}\n")
        
        for idx, (movie_id, score, title, genres, method) in enumerate(recommendations, 1):
            print(f"{idx}. {title}")
            print(f"   Score: {score:.3f} | Genres: {genres} | Method: {method}")
            print()
def main():
    engine = RecommendationEngine()
    
    # Load data from Kaggle
    # Download from: https://www.kaggle.com/datasets/grouplens/movielens-latest-small
    try:
        engine.load_data('movies.csv', 'ratings.csv')
        engine.preprocess_data()
    except FileNotFoundError:
        print("\n❌ Error: Data files not found!")
        print("Please download the MovieLens dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/grouplens/movielens-latest-small")
        print("\nPlace movies.csv and ratings.csv in the same directory as this script.")
        return
    
    # Select a user
    user_id = 1
    print(f"USER {user_id} PROFILE")
    print(f"{'='*80}")
    profile = engine.get_user_profile(user_id)
    if profile:
        print(f"\nUser has rated {len(profile)} movies:")
        for movie in profile[:5]:  # Show top 5
            print(f"  ⭐ {movie['rating']}/5 - {movie['title']} ({movie['genres']})")
    
    # Generate recommendations using different methods
    
    # 1. User-based Collaborative Filtering
    recs = engine.collaborative_filtering_user_based(user_id, n_recommendations=5)
    engine.print_recommendations(recs, user_id)
    
    # 2. Item-based Collaborative Filtering
    recs = engine.collaborative_filtering_item_based(user_id, n_recommendations=5)
    engine.print_recommendations(recs, user_id)
    
    # 3. Content-Based Filtering
    recs = engine.content_based_filtering(user_id, n_recommendations=5)
    engine.print_recommendations(recs, user_id)
    
    # 4. Matrix Factorization
    recs = engine.matrix_factorization_svd(user_id, n_recommendations=5)
    engine.print_recommendations(recs, user_id)
    
    # 5. Hybrid Approach (BEST)
    recs = engine.hybrid_recommendations(user_id, n_recommendations=10)
    engine.print_recommendations(recs, user_id)
    
    # Evaluate model
    engine.evaluate_model()

if __name__ == "__main__":
    main()