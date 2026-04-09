import pickle
import pandas as pd
import streamlit as st
import gzip # <-- Thêm dòng này
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. LOAD MODELS FUNCTION (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    # SVD Model (Giữ nguyên)
    with open('models/recommender_model_v1.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    
    # TFIDF Vectorizer (Giữ nguyên)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    # Matrix (DÙNG GZIP ĐỂ MỞ FILE NÉN)
    with gzip.open('models/tfidf_matrix.pkl.gz', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Metadata (Pandas tự động giải nén)
    df_metadata = pd.read_pickle('models/df_metadata.pkl.gz')
    
    return svd_model, tfidf, tfidf_matrix, df_metadata

# ==========================================
# 2. INGREDIENT-BASED PREDICTION (CONTENT-BASED)
# ==========================================
def get_content_based_recommendations(selected_ingredients, tfidf, tfidf_matrix, df_metadata, top_n=5):
    # Join the selected ingredients into a single string
    ingredients_string = " ".join(selected_ingredients)
    
    # Transform text into vector & compute Cosine Similarity
    input_vec = tfidf.transform([ingredients_string])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Get the indices of the Top N recipes with the highest scores
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        score = sim_scores[idx]
        if score > 0: # Only include recipes that actually match
            recipe = df_metadata.iloc[idx]
            results.append({
                'name': recipe['name'],
                'score': score,
                'minutes': recipe['minutes'],
                'ingredients': recipe['ingredients'] if isinstance(recipe['ingredients'], list) else [],
                'steps': recipe['steps']
            })
    return results

# ==========================================
# 3. USER ID PREDICTION (SVD)
# ==========================================
def get_svd_recommendations(user_id, svd_model, df_metadata, top_n=5):
    # Sample 500 recipes for faster prediction (prevents web app from freezing)
    sample_recipes = df_metadata.sample(500, random_state=42)
    
    predictions = []
    for _, row in sample_recipes.iterrows():
        recipe_id = row['recipe_id']
        # Predict the star rating this user would give to this recipe
        pred = svd_model.predict(uid=user_id, iid=recipe_id)
        predictions.append({
            'recipe_id': recipe_id,
            'predicted_rating': pred.est,
            'name': row['name'],
            'minutes': row['minutes']
        })
    
    # Sort predictions from highest to lowest rating
    top_predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)[:top_n]
    return top_predictions