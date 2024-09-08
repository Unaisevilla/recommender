import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from app.config import RESTAURANT_B_PATH, RESTAURANT_NEW_PATH, HISTORY_DF_PATH, STOPWORDS_LIST

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load data
def load_data():
    with open(HISTORY_DF_PATH, 'r') as f:
        history_df = pd.read_json(f)
    
    with open(RESTAURANT_B_PATH, 'r') as f:
        restaurant_b = pd.read_json(f)
    
    with open(RESTAURANT_NEW_PATH, 'r') as f:
        restaurant_new = pd.read_json(f)

    return restaurant_b, restaurant_new, history_df

# Preprocess data
def preprocess_data(df):
    df['Ingredients'] = df['Ingredients'].str.lower()
    return df

# Initialize recommender
def initialize_recommender():
    global restaurant_b, restaurant_new, history_df, vectorizer, tfidf_matrix_b, tfidf_matrix_new

    # Load and preprocess data
    restaurant_b, restaurant_new, history_df = load_data()
    
    restaurant_b = preprocess_data(restaurant_b)
    restaurant_new = preprocess_data(restaurant_new)

    # Vectorize ingredients
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_LIST)
    
    tfidf_matrix_b = vectorizer.fit_transform(restaurant_b['Ingredients'])
    tfidf_matrix_new = vectorizer.transform(restaurant_new['Ingredients'])

@app.on_event("startup")
async def startup_event():
    initialize_recommender()

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
async def search_recipes(query: Optional[str] = ""):
    global restaurant_b

    if query:
        filtered_df = restaurant_b[restaurant_b['Recipe Name'].str.contains(query, case=False, na=False)]
    else:
        filtered_df = restaurant_b

    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
    return filtered_df[['food_id', 'Recipe Name']].to_dict(orient='records')

@app.get("/history")
async def get_history():
    global history_df, restaurant_b

    history_df_clean = history_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Adding recipe names to the history
    merged_history = history_df_clean.merge(restaurant_b[['food_id', 'Recipe Name']],
                                            on='food_id', how='left')
    return merged_history.to_dict(orient='records')

@app.post("/add_to_history/{food_id}")
async def add_to_history(food_id: int):
    global history_df
    if food_id not in history_df['food_id'].values:
        new_entry = pd.DataFrame([{'user_id': 1, 'food_id': food_id, 'eventType': 1}])
        history_df = pd.concat([history_df, new_entry], ignore_index=True)
        history_df.replace([np.inf, -np.inf], np.nan).dropna().to_json(HISTORY_DF_PATH, orient='records', indent=4)

        initialize_recommender()  # Reinitialize recommender

        merged_history = history_df.merge(restaurant_b[['food_id', 'Recipe Name']],
                                          on='food_id', how='left')
        return merged_history.to_dict(orient='records')
    else:
        raise HTTPException(status_code=400, detail="Food ID already exists in history")

@app.delete("/delete_from_history/{food_id}")
async def delete_from_history(food_id: int):
    global history_df

    # Print out the current columns for debugging purposes
    print(f"History DF Columns before deletion: {history_df.columns.tolist()}")

    # Ensure proper deletion of the specified food_id
    history_df = history_df[history_df['food_id'] != food_id]

    # Add conditional check for empty DataFrame after deletion
    if history_df.empty:
        # Save the updated (now empty) history to JSON
        history_df.replace([np.inf, -np.inf], np.nan).dropna().to_json(HISTORY_DF_PATH, orient='records', indent=4)
        
        # Return an empty list as no items exist
        return []

    # Save updated history to JSON
    history_df.replace([np.inf, -np.inf], np.nan).dropna().to_json(HISTORY_DF_PATH, orient='records', indent=4)

    initialize_recommender()  # Reinitialize recommender

    # Merge with restaurant_b to get Recipe Names
    merged_history = history_df.merge(restaurant_b[['food_id', 'Recipe Name']],
                                      on='food_id', how='left')

    return merged_history.to_dict(orient='records')

@app.get("/recommendations")
async def recommend():
    global history_df, restaurant_b, restaurant_new, vectorizer, tfidf_matrix_b, tfidf_matrix_new

    user_id = 1  # Static user ID for this example

    # Fetch user's historical data
    user_history = history_df[history_df['user_id'] == user_id]

    # Fetch user's rated items
    user_rated_food_ids = user_history['food_id'].tolist()

    # Check if user rated food list is not empty
    if not user_rated_food_ids:
        raise HTTPException(status_code=404, detail="User has no history.")

    # Create user's profile by averaging the TF-IDF vectors of the food items in user's history
    user_item_profiles = tfidf_matrix_b[user_rated_food_ids]
    user_profile = user_item_profiles.mean(axis=0)

    # Ensure user_profile is numpy array
    user_profile = np.asarray(user_profile).flatten()

    # Calculate cosine similarity between user's profile and the TF-IDF vectors of the new restaurant items
    cosine_similarities = cosine_similarity([user_profile], tfidf_matrix_new).flatten()

    # Get indices of the top recommendation
    similar_indices = cosine_similarities.argsort()[-10:][::-1]

    recommendations = restaurant_new.iloc[similar_indices]
    recommendations['recStrength'] = cosine_similarities[similar_indices]

    recommendations_clean = recommendations.replace([np.inf, -np.inf], np.nan).dropna()
    return recommendations_clean[['food_id', 'Recipe Name', 'Ingredients', 'recStrength']].to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)