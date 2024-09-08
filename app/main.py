import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import pandas as pd
import numpy as np
from app.config import *
from app.data_processing import load_data, preprocess_data, preprocess_activities, get_ingredients_lists
from app.models.hybrid import HybridRecommender
import random

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    global restaurant_b, activities_df, restaurant_new, history_df
    global hybrid_recommender, ingredients_flat_old, ingredients_flat_new

    restaurant_b, activities_df, restaurant_new, history_df = load_data()
    
    # Preprocess data
    restaurant_b = preprocess_data(restaurant_b)
    restaurant_new = preprocess_data(restaurant_new)
    ingredients_flat_old, ingredients_flat_new = get_ingredients_lists(restaurant_b, restaurant_new)
    
    # Preprocess activities_df to remove duplicates
    activities_df = preprocess_activities(activities_df)
    
    # Instantiate the hybrid recommender
    hybrid_recommender = HybridRecommender(
        history_df,
        restaurant_new,
        activities_df,
        STOPWORDS_LIST,
        COLLAB_MODEL_PATH,
        USER_EMBEDDINGS_PATH,
        COLLAB_NUM_EPOCHS,
        COLLAB_BATCH_SIZE,
        COLLAB_LEARNING_RATE,
        ingredients_flat_old,
        ingredients_flat_new
    )

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
async def search_recipes(query: Optional[str] = ""):
    if query:
        filtered_df = restaurant_b[restaurant_b['Recipe Name'].str.contains(query, case=False, na=False)]
    else:
        filtered_df = restaurant_b
    
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
    return filtered_df[['food_id', 'Recipe Name']].to_dict(orient='records')

@app.get("/history")
async def get_history():
    history_df_clean = history_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Adding recipe names to the history
    merged_history = history_df_clean.merge(restaurant_b[['food_id', 'Recipe Name']],
                                            on='food_id', how='left')
    return merged_history.to_dict(orient='records')

@app.post("/add_to_history/{food_id}")
async def add_to_history(food_id: int):
    global history_df
    if food_id not in history_df['food_id'].values:
        new_entry = pd.DataFrame([{'user_id': 1, 'food_id': food_id, 'eventType': random.randint(1, 2)}])
        history_df = pd.concat([history_df, new_entry], ignore_index=True)
        history_df.replace([np.inf, -np.inf], np.nan).dropna().to_csv(HISTORY_DF_PATH, index=False)
        merged_history = history_df.merge(restaurant_b[['food_id', 'Recipe Name']],
                                          on='food_id', how='left')
        return merged_history.to_dict(orient='records')
    else:
        raise HTTPException(status_code=400, detail="Food ID already exists in history")

@app.delete("/delete_from_history/{food_id}")
async def delete_from_history(food_id: int):
    global history_df
    # Ensure proper deletion of the specified food_id
    history_df = history_df[history_df['food_id'] != food_id]
    
    # Save updated history to CSV
    history_df.replace([np.inf, -np.inf], np.nan).dropna().to_csv(HISTORY_DF_PATH, index=False)
    
    # Merge with restaurant_b to get Recipe Names
    merged_history = history_df.merge(restaurant_b[['food_id', 'Recipe Name']],
                                      on='food_id', how='left')
    
    return merged_history.to_dict(orient='records')

@app.get("/recommendations")
async def recommend():
    try:
        user_id = 1  # You can modify this to get dynamic user_id
        recommendations = hybrid_recommender.recommend_items(user_id=user_id, topn=10, verbose=True)
        recommendations_clean = recommendations.replace([np.inf, -np.inf], np.nan).dropna()
        return recommendations_clean.to_dict(orient='records')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)