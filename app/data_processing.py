import pandas as pd
from app.config import *

def load_data():
    restaurant_b = pd.read_json(RESTAURANT_B_PATH)
    restaurant_new = pd.read_json(RESTAURANT_NEW_PATH)
    history_df = pd.read_json(HISTORY_DF_PATH)
    return restaurant_b, restaurant_new, history_df

def process_ingredients(ingredients_str):
    ingredients_list = ingredients_str.split(',')
    ingredients_list = [ingredient.strip() for ingredient in ingredients_list]
    ingredients_list = list(set(ingredients_list))  # Remove duplicates
    return ' '.join(ingredients_list)

def preprocess_data(df, column='Ingredients'):
    df['Processed_Ingredients'] = df[column].apply(process_ingredients)
    return df

def get_ingredients_lists(restaurant_b, restaurant_new):
    ingredients_flat_old = restaurant_b['Processed_Ingredients'].tolist()
    ingredients_flat_new = restaurant_new['Processed_Ingredients'].tolist()
    return ingredients_flat_old, ingredients_flat_new

def preprocess_activities(activities_df):
    # Aggregate duplicates by summing the eventType values
    activities_df = activities_df.groupby(['user_id', 'food_id'], as_index=False).sum()
    return activities_df