import pandas as pd
import numpy as np
from app.models.content_based import ContentBasedRecommender
from app.models.collaborative import CollaborativeRecommender
from sklearn.preprocessing import normalize

class HybridRecommender:
    def __init__(self, history_df, restaurant_new, activities_df, stopwords_list, model_path, embeddings_path, num_epochs, batch_size, learning_rate, ingredients_flat_old, ingredients_flat_new):
        self.content_recommender = ContentBasedRecommender(history_df, restaurant_new, stopwords_list, ingredients_flat_old, ingredients_flat_new)
        self.collaborative_recommender = CollaborativeRecommender(activities_df, self.content_recommender.new_item_ids, model_path, embeddings_path, num_epochs, batch_size, learning_rate)
        self.history_df = history_df
        self.activities_df = activities_df
        self.user_profiles = self._build_users_profiles()

    def _build_users_profiles(self):
        interactions_indexed_df = self.history_df[self.history_df['food_id'].isin(self.content_recommender.user_item_ids)].set_index('user_id')
        user_profiles = {user_id: self._build_user_profile(user_id, interactions_indexed_df) for user_id in interactions_indexed_df.index.unique()}
        return user_profiles
    
    def _build_user_profile(self, user_id, interactions_indexed_df):
        interactions_user_selected = interactions_indexed_df.loc[user_id]
        user_item_profiles = self.content_recommender.get_item_profiles(interactions_user_selected['food_id'], is_new=False)
        
        user_item_strengths = np.array(interactions_user_selected['eventType']).reshape(-1, 1)
        total_strengths = np.sum(user_item_strengths)
        if total_strengths == 0:
            return np.zeros(user_item_profiles.shape[1])
        
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / total_strengths
        user_profile_norm = normalize(np.asarray(user_item_strengths_weighted_avg))
        return user_profile_norm

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        user_profile = self.user_profiles.get(user_id)
        if user_profile is None:
            raise ValueError(f"User ID {user_id} not found.")
        
        num_orders = self.activities_df[self.activities_df['user_id'] == user_id].shape[0]
        
        if num_orders < 5:
            similar_items = self.content_recommender.recommend(user_profile, topn)
        else:
            similar_items_content = self.content_recommender.recommend(user_profile, topn)
            similar_items_collab = self.collaborative_recommender.recommend(user_id, topn)
            combined_similarities = [
                (simc[0], 0.7 * simc[1] + 0.3 * simw[1])
                for simc, simw in zip(similar_items_collab, similar_items_content)
            ]
            similar_items = combined_similarities
        
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['food_id', 'recStrength']).head(topn)

        if verbose:
            recommendations_df = recommendations_df.merge(
                self.content_recommender.restaurant_new, how='left', left_on='food_id', right_on='food_id')[['recStrength', 'food_id', 'Recipe Name', 'Ingredients']]

        return recommendations_df