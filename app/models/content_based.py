from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

class ContentBasedRecommender:
    def __init__(self, history_df, restaurant_new, stopwords_list, ingredients_flat_old, ingredients_flat_new):
        self.history_df = history_df
        self.restaurant_new = restaurant_new
        self.user_item_ids = self.history_df['food_id'].tolist()
        self.new_item_ids = self.restaurant_new['food_id'].tolist()
        self.vectorizer = self._create_vectorizer(stopwords_list)
        self.ingredients_flat_old = ingredients_flat_old
        self.ingredients_flat_new = ingredients_flat_new
        self.user_tfidf_matrix = self._vectorize_user_history()
        self.new_tfidf_matrix = self._vectorize_new_restaurant()

    def _create_vectorizer(self, stopwords_list):
        return TfidfVectorizer(
            analyzer='word',
            stop_words=stopwords_list,
            max_features=None,
            ngram_range=(1, 2),
            min_df=0.001,
            max_df=1.0
        )
    
    def _vectorize_user_history(self):
        return self.vectorizer.fit_transform(self.ingredients_flat_old)

    def _vectorize_new_restaurant(self):
        return self.vectorizer.transform(self.ingredients_flat_new)
    
    def get_profile(self, item_id, is_new):
        if is_new:
            idx = self.new_item_ids.index(item_id)
            return self.new_tfidf_matrix[idx:idx + 1]
        else:
            idx = self.user_item_ids.index(item_id)
            return self.user_tfidf_matrix[idx:idx + 1]
    
    def get_item_profiles(self, ids, is_new):
        item_profiles_list = [self.get_profile(x, is_new) for x in ids]
        return vstack(item_profiles_list)
    
    def recommend(self, user_profile, topn=10):
        cosine_similarities = cosine_similarity(user_profile, self.new_tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = [(self.new_item_ids[i], cosine_similarities[0, i]) for i in similar_indices if i < len(self.new_item_ids)]
        similar_items = sorted(similar_items, key=lambda x: -x[1])
        return similar_items[:topn]