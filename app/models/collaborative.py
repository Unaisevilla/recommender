from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint
import os
import pickle
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, activities_df, new_item_ids, model_path, embeddings_path, num_epochs, batch_size, learning_rate):
        self.activities_df = activities_df
        self.new_item_ids = new_item_ids
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if os.path.exists(model_path) and os.path.exists(embeddings_path):
            self.model, self.user_embeddings = self._load_model()
        else:
            self.model, self.user_embeddings = self._build_collaborative_model()

    def _build_collaborative_model(self):
        interactions_matrix = self.activities_df.pivot(index='user_id', columns='food_id', values='eventType').fillna(0)
        interactions_matrix = interactions_matrix.reindex(columns=self.new_item_ids, fill_value=0)
        
        def build_autoencoder(input_dim):
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(64, activation='relu')(input_layer)
            encoder = Dense(32, activation='relu')(encoder)
            decoder = Dense(64, activation='relu')(encoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=MeanSquaredError())
            return autoencoder

        input_dim = interactions_matrix.shape[1]
        autoencoder = build_autoencoder(input_dim)

        # Save the best model during training
        checkpoint = ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min')
        
        autoencoder.fit(interactions_matrix.values, interactions_matrix.values, epochs=self.num_epochs, batch_size=self.batch_size, shuffle=True, validation_split=0.2, callbacks=[checkpoint])
        
        # Load the best model
        autoencoder = load_model(self.model_path)
        
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
        user_embeddings = encoder.predict(interactions_matrix.values)
        
        # Save embeddings
        with open(self.embeddings_path, 'wb') as file:
            pickle.dump(user_embeddings, file)
        
        return autoencoder, user_embeddings
    
    def _load_model(self):
        autoencoder = load_model(self.model_path)
        
        with open(self.embeddings_path, 'rb') as file:
            user_embeddings = pickle.load(file)
        
        return autoencoder, user_embeddings
    
    def recommend(self, user_id, topn=10):
        user_embedding = self.user_embeddings[user_id - 1].reshape(1, -1)
        cosine_similarities = cosine_similarity(user_embedding, self.user_embeddings)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = [(self.new_item_ids[i], cosine_similarities[0, i]) for i in similar_indices if i < len(self.new_item_ids)]
        similar_items = sorted(similar_items, key=lambda x: -x[1])
        return similar_items[:topn]