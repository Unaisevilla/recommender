import os

# Paths to data files
RESTAURANT_B_PATH = 'data/restaurant_b.json'
RESTAURANT_NEW_PATH = 'data/restaurant_new.json'
HISTORY_DF_PATH = 'data/history_df.json'

# Paths to saved models
COLLAB_MODEL_PATH = "models/collab_model.keras"
USER_EMBEDDINGS_PATH = "models/user_embeddings.pkl"

# Hyperparameters
COLLAB_NUM_EPOCHS = 10
COLLAB_BATCH_SIZE = 256
COLLAB_LEARNING_RATE = 0.001
STOPWORDS_LIST = ['english', 'spanish']  # Add your stopwords here
