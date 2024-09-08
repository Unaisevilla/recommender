import os

# Paths to data files
RESTAURANT_B_PATH = "data/restaurant_b.csv"
ACTIVITIES_DF_PATH = "data/activities_df.csv"
RESTAURANT_NEW_PATH = "data/restaurant_new.csv"
HISTORY_DF_PATH = "data/history_df.csv"

# Paths to saved models
COLLAB_MODEL_PATH = "models/collab_model.keras"
USER_EMBEDDINGS_PATH = "models/user_embeddings.pkl"

# Hyperparameters
COLLAB_NUM_EPOCHS = 10
COLLAB_BATCH_SIZE = 256
COLLAB_LEARNING_RATE = 0.001
STOPWORDS_LIST = ['english', 'spanish']  # Add your stopwords here
