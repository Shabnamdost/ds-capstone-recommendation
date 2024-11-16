"Recommender System"

import pickle 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import os
print(os.getcwd())

def build_model_nmf(n_components: int = 2000, max_iter: int = 1000) -> str: 
    
    # Load prepared data
    ratings = pd.read_csv(r'C:\Users\shabn\neuefische\ds-capstone-recommendation\data\ml-latest-small\ratings_modified.csv')

    # Initialize sparse user-item matrix
    R = csr_matrix(
        (ratings["rating"], (ratings["user_id"], ratings["movie_id"]))
    )

    # Instantiate model and fit
    model =NMF(n_components=n_components, max_iter=max_iter)
    print(
        "NMF model instantiated with following hyperparameters:\n"
        f"n_components={n_components}\n"
        f"max_iter={max_iter}\n"
        "Starting to fit.\n"
    )
    
    # Fit it to the ratings matrix
    model.fit(R)

    # Print reconstruction error
    print(f"NMF model built. Reconstruciton error: {model.reconstruction_err_}")

    # Save model 
    file_name = "data/model_nmf.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)
    
    return file_name


def recommended_movies(query, model, ratings, movies, k=5):
    # Ensure movie_ids from the model match the ratings DataFrame columns
    movie_ids = ratings.columns
    
    # Create a user vector with 0s for all movies the user hasn't rated
    new_user_row = pd.Series(0, index=movie_ids)
    
    # Fill in the ratings from the query
    for movie_id, rating in query.items():
        if movie_id in movie_ids:
            new_user_row[movie_id] = rating

    # Reshape the row to create a single-row matrix for the model
    new_user_matrix = new_user_row.values.reshape(1, -1)

    # Transform the user vector using the NMF model to get the user-feature matrix P
    P_new_user_matrix = model.transform(new_user_matrix)

    # Reconstruct the user-movie matrix for the new user
    Q_matrix = model.components_
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)

    # Create a DataFrame for predicted scores
    predicted_scores = pd.DataFrame(R_hat_new_user_matrix, columns=movie_ids, index=["new_user"])

    # Rank movies by predicted scores
    ranked = predicted_scores.T.sort_values("new_user", ascending=False)

    # Remove movies that the user has already rated
    ranked = ranked[~ranked.index.isin(query.keys())]

    # Get top-k recommendations
    recommendations = ranked.head(k).reset_index()
    recommendations.columns = ["movie_id", "score"]

    # Merge with movie titles
    recommendations = recommendations.merge(movies, on="movie_id")

    return recommendations[["movie_id", "title", "score"]]


### space for Nearest Neighbor

def main() -> None:
    file_name_nmf = build_model_nmf()
    print(f"NMF model saved to {file_name_nmf}.")

if __name__ == "__main__":
    main()

    