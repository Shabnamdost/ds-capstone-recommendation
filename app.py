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


### space for Nearest Neighbor

def main() -> None:
    file_name_nmf = build_model_nmf()
    print(f"NMF model saved to {file_name_nmf}.")

if __name__ == "__main__":
    main()

    