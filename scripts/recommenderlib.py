import pickle
import requests
from PIL import Image
from io import BytesIO
import os,sys

import time as time 
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# function
def load_model(model_name):
    if model_name == 'NearestNeighbor':
        with open('./model_neighbors.pkl', 'rb') as file:
            model = pickle.load(file)
            return model
    elif model_name == 'NMF':
            with open('./model_nmf.pkl', 'rb') as file:
                model = pickle.load(file)
                return model
    else:
            print('The model has not been listed')
            print_test()

def print_test():
     print('It is time for prayer')

def image_merger(image_paths, show_type):
    """ It gets in paths of images and merge the images vertically (ver) or horizontally (hor)"""
    # image_paths is a list of paths of images
    # show_type is direction to show the merge image

    # Open images and store them in a list
    images = [Image.open(img_path) for img_path in image_paths]

    # Determine the width and height for the final merged image
    # For horizontal merge
    # calculating the total width width and maximum hight of all images
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # For vertical merge
    # calculating the total hights width and maximum wdith of all images
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # Create a new blank image for horizontal merge
    merged_image_horizontal = Image.new('RGB', (total_width, max_height))

    # Create a new blank image for vertical merge
    merged_image_vertical = Image.new('RGB', (max_width, total_height))

    # Paste images side by side for horizontal merge
    x_offset = 0
    for img in images:
        merged_image_horizontal.paste(img, (x_offset, 0))
        x_offset += img.width

    # Paste images on top of each other for vertical merge
    y_offset = 0
    for img in images:
        merged_image_vertical.paste(img, (0, y_offset))
        y_offset += img.height

# Save the merged images
    merged_image_horizontal.save('./images/merged_image_horizontal.jpg')
    merged_image_vertical.save('./images/merged_image_vertical.jpg')
    
    if show_type == 'hor':
        return merged_image_horizontal.show()
    else:
        return merged_image_vertical.show()
# --------------------------------- Getting teh movie data ----------------
def get_movie_data(movie_tmbd, api_key, save_directory='./images'):
        # save_directory='./images'
        """
        Function to get movie poster's URLs and save the images locally.
        
        Parameters:
        - movie_tmbd: list of int, a list of TMDB movie IDs.
        - api_key: str, TMDB API key.
        - save_directory: str, path to the directory where images will be saved (default is './images').

        Returns:
        - list_poster_url: list of str, list of poster URLs.
        """
        # movies_data = []
        # Ensure the directory for saving images exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # image_paths = []
        # for movie_tmbd in movie_tmbds:
            # print(f"Fetching poster for movie ID: {movie_tmbd}")
        base_url = f'https://api.themoviedb.org/3/movie/{movie_tmbd}'

        # Send a GET request to TMDB API
        response = requests.get(base_url, params={'api_key': api_key})
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}'
                    # list_poster_url.append(poster_url)

                    # Send a GET request to the image URL
                response_image = requests.get(poster_url)
                image = Image.open(BytesIO(response_image.content))

                # Save the image locally
                image_path = os.path.join(save_directory, f"poster_tmbd_{movie_tmbd}.jpg")
                image.save(image_path)
                    # image_paths.append(image_path)
                    # appending the data of the movie into the data list
                # movies_data.append(data)
                    # print(f"Saved poster for movie ID {movie_tmbd} at {image_path}")
            else:
                    print(f"Poster not found for movie ID {movie_tmbd}.")
        else:
            print(f"Failed to fetch movie details for ID {movie_tmbd}. Status code: {response.status_code}")

        return image_path, data


# -------------------------------- Getting the poster Paths ----------------
def get_movies_data(movie_tmbds, api_key, save_directory='./images'):
        # save_directory='./images'
        """
        Function to get movie posters' URLs and save the images locally.
        
        Parameters:
        - movie_tmbds: list of int, a list of TMDB movie IDs.
        - api_key: str, TMDB API key.
        - save_directory: str, path to the directory where images will be saved (default is './images').

        Returns:
        - list_poster_url: list of str, list of poster URLs.
        """
        movies_data = []
        # Ensure the directory for saving images exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        image_paths = []
        for movie_tmbd in movie_tmbds:
            # print(f"Fetching poster for movie ID: {movie_tmbd}")
            base_url = f'https://api.themoviedb.org/3/movie/{movie_tmbd}'

            # Send a GET request to TMDB API
            response = requests.get(base_url, params={'api_key': api_key})
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}'
                    # list_poster_url.append(poster_url)

                    # Send a GET request to the image URL
                    response_image = requests.get(poster_url)
                    image = Image.open(BytesIO(response_image.content))

                    # Save the image locally
                    image_path = os.path.join(save_directory, f"poster_tmbd_{movie_tmbd}.jpg")
                    image.save(image_path)
                    image_paths.append(image_path)
                    # appending the data of the movie into the data list
                    movies_data.append(data)
                    # print(f"Saved poster for movie ID {movie_tmbd} at {image_path}")
                else:
                    print(f"Poster not found for movie ID {movie_tmbd}.")
            else:
                print(f"Failed to fetch movie details for ID {movie_tmbd}. Status code: {response.status_code}")

        return image_paths, movies_data
      


#--------------------------------- Main Function -----------------------
def recommender_nearest_neghibor(query, model_name, df_ratings, df_movies, k):
        """ This function takes the query of new user and model trained and generates the poster of 
            recommended movies
            query: the rating info of the new user that is to be obtained in th stremlit app
            model: a pretarined model saved in the model directory of the main repo
            df_ratings: the table on the available ratings 
            df_movies: the table on the movies
            k: number of recommended movies"""
         # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
     # where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
        R = csr_matrix((df_ratings["rating"], (df_ratings["userId"], df_ratings["movieId"])))
    
        df_r = pd.DataFrame(R.todense())
    # making a datafrom from the query of new user
        # making a datafrom from the query of new user

        df_new_user = pd.DataFrame(query, columns=df_movies["movieId"], index=["new_user"])
    
        df_new_user_filled = df_new_user.fillna(0)
        # call in the model_load
        model = load_model(model_name)
        # Calculate the distances to all other users in the data!
        similarity_scores, neighbor_ids = model.kneighbors(
            df_new_user_filled,
            n_neighbors=5,
            return_distance=True,)

        # sklearn returns a list of predictions
        # extract the first and only value of the list
        df_neighbors = pd.DataFrame(
            data={
                "neighbor_id": neighbor_ids[0],
                "similarity_score": similarity_scores[0],
            }
        )

        df_neighbors.sort_values("similarity_score", ascending=False, inplace=True)

        # Look at ratings for 5 users that are similar
        neighborhood = df_r.iloc[neighbor_ids[0]]

        # Filter out seen movies
        neighborhood_filtered = neighborhood.drop(query.keys(), axis=1)

        # Calculate the Ratings for similar users based on similarity scores
        # This process will generate the new set of ratings based on the similarity with the refrence user (new user)
        df_get_score = df_neighbors.set_index("neighbor_id")

        # Multiply the ratings with the similarity score of each user and
        # calculate the summed up rating for each movie

        df_score = neighborhood_filtered.apply(
            lambda x: df_get_score.loc[x.index]["similarity_score"] * x
        )

        # Ranking teh movies based on the calcualted scores based on quantified similarity
        df_score_ranked = df_score.sum(axis=0).reset_index().sort_values(0, ascending=False)
        df_score_ranked.columns = ["movieId", "score"]
        df_score_ranked.reset_index(drop=True, inplace=True)

        # making a dataframe for recommendations
        recommendations = df_movies[df_movies["movieId"].isin(df_score_ranked.iloc[:k]["movieId"])]


        # # getting the poster images and saving them in a directory 
        api_key = '32963fd453f575aa44262db989d926d6'
        # image_paths = get_movie_posters(df_movies.tmdbId.loc[recommendations.movieId], api_key)

        # return recommendations, image_merger(image_paths, 'hor')
        # movie_tmbds = df_movies.tmdbId.loc[recommendations.movieId]
        # image_paths, movies_data = get_movies_data(movie_tmbds, api_key, save_directory='./images')
        # merging the images of movies

        # image_merger(image_paths, show_type = 'hor')
        return recommendations

#------------------------------ NMF Recommender ----------------------
def recommender_nmf(query, model_name, df_ratings, df_movies, k):

    # Create user vector
    df_new_user = pd.DataFrame(query, columns=df_movies["movieId"], index=["new_user"])
    df_new_user_modified = df_new_user.fillna(0)

    # call in the model_load
    model = load_model(model_name)

    # Create user-feature matrix P for new user
    P_new_user_matrix = model.transform(df_new_user_modified)

    # New dataframe 
    P_new_user = pd.DataFrame(
        P_new_user_matrix,
        columns=model.get_feature_names_out(),
        index=["new_user"],
    )

    # Reconstruct user-movie matrix/dataframe for new user
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix)
    R_hat_new_user_matrix = np.dot(P_new_user, Q)
    R_hat_new_user = pd.DataFrame(R_hat_new_user_matrix, index=["new_user"])

    ranked = R_hat_new_user.T.sort_values("new_user", ascending=False)
    recommended = ranked[~ranked.index.isin(query)].reset_index()
    recommended.columns = ["movieid", "score"]

    # Get movie ids and corresponding titles the same order
    movie_ids = recommended.iloc[:k]["movieid"]
    # titles = [df_movies.loc[id]["title"] for id in movie_ids]
    
    recommendations = df_movies[df_movies.movieId.isin(movie_ids)]

    # getting the poster images and saving them in a directory 
    # api_key = '32963fd453f575aa44262db989d926d6'
    # image_paths, movies_data = get_movies_data(df_movies.tmdbId.loc[recommendations.movieId], api_key)
    
    return recommendations