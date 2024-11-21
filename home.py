# importing the libraries for the data processing part

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


import streamlit as st
import matplotlib.pyplot as plt
# importing the recommender library of functions
from scripts.recommenderlib import *

from streamlit_carousel import carousel

# use ast.literal_eval() to safely convert a string list to a list.
import ast

#------------------ Background Image --------
def background_image(backgroundimageurl):
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url({backgroundimageurl});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def custom_title(text, size=50, color="black"):
    st.markdown(
        f"<h1 style='font-size: {size}px; color: {color};'>{text}</h1>",
        unsafe_allow_html=True
    )
# ------------------- Loading Data Set -------------

def load_movies_dataset():
    """ This function loads data sets """
    # # -------------------- Loading Data Set

# # first we read our data set from directroy
    # df_ratings = pd.read_csv('./data/ml-latest-small/ratings_cleaned.csv')
    df_movies = pd.read_csv('./data/ml-latest-small/movies_cleaned.csv')

    return df_movies

def load_ratings_dataset():
    """ This function loads rsating data set """
    # # -------------------- Loading Data Set

# # first we read our data set from directroy
    df_ratings = pd.read_csv('./data/ml-latest-small/ratings_cleaned.csv')
    # df_movies = pd.read_csv('./data/ml-latest-small/movies_cleaned.csv')

    return df_ratings


def basic_image_carousel_with_buttons(images):
    # List of image paths or URLs
    

    # Session state to track the current image index
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0

    # Function to navigate images
    def next_image():
        st.session_state.current_image = (st.session_state.current_image + 1) % len(images)

    def previous_image():
        st.session_state.current_image = (st.session_state.current_image - 1) % len(images)

    # Display the current image
    st.image(images[st.session_state.current_image], width=500, caption=f"Image {st.session_state.current_image + 1}")

    # Buttons to navigate
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous"):
            previous_image()
    with col2:
        if st.button("Next"):
            next_image()
# ------------ Autoplay Carousal ---------------------- 
def autoplay_carousal_images(images):
    # Session state to track the current image index
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0

    # Display the current image
    st.image(images[st.session_state.current_image], width=300, caption=f"Image {st.session_state.current_image + 1}")

    # Auto-update every 3 seconds
    time.sleep(1.5)  # Delay for autoplay
    st.session_state.current_image = (st.session_state.current_image + 1) % len(images)  # Update to next image

    # Rerun the app
    st.rerun()

#-------------------------------------------------
# --------------- AutoPlay Carousal ------------------------
def autoplay_carousal(images):
    """ This function generates an autoplay carousal which displays posters of movies
     sequentially """
    # Initialize session state
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    # Control autoplay with a checkbox
    start_stop = st.checkbox("Autoplay Carousel", value=st.session_state.is_playing)
    st.session_state.is_playing = start_stop

    # Display the current image
    st.image(images[st.session_state.current_image], width=500, caption=f"Image {st.session_state.current_image + 1}")

    # If autoplay is enabled, update the image index in a loop
    if st.session_state.is_playing:
        time.sleep(1)  # Wait for 3 seconds
        st.session_state.current_image = (st.session_state.current_image + 1) % len(images)
        st.rerun()
# -------------------Function to get info of Movies----------------------

def get_movie_info(movie_tmdbId, api_key = '32963fd453f575aa44262db989d926d6'):
    
    df_movies = load_movies_dataset()

    image_path, movie_data = get_movie_data(movie_tmdbId, api_key, save_directory='./images')
   
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_path, caption=" ")
    
    with col2:

        # movie_data = movies_data[0]
        # now aggregating all the genres and place them in a list
        
        df_movies_select = df_movies.set_index('tmdbId').loc[int(movie_tmdbId)]
        # st.write( )
        # genres_ = []
        genres_ = ast.literal_eval(df_movies_select.genres)
        
        # for i in range(len( movie_data['genres'])):
        #     # st.write(movie_data.get('title'), movie_tmdbId, movie_data['genres'][i]['name'])
        #     genres_.append( movie_data['genres'][i]['name'] )


        genres_p = ''
        for genre in genres_:
                     genres_p = genres_p + genre +', '

        # Title    
        st.markdown(
                    f"""
                            <span style=" font-size:20px; color: yellow;"> Title: </span> <span style="font-size:20px; color: white;"> {movie_data.get('title')}
                    </p>
                    """, unsafe_allow_html=True)

        # Genres
        st.markdown(
                    f"""
                            <span style=" font-size:20px; color: yellow;"> Genres: </span> <span style="font-size:20px; color: white;"> {genres_p[:-2]}
                    </p>
                    """, unsafe_allow_html=True)
        # Relase Data
        st.markdown(
                    f"""
                            <span style=" font-size:20px; color: yellow;"> Release Date: </span> <span style="font-size:20px; color: white;"> {movie_data.get('release_date')}
                    </p>
                    """, unsafe_allow_html=True)
            # Overview
        st.markdown(
                    f"""
                            <span style=" font-size:20px; color: yellow;"> Overview: </span> <span style="font-size:20px; color: white;"> {movie_data.get('overview')}
                    </p>
                    """, unsafe_allow_html=True)
    #------------------------------------------------------------------------------------
def reorder_dataframe_by_genres(df, genres_to_prioritize):
    """
    Reorders a DataFrame based on specified genres, moving rows containing the genres to the top.

    Parameters:
    - df (pd.DataFrame): The original DataFrame containing a 'genres' column.
    - genres_to_prioritize (list): List of genres to prioritize (e.g., ['Action', 'Drama']).

    Returns:
    - pd.DataFrame: A reordered DataFrame with prioritized genres at the top.
    """
    df = df.reset_index()
    # Make a copy of the DataFrame to avoid modifying the original
    df_ordered = df.copy()

    # Initialize indices for ordered and remaining rows
    prioritized_indices = []
    other_indices = []

    # Iterate through rows to determine which belong to the prioritized genres
    for idx, row in df.iterrows():
        genres = ast.literal_eval(row['genres'])  # Parse the 'genres' string into a list
        if any(genre in genres for genre in genres_to_prioritize):
            prioritized_indices.append(idx)
        else:
            other_indices.append(idx)

    # Create the reordered DataFrame
    df_prioritized = df.loc[prioritized_indices]
    df_remaining = df.loc[other_indices]
    df_reordered = pd.concat([df_prioritized, df_remaining], axis=0).reset_index(drop=True)

    return df_reordered

#-----------------------------------------R
def reorder_dataframe_by_genres_v2(df, genres):
    """
    Reorders a DataFrame by placing rows with specified genres at the top.

    Parameters:
    - df (pd.DataFrame): The input DataFrame. Must have a 'genres' column with list-like values.
    - genres (list): A list of genres to match.

    Returns:
    - pd.DataFrame: The reordered DataFrame.
    """
    # Reset index for consistent row operations
    df = df.reset_index(drop=True)
    df_ordered = df.copy()  # Copy to reorder rows
    index_ = []  # Stores indices of rows matching genres

    j = 0  # Tracks placement of matched rows
    index = [i for i in range(len(df.genres))]  # Original index list

    # Iterate over the rows to find matches
    for i in index:
        result = ast.literal_eval(df.genres[i])  # Convert genre string to list
        if all(genre in result for genre in genres):  # Check if all genres are in the result
            index_.append(i)
            df_ordered.iloc[j, :] = df.iloc[i, :]  # Place matching row at the top
            j += 1

    # Remaining rows to drop from the ordered DataFrame
    index_2 = index[j:]
    df2 = df.drop(index_, axis=0)  # Rows not matching the genres
    df_ordered = df_ordered.drop(index_2, axis=0)  # Drop extra rows from ordered DataFrame

    # Concatenate matched rows with the remaining rows
    df_reordered = pd.concat([df_ordered, df2], axis=0).reset_index(drop=True)
    return df_reordered


# -------------------- 
def display_movie_sidebar(image_path, movie_title, i):
    """
    Display a movie's image, title, and rating slider in the Streamlit sidebar.

    Args:
    - image_path (str): Path to the movie's image.
    - movie_title (str): Title of the movie.
    - movie_index (int): Index of the movie for unique slider labeling.

    Returns:
    - int: The rating value selected for the movie.
    """

    # get_movies_data(movie_tmbds, api_key, save_directory='./images')

    # Display the movie image
    image = Image.open(image_path)
    st.sidebar.image(image, use_container_width=False)

    # Display the movie title with styling
    # st.sidebar.markdown(
    #     f"""
    #     <p style="font-size:18px; color:black;">
    #         Rate Movie: <b>{movie_title}</b>
    #     </p>
    #     """,
    #     unsafe_allow_html=True
    # )

    # Display the slider for rating
    rating = st.sidebar.slider(
        label=f"Rating for {movie_title}",  # Unique label
        min_value=0,
        max_value=5,
        value=0,  # Default value
        step=1, key =f"kbar{i}"
    )
    
    
    return rating

# ---------- End of function definition ------------------

# --------------Main Body of the STreamlit Code -------------

url1 = "data:image/jpeg;base64,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"


# Add CSS to set the background image
st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://th.bing.com/th?id=OIP.uScj9ImAWCZw9jNYPpr0EwHaEK&w=333&h=187&c=8&rs=1&qlt=90&o=6&dpr=1.5&pid=3.1&rm=2");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://th.bing.com/th?id=OIP.uScj9ImAWCZw9jNYPpr0EwHaEK&w=333&h=187&c=8&rs=1&qlt=90&o=6&dpr=1.5&pid=3.1&rm=2");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# background_image(url)


st.markdown("<h1 style='text-align: center; font-size:60px; color: white;'> PickWise </h1>", unsafe_allow_html=True)

# Content of your Streamlit app
st.markdown("<h1 style='text-align: center; font-size:40px; color: white;'> Your Ultimate Movie Companion </h1>", unsafe_allow_html=True)
# custom_title(" Make the Decision Easier and Entertain More.", size = 40, color = 'white')
# Making the Decision Easier and Enjoy More.


st.image('.\images\complete_poster_3.jpeg', width=800)

images = [
        "./images\poster_tmbd_197.0.jpg",
        "./images\poster_tmbd_9331.0.jpg",
        "./images\poster_tmbd_36955.0.jpg",
        "./images\poster_tmbd_5503.0.jpg"
    ]

# ---------- TESTing Section--------------





# -------------------------------------------
# autoplay carousal
# autoplay_carousal(images)


# ------------ Loading Data for Movie Selction and REcommendration 
df_movies = load_movies_dataset()
df_ratings = load_ratings_dataset()
# Options for the selection box
st.markdown(
    f"""
    <p style="font-size:25px; color:white; font-weight: bold;">
        Please Select the Genres:
    </p>
    """,
    unsafe_allow_html=True
    )


# mutilselect box to select favorire genres

genres_options = ['Drama', 'Comedy', 'Action', 'Thriller', 'Animation']

selected_movies_genres = st.multiselect("", genres_options, )

# We have to make a list of movies that have rating of at least one
rated_movie_list = df_ratings.movieId.unique().tolist()

# st.write(len(rate_movie_list))
df_movies_rated =  df_movies[df_movies.movieId.isin(rated_movie_list)]
# reseting the index 
df_movies_rated = df_movies_rated.reset_index()

# st.write(df_movies)
# st.write(df_movies_rated)
# make a list of selected genres 
movies_options_p =[]
for i in range(len(df_movies_rated.genres)):
    if len(selected_movies_genres) == 1:
        result = ast.literal_eval(df_movies_rated.loc[i, 'genres'])
        if (selected_movies_genres[0] in result):
            movies_options_p.append(df_movies_rated.loc[i, 'title'])
    if len(selected_movies_genres) == 2:
        result = ast.literal_eval(df_movies_rated.loc[i, 'genres'])
        if (selected_movies_genres[0] in result) and (selected_movies_genres[1] in result) :
            movies_options_p.append(df_movies_rated.loc[i, 'title'])
    if len(selected_movies_genres) == 3:
        result = ast.literal_eval(df_movies_rated.loc[i, 'genres'])
        if (selected_movies_genres[0] in result) and (selected_movies_genres[1] in result) and (selected_movies_genres[2] in result) :
            movies_options_p.append(df_movies_rated.loc[i, 'title'])

st.markdown(
    f"""
    <p style="font-size:18px; color:white;">
        Number of Movie Options:  {len(movies_options_p)}:
    </p>
    """,
    unsafe_allow_html=True
    )


# it is only for test
movies_options = df_movies.title.tolist()[:10]
st.markdown(
    f"""
    <p style="font-size:25px; color:white; font-weight: bold;">
        Please Select at least 5 Movies you watched:
    </p>
    """,
    unsafe_allow_html=True
    )


# Multiple selection box with default values
selected_movies_titles = st.multiselect("", movies_options_p)

selected_movies_df = df_movies[df_movies.title.isin(selected_movies_titles)]
selected_moviesId = selected_movies_df['movieId'].tolist()
selected_tmdbIds = df_movies[df_movies.movieId.isin(selected_moviesId)]['tmdbId'].tolist()

# here we get teh dat on teh selected movie 
api_key = '32963fd453f575aa44262db989d926d6'
image_paths, movies_data = get_movies_data(selected_tmdbIds, api_key, save_directory='./images')





# get_movie_data(movie_tmbd, api_key, save_directory='./images'):

# st.write()

# # Initialize session state for sliders
# if "slider_values" not in st.session_state:
#     st.session_state.slider_values = {
#         f"slider_{i}": 50 for i in range(1,6)  # Default value for 5 sliders
#     }

# # Function to display sliders
# def create_sidebar_sliders():
#     # st.write(k)
#     with st.sidebar:
#         st.title("Multiple Sliders")
#         for i in range(1, 6):  # Adjust range for the number of sliders
#             st.session_state.slider_values[f"slider_{i}"] = st.slider(
#                 f"Rate Movie {i}",
#                 0,
#                 100,
#                 st.session_state.slider_values[f"slider_{i}"],  # Default to session state value
#                 key=f"slider_{i}",
#             )

# # Call the function to display sliders
# create_sidebar_sliders()

# # Display slider values in the main app
# st.write("Current Slider Values:")
# st.write(st.session_state.slider_values)
# # st.stop()


# # ----------------------------------------------------------------------
# dict_rates = {}
# if "slider_values" not in st.session_state:
#     st.session_state.slider_values = {
#         f"slider_{i}": 0 for i in range(len(selected_movies_titles))  # Default value for 5 sliders
#     }


# # Function to display sliders
# def create_sidebar_sliders(selected_movies_titles, selected_moviesId):
#     with st.sidebar:
#         # st.title("Multiple Sliders")
#         for i in range(len(selected_movies_titles)):  # Adjust range for the number of sliders
#             st.session_state.slider_values[f"slider_{i}"] = st.slider(
#                 f"Rate Movie {selected_movies_titles[i]}",
#                 0,
#                 5,
#                 st.session_state.slider_values[f"slider_{i}"],  # Default to session state value
#                 key=f"slider_{i}",
#             )
#             dict_rates[dict_rates[i]] = st.session_state.slider_values[f"slider_{i}"] 
#         return dict_rates

# # Call the function to display sliders
# create_sidebar_sliders(selected_movies_titles, selected_moviesId)

# # Display slider values in the main app
# st.write("Current Slider Values:")
# st.write(dict_rates)

# -------------------------------------------------------------------------

dict_rates = {}

# st.session_state.slider_values = []
for i in range(len(selected_movies_titles)):
#     # # Slider for selecting the rating
#     # st.markdown(
#     # f"""
#     # <p style="font-size:24px; color:yellow;">
#     #     Rate Movie "{selected_movies_titles[i]}"
#     # </p>
#     # """,
#     # unsafe_allow_html=True
#     # )
#     # rating = st.slider(f"{i}", 0, 5, 0)

#     # # Display stars dynamically
#     # stars = "★" * rating + "☆" * (5 - rating)
#     # st.markdown(
#     # f"""
#     # <p style="font-size:24px; color:yellow;">
#     #     Your Rating: {stars}
#     # </p>
#     # """,
#     # unsafe_allow_html=True
#     # )
#     # dict_rates[selected_moviesId[i]] = rating

 
    # get_movies_data(movie_tmbds, api_key, save_directory='./images')

    # Display the movie image
    
    dict_rates[selected_moviesId[i]] = display_movie_sidebar(image_paths[i], selected_movies_titles[i],i )



# ------------- Selecting the Recommender machines ---------------------------
style = """
    <style>
    .custom-select-label {
        font-size: 25px;
        color: white;
        font-weight: bold;
    }
    </style>
    <label class="custom-select-label">Select a Recommender Model:</label>
    """

st.markdown(style, unsafe_allow_html=True)



    
# ---------------- Model Selection ------------------
options = ["None", "Engine 1 (Non-Negative Matrix Factorization)", "Engine 2 (NearestNeighbor)"]
model_name = st.selectbox(" ", options, label_visibility = "visible" )



if st.button("Recommend Movies:"):
    with st.spinner("Task is in progress..."):
        if model_name == "Engine 1 (Non-Negative Matrix Factorization)":
            model_name = 'NMF'
            df_recommendation  = recommender_nmf(dict_rates, model_name , df_ratings, df_movies, k = 10)
        elif model_name == "Engine 2 (NearestNeighbor)":
            model_name = 'NearestNeighbor'
            df_recommendation  = recommender_nearest_neghibor(dict_rates, model_name, df_ratings, df_movies, k = 10)
            
        else:
            st.markdown(
            f"""
            <p style="font-size:20px; color:yellow;">
                Sorry we can not recommend since you did not yet select the model:
            </p>
            """,
            unsafe_allow_html=True
        )
        # st.write(df_recommendation)
        # A function to reorder the rows in the recommendaion data fram according to teh genres
        df_recommendation_reordered = reorder_dataframe_by_genres(df_recommendation, genres_to_prioritize = selected_movies_genres)
        # df_recommendation_reordered = reorder_dataframe_by_genres_v2(df_recommendation, genres = selected_movies_genres)
        # st.write(df_recommendation_reordered)
        st.markdown(
        f"""
        <p style="font-size:24px; color:yellow;">
            Reommended Movies Based on Model {model_name}:
        </p>
        """,
        unsafe_allow_html=True
        )

        # st.write(movies_data)
        # Create three columns
        # col1, col2, col3, col4, col5 = st.columns(5)

        # # Place images in columns
        # with col1:
        #     st.image(image_path[0], caption="Rating {5}")

        # with col2:
        #     st.image(image_path[1], caption="Rating {5}")

        # with col3:
        #     st.image(image_path[2], caption="Rating {5}")

        # with col4:
        #     st.image(image_path[3], caption="Rating {5}")

        # with col5:
        #     st.image(image_path[4], caption="Rating {5}")

        movie_tmbds = df_movies.tmdbId.loc[df_recommendation_reordered.movieId].tolist()

        for i in range(len(movie_tmbds)):
            get_movie_info(movie_tmbds[i], api_key = '32963fd453f575aa44262db989d926d6') # col = st.columns(1)
    st.success("Task completed!")

# Reordering and shaffeling the rows so that the most relevent item comes first


# ----------------------------------------------------------------------------------


# st.image(image_path[0], caption="Rating {5}")
# st.markdown(
#         f"""
#         <p style="font-size:20px; color:yellow;">
#             Title: {movies_data[0].get('title')}
#         </p>
#         """, unsafe_allow_html=True)

# # genres_ []
# [genres_.append(movie_data['genres'][i]['name']) for i in range(len( movie_data['genres']) )]
# st.markdown(
#         f"""
#         <p style="font-size:20px; color:yellow;">
#             Renres: {genres_[i] for i in range(len(genres_))}
#         </p>
#         """, unsafe_allow_html=True)uuuuuuuuuuunccccccccccc7







# # Session state to track rating
# if "rating" not in st.session_state:
#     st.session_state.rating = 0

# # Function to update the rating
# def update_rating(selected_rating):
#     st.session_state.rating = selected_rating

# # Display star buttons
# st.write("Rate this:")
# for i in range(1, 6):
#     star_label = "★" if i <= st.session_state.rating else "☆"
#     if st.button(star_label, key=f"star_{i}"):
#         update_rating(i)

# # Display the final rating
# st.write(f"Your Rating: {'★' * st.session_state.rating}{'☆' * (5 - st.session_state.rating)}")


# # import streamlit as st
# from streamlit_star_rating import st_star_rating

# # Create a star rating widget
# rating = st_star_rating(label="Rate this:", maxValue=5, defaultValue=3)

# # Display the selected rating
# st.write(f"Your Rating: {rating}")




# # st.write('## Hello, I have started using streamlit')
# # r_i1 = st.text_input("Please give you rating for Movie Jumanji")

# # # st.write(f"Your rating multplied by 6 is {float(r_i1)*6}")

# # is_clicked = st.button('Give your rateing for Movie 1')

# # # df_rating = pd.read_csv('./data/ml-latest-small/ratings.csv')
# # df_movies = pd.read_csv('./data/ml-latest-small/movies_modified.csv')

# # st.write(df_movies)
# # st.write(df_movies)
# # years_dist = df_movies.released_yr

# # fig, ax = plt.subplots()
# # ax.hist(years_dist, bins=df_movies.released_yr.nunique())

# # st.pyplot(fig)

# tab1, tab2 = st.tabs(["Popular Movies", "Your Ratings"])

# with tab1:
#     # recommender("pop")
#     st.write("Hellow Tab 1")

# with tab2:
#     # recommender("rating")
#     st.write("Hellow Tab 2")
