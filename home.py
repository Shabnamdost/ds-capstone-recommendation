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
# import numpy as np
# import pandas as pd
# import time

from streamlit_carousel import carousel

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

def load_dataset():
    """ This function loads data sets """
    # # -------------------- Loading Data Set

# # first we read our data set from directroy
    df_ratings = pd.read_csv('./data/ml-latest-small/ratings_cleaned.csv')
    df_movies = pd.read_csv('./data/ml-latest-small/movies_cleaned.csv')
    df_links = pd.read_csv('./data/ml-latest-small/links_cleaned.csv')

    return df_ratings, df_movies, df_links


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
# ------------------------------------------------------------

# ---------- End of function definition ------------------

# --------------Main Body of the STreamlit Code -------------


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


# background_image(url)

# Setting the title
custom_title("SH-HA Recommender Prime (SHARP)", size=60, color="white")
# Content of your Streamlit app
custom_title(" Manking the Decision Easier and Entertaining More.", size = 40, color = 'white')
# Making the Decision Easier and Enjoy More.


st.image('.\images\complete_poster_2.jpg', width=800)

images = [
        "./images\poster_tmbd_197.0.jpg",
        "./images\poster_tmbd_9331.0.jpg",
        "./images\poster_tmbd_36955.0.jpg",
        "./images\poster_tmbd_5503.0.jpg"
    ]

# ---------- TESTing Section--------------
# import streamlit as st




autoplay_carousal(images)

# List of images

# # ---- Model Selection 
# options2 = ["NMF", "NearestNeighbor", "LongChain"]
# selected_option2 = st.selectbox("Select a Recommender Model:", options2)

# # Display the selected option
# st.write(f"Reommended Movies Based on Model: {selected_option2}", size = 10, color = 'white')


# # Create three columns
# col1, col2, col3 = st.columns(3)

# # Place images in columns
# with col1:
#     st.image(images[0], caption="Image 1")

# with col2:
#     st.image(images[1], caption="Image 2")

# with col3:
#     st.image(images[2], caption="Image 3")



# st.header("My Page Layout")

# # First container
# with st.container():
#     st.subheader("Horizontal Layout")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(images[0], caption="Left Image")
#     with col2:
#         st.image(images[1], caption="Right Image")

# # Second container
# with st.container():
#     st.subheader("Vertical Layout")
#     st.image(images[2], caption="Image Below")


# ------------ Loading Data for Movie Selction and REcommendration 
df_ratings, df_movies, df_links = load_dataset()

# Options for the selection box
movies_options = df_movies.title.tolist()[:10]

st.markdown(
    f"""
    <p style="font-size:24px; color:white;">
        Please Select 5 Movies:
    </p>
    """,
    unsafe_allow_html=True
    )

# Multiple selection box with default values
selected_movies_titles = st.multiselect(
    "Select options:", movies_options, 
)

selected_movies_df = df_movies[df_movies.title.isin(selected_movies_titles)]
selected_moviesId = selected_movies_df['movieId'].tolist()
selected_tmdbIds = df_links[df_links.movieId.isin(selected_moviesId)]['tmdbId'].tolist()
st.write(f"You selected: {selected_movies_titles}")

# st.write(selected_moviesId,)
# st.write(selected_tmdbIds)

dict_rate = {}
for i in range(len(selected_movies_titles)):
    # Slider for selecting the rating
    st.markdown(
    f"""
    <p style="font-size:24px; color:yellow;">
        Rate Movie {selected_movies_titles[i]}:
    </p>
    """,
    unsafe_allow_html=True
    )
    rating = st.slider(f"{i}", 0, 5, 0)

    # Display stars dynamically
    stars = "★" * rating + "☆" * (5 - rating)
    st.markdown(
    f"""
    <p style="font-size:24px; color:yellow;">
        Your Rating {stars}:
    </p>
    """,
    unsafe_allow_html=True
    )
    dict_rate[i] = rating


# ----------------------------------------
style = """
    <style>
    .custom-select-label {
        font-size: 30px;
        color: white;
        font-weight: bold;
    }
    </style>
    <label class="custom-select-label">Select a Recommender Model:</label>
    """

st.markdown(style, unsafe_allow_html=True)

# ---------------- Model Selection ------------------
options = ["None", "NMF", "NearestNeighbor", "LongChain"]
model_name = st.selectbox(" ", options, label_visibility = "visible" )

# Display the selected option

st.markdown(
    f"""
    <p style="font-size:24px; color:yellow;">
        Reommended Movies Based on Model {model_name}:
    </p>
    """,
    unsafe_allow_html=True
)


# ------------------ Getiing the Model Selection Info and Recommend Movies
from scripts.recommenderlib import *

if  len(dict_rate)  == 5:


    if model_name == 'NearestNeighbor':
        
        df_recommendation, image_path  = recommender_nearest_neghibor(dict_rate, model_name, df_ratings, df_movies, df_links, k = 5)
    elif model_name == 'NMF':
        
        df_recommendation, image_path  = recommender_nmf(dict_rate, model_name, df_ratings, df_movies, df_links, k = 5)
    else:
        st.markdown(
        f"""
        <p style="font-size:20px; color:yellow;">
            Sorry we can not recommend since you did not yet select the model:
        </p>
        """,
        unsafe_allow_html=True
    )

# st.write("""
#     <p style="font-size:40px; color: white">The Recommended Movies:.</p>
# """, unsafe_allow_html=True)

# Create three columns
col1, col2, col3, col4, col5 = st.columns(5)

# Place images in columns
with col1:
    st.image(image_path[0], caption="Rating {5}")

with col2:
    st.image(image_path[1], caption="Rating {5}")

with col3:
    st.image(image_path[2], caption="Rating {5}")

with col4:
    st.image(image_path[3], caption="Rating {5}")

with col5:
    st.image(image_path[4], caption="Rating {5}")



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

# # df_links = pd.read_csv('./data/ml-latest-small/links.csv')
# # st.write(df_movies)
# # st.write(df_movies)
# # years_dist = df_movies.released_yr

# # fig, ax = plt.subplots()
# # ax.hist(years_dist, bins=df_movies.released_yr.nunique())

# # st.pyplot(fig)

tab1, tab2 = st.tabs(["Popular Movies", "Your Ratings"])

with tab1:
    # recommender("pop")
    st.write("Hellow Tab 1")

with tab2:
    # recommender("rating")
    st.write("Hellow Tab 2")
