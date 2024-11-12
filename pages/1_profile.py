import streamlit as st

# import pandas as pd


# Add CSS to set the background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.pixelstalk.net/wp-content/uploads/2016/06/Free-Images-Wallpaper-HD-Background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Content of your Streamlit app
st.title("Streamlit App with Background Image")
st.write("This is an example of adding a background image to your Streamlit app.")


st.write('## Hello, I have started using streamlit')
r_i1 = st.text_input("Please give you rating for Movie Jumanji")

st.write(f"Your rating multplied by 6 is {float(r_i1)*6}")

is_clicked = st.button('Give your rateing for Movie 1')

# df_rating = pd.read_csv('./data/ml-latest-small/ratings.csv')
# df_movies = pd.read_csv('./data/ml-latest-small/movies_modified.csv')

# df_links = pd.read_csv('./data/ml-latest-small/links.csv')
# st.write(df_movies)

