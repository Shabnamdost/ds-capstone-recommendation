import streamlit as st

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

# Page Title
# st.title("Recommender Systems: Solving the Puzzle of Personalized Choices")
st.markdown("<h1 style='text-align: center; font-size:40px; color: white;'> Recommender Systems: Solving the Puzzle of Personalized Choice </h1>", unsafe_allow_html=True)
st.markdown("<h1 style=' font-size:30px; color: white;'> Our Team </h1>", unsafe_allow_html=True)

# st.header("Our Team")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image('./images/hamid.jpg')
with col2:
    st.markdown("<h1 style=' font-size:22px; color: white;'> Hamid Mehdipour </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style=' font-size:22px; color: white;'> Ph. D. Physics </h1>", unsafe_allow_html=True)

    # st.write('#### Ph. D. Physics')

with col3:
    st.image('./images/shabnam.jpg')
with col4:
    st.markdown("<h1 style=' font-size:22px; color: white;'> Shabnam Dost </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style=' font-size:22px; color: white;'> MSc. Business Intelligence </h1>", unsafe_allow_html=True)



# Introduction
st.markdown("<h1 style=' font-size:30px; color: white;'> Project Overview </h1>", unsafe_allow_html=True)

        # Number of Movie Options:  {len(movies_options_p)}:
   

# st.header("Project Overview")
st.markdown("""<p style="font-size:18px; color:white;"> This project aims to create a personalized movie recommendation system 
to enhance user satisfaction and engagement in the global streaming industry. 
The innovative approach ensures ethical and inclusive recommendations 
while expanding to multimodal integrations.</p>
""", unsafe_allow_html=True)

# Stakeholders Section
# st.header("Stakeholders")
st.markdown("<h1 style=' font-size:30px; color: white;'> Stakeholders </h1>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .custom-text {
        color: white; /* Set text color to white */
    }
    </style>
    <div class="custom-text">
        <b>- End-users </b>: Individuals looking for personalized content recommendations.<br>
        <b>- Data Scientists</b>: Developers and algorithm designers.<br>
        <b>- Investors</b>: Financial backers interested in scaling the platform.
    </div>
    """,
    unsafe_allow_html=True
)


# Objectives Section
# st.header("Key Objectives")
st.markdown("<h1 style=' font-size:30px; color: white;'> Key Objectives </h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-text {
        color: white; /* Set text color to white */
    }
    </style>
    <div class="custom-text">
        <b>1. Personalized Recommendations:</b> Tailored suggestions based on user preferences.<br>
        <b>2. Global Challenges:</b> Addressing issues in the global streaming industry.<br>
        <b>3. Innovation:</b> Utilizing cutting-edge recommendation approaches.
    </div>
    """,
    unsafe_allow_html=True
)

# Statistics
# st.header("Statistics")
st.markdown("<h1 style=' font-size:30px; color: white;'> Statistics </h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .custom-text {
        color: white; /* Set text color to white */
    }
    </style>
    <div class="custom-text">
        The recommendation system leverages data from: <br>
        - Open-source datasets. <br>
        - User ratings and tags.<br>
        - Over 9,724 movies classified with IMDB genres.
    </div>
    """,
    unsafe_allow_html=True
)

# Future Goals
# st.header("Future Goals")
st.markdown("<h1 style=' font-size:30px; color: white;'> Future Goals </h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .custom-text {
        color: white; /* Set text color to white */
    }
    </style>
    <div class="custom-text">
        <b>- Enhanced Personalization:</b> Improved algorithms for better user alignment. <br>
        <b>- Ethical and Inclusive Recommendations:</b> Focus on fairness and inclusivity.<br>
        <b>- Multimodal Integration:</b> Combining various media formats for robust recommendations.
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
# st.success("This presentation was created by Hamid Mehdipoury, and Shabnam Dost.")
