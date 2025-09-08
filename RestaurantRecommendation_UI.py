import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Swiggy Restaurant Recommendation", layout="wide")

# Load models and data
@st.cache_resource
def load_resources():
    df = pd.read_csv(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\kmeans_trained_df.csv")
    clustered_df = pd.read_csv(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\clustered_df.csv")

    with open(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\city_encoder.pkl", "rb") as f:
        city_encoder = pickle.load(f)

    with open(r"E:\Priyanth\AIML\Project\RestaurantRecommendation\cuisine_encoder.pkl", "rb") as f:
        cuisine_encoder = pickle.load(f)

    return df, clustered_df, scaler, kmeans, city_encoder, cuisine_encoder


def recommend_by_all_inputs(city, cuisine, rating, rating_count, cost, df,
                            city_encoder, cuisine_encoder, scaler, kmeans, clustered_df):
    numeric_input = pd.DataFrame([[rating, rating_count, cost]],
                                 columns=['rating', 'rating_count', 'cost'])

    scaled_numeric = pd.DataFrame(scaler.transform(numeric_input), columns=numeric_input.columns)

    city_encoded = pd.DataFrame(city_encoder.transform([[city]]),
                                columns=city_encoder.classes_)

    cuisine_encoded = pd.DataFrame(cuisine_encoder.transform([[cuisine]]),
                                   columns=cuisine_encoder.classes_)

    input_vector = pd.concat([scaled_numeric, city_encoded, cuisine_encoded], axis=1)

    missing_cols = set(df.columns) - set(input_vector.columns)
    for col in missing_cols:
        input_vector[col] = 0

    input_vector = input_vector[df.columns]

    cluster = kmeans.predict(input_vector)[0]

    return clustered_df[clustered_df['cluster'] == cluster]


def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Initialize screen
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

df, clustered_df, scaler, kmeans, city_encoder, cuisine_encoder = load_resources()

# HOME SCREEN
if st.session_state['page'] == 'home':
    set_background("https://images.unsplash.com/photo-1555396273-367ea4eb4db5")
    # set_background("https://images.unsplash.com/photo-1498654896293-37aacf113fd9")
    st.markdown("<h1 style='color:white;'>Welcome to Swiggy Restaurant Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:white;'>Find the best places to eat, tailored to your taste!</h3>", unsafe_allow_html=True)

    if st.button("Go to Search"):
        st.session_state['page'] = 'search'
        st.rerun()


# SEARCH SCREEN
elif st.session_state['page'] == 'search':
    # set_background("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4")
    set_background("https://images.unsplash.com/photo-1498654896293-37aacf113fd9")
    st.markdown("<h2 style='color:white;'>Search for Restaurants</h2>", unsafe_allow_html=True)

    city = st.text_input('Enter city')
    cuisine = st.text_input('Enter cuisine')
    rating = st.number_input('Enter rating', min_value=0.0, max_value=5.0, step=0.1)
    rating_count = st.number_input('Enter rating count', min_value=1)
    cost = st.number_input('Enter cost', min_value=1)

    if st.button("Search"):
        if city and cuisine and rating and rating_count and cost:
            st.session_state['inputs'] = {
                "city": city,
                "cuisine": cuisine,
                "rating": rating,
                "rating_count": rating_count,
                "cost": cost
            }
            st.session_state['page'] = 'results'
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

    if st.button("Back to Home"):
        st.session_state['page'] = 'home'
        st.rerun()


# RESULTS SCREEN
elif st.session_state['page'] == 'results':
    set_background("https://images.unsplash.com/photo-1555396273-367ea4eb4db5")
    st.markdown("<h2 style='color:white;'>🍴 Recommended Restaurants</h2>", unsafe_allow_html=True)

    user_input = st.session_state.get('inputs', {})

    if not user_input:
        st.warning("No inputs provided. Please go back to Search.")
        if st.button("Back to Search"):
            st.session_state['page'] = 'search'
            st.rerun()
    else:
        result_df = recommend_by_all_inputs(
            user_input['city'], user_input['cuisine'], user_input['rating'],
            user_input['rating_count'], user_input['cost'],
            df, city_encoder, cuisine_encoder, scaler, kmeans, clustered_df
        )

        result_df = result_df.sort_values(by='rating', ascending=False)

        result_df = result_df[
            result_df['cuisine'].apply(lambda x: any(user_input['cuisine'].lower() in item.strip().lower()
                                                     for item in x.split(',')))]
        result_df = result_df[
            result_df['city'].apply(lambda x: any(user_input['city'].lower() in item.strip().lower()
                                                  for item in x.split(',')))]

        if result_df.empty:
            st.warning("No matching restaurants found for the given inputs.")
        else:
            st.dataframe(result_df[['name', 'city', 'cuisine', 'cost', 'rating', 'rating_count']].head(30).reset_index(drop=True))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Again"):
            st.session_state['page'] = 'search'
            st.rerun()
    with col2:
        if st.button("Back to Home"):
            st.session_state['page'] = 'home'
            st.rerun()
