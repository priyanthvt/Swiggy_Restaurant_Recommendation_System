import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\kmeans_trained_df.csv')
clustered_df = pd.read_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\clustered_df.csv')

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\city_encoder.pkl', 'rb') as f:
    city_encoder = pickle.load(f)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cuisine_encoder.pkl', 'rb') as f:
    cuisine_encoder = pickle.load(f)

# st.title("Swiggy Restaurant Recommendation System")
st.markdown("### Swiggy Restaurant Recommendation System")
st.markdown("#### Find the best restaurants based on your preferences")
st.markdown("##### Search Filters")

city = st.text_input('Enter city')
cuisine = st.text_input('Enter cuisine')
rating = st.number_input('Enter rating', min_value=0.0, max_value=5.0, step=0.1)
rating_count = st.number_input('Enter rating count', min_value=1)
cost = st.number_input('Enter cost', min_value=1)

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

if st.button('Search'):
    if city and cuisine and rating and rating_count and cost:
        result_df = recommend_by_all_inputs(
            city, cuisine, rating, rating_count, cost,
            df, city_encoder, cuisine_encoder, scaler, kmeans, clustered_df
        )
        result_df = result_df.sort_values(by = 'rating', ascending = False)
        result_df = result_df[result_df['cuisine'].apply(lambda x : any(cuisine.lower() in item.strip().lower() for item in x.split(',')))]
        result_df = result_df[result_df['city'].apply(lambda x : any(city.lower() in item.strip().lower() for item in x.split(',')))]

        if result_df.empty:
            st.warning("No matching restaurants found for the given inputs.")
        else:
            st.dataframe(result_df[['name', 'city', 'cuisine', 'cost', 'rating', 'rating_count']].head(30).reset_index(drop = True))

    else:
        st.warning("Please fill in all fields.")
