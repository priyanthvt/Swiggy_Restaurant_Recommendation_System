import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\swiggy.csv')

df = df.drop(columns = ['id', 'lic_no', 'link', 'address', 'menu'])
df = df.drop_duplicates()
df = df.dropna()

df['rating'] = pd.to_numeric(df['rating'], errors = 'coerce')
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['rating'] = df['rating'].round(2)

df['rating_count'] = df['rating_count'].replace({'Too Few Ratings' : 10, '20+ ratings' : 25,
                                                 '50+ ratings' : 75, '100+ ratings' : 250,
                                                 '500+ ratings' : 750, '1K+ ratings' : 2500,
                                                 '5K+ ratings' : 7500, '10K+ ratings' : 12500})

df['cost'] = df['cost'].replace(r'[^0-9]', '', regex = True).astype('int')
df = df.reset_index(drop = True)
# print(df)
print(f'Cleaned data : {df.shape}')

df.to_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cleaned_data.csv')
print('Cleaned Data saved as csv')

df['city_list'] = df['city'].apply(lambda x : x.split(','))
df['cuisine_list'] = df['cuisine'].apply(lambda x : x.split(','))

mlb_city = MultiLabelBinarizer()
encoded_city = pd.DataFrame(mlb_city.fit_transform(df['city_list']), columns = mlb_city.classes_)

mlb_cuisine = MultiLabelBinarizer()
encoded_cuisine = pd.DataFrame(mlb_cuisine.fit_transform(df['cuisine_list']), columns = mlb_cuisine.classes_)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\city_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_city, f)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cuisine_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_cuisine, f)

encoded_df = pd.concat([encoded_city, encoded_cuisine], axis=1)
# print(encoded_df)
print(f'Encoded data : {encoded_df.shape}')

encoded_df.to_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\encoded_data.csv')
print('Encoded data saved as csv')

numeric_columns = ['rating', 'rating_count', 'cost']

scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[numeric_columns])
scaled_df = pd.DataFrame(scaled_array, columns = numeric_columns)
print(scaled_df)
print(f'Scaled df : {scaled_df.shape}')

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

kmeans_training_df = pd.concat([scaled_df, encoded_df], axis = 1)
print(kmeans_training_df)
print(f'kmeans training df : {kmeans_training_df.shape}')
kmeans_training_df.to_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\kmeans_trained_df.csv', index = False)
print('Kmeans training df saved')

kmeans = KMeans(n_clusters = 15, random_state = 100)
kmeans_training_df['cluster'] = kmeans.fit_predict(kmeans_training_df)
print(kmeans_training_df)
print(f'kmeans training df with cluster : {kmeans_training_df.shape}')

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

df['cluster'] = kmeans_training_df['cluster']

print(df)
print(f'cleaned df with cluster : {df.shape}')

df.to_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\clustered_df.csv', index = False)
print('Clustered df saved')
