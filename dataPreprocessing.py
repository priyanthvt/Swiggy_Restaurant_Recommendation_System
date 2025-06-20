import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

df = pd.read_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cleaned_data.csv')

df['city_list'] = df['city'].apply(lambda x : x.split(','))
df['cuisine_list'] = df['cuisine'].apply(lambda x : x.split(','))

mlb_city = MultiLabelBinarizer()
encoded_city = pd.DataFrame(mlb_city.fit_transform(df['city_list']), columns = mlb_city.classes_)

mlb_cuisine = MultiLabelBinarizer()
encoded_cuisine = pd.DataFrame(mlb_cuisine.fit_transform(df['cuisine_list']), columns = mlb_cuisine.classes_)

encoded_df = pd.concat([encoded_city, encoded_cuisine], axis=1)
print(encoded_df)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\city_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_city, f)

with open(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cuisine_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_cuisine, f)

print('Pickle saved')
