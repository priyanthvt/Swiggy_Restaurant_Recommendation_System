import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\swiggy.csv')

df = df.drop(columns = ['id', 'lic_no', 'link', 'address', 'menu'])
df = df.drop_duplicates()
df = df.dropna()

df['rating'] = pd.to_numeric(df['rating'], errors = 'coerce').fillna(0.0)

df['rating_count'] = df['rating_count'].replace({'Too Few Ratings' : 10, '20+ ratings' : 25,
                                                 '50+ ratings' : 75, '100+ ratings' : 250,
                                                 '500+ ratings' : 750, '1K+ ratings' : 2500,
                                                 '5K+ ratings' : 7500, '10K+ ratings' : 12500})

df['cost'] = df['cost'].replace(r'[^0-9]', '', regex = True).astype('int')

print(df)

df.to_csv(r'C:\Users\Sheasaanth\Desktop\Priyanth\Projects\Swiggy_Restaurant_Recommendation_System\cleaned_data.csv')
print('Saved as csv')
