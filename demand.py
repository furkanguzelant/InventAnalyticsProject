import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error


data = pd.read_csv('data/train.csv')
product = pd.read_csv('data/product.csv')

data.date = pd.to_datetime(data.date)


# Fill nan values
data['price']= data['price'].fillna(data['price'].mean())
data['promotion_type'] = data['promotion_type'].fillna("No promotion")


product['color_type'] = product['color_type'].fillna(product['color_type'].mode()[0])
product['life_style'] = product['life_style'].fillna(product['life_style'].mode()[0])
product['fabric'] = product['fabric'].fillna(product['fabric'].mode()[0])
product['weight_of_fabric'] = product['weight_of_fabric'].fillna(product['weight_of_fabric'].mode()[0])
product['neck_style'] = product['neck_style'].fillna(product['neck_style'].mode()[0])
product['form_type'] = product['form_type'].fillna(product['form_type'].mode()[0])
product['sleeve_type'] = product['sleeve_type'].fillna(product['sleeve_type'].mode()[0])
product['washing_style'] = product['washing_style'].fillna(product['washing_style'].mode()[0])
product['fabric_type'] = product['fabric_type'].fillna(product['fabric_type'].mode()[0])



print(product.isna().sum())


desc = data.describe()
corr_matrix = data.corr()

dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='7D')


g = data.groupby('product_id')
first = g.get_group((list(g.groups)[0]))


X = data.drop(["season_type","promotion_type", "date"], axis=1)

X['day'] = data['date'].dt.day
X['month'] = data['date'].dt.month
X['year'] = data['date'].dt.year
X['quarter'] = data['date'].dt.quarter

ohe = OneHotEncoder()
season_encoded = ohe.fit_transform(data[["season_type"]])
season_encoded = season_encoded.toarray()
X = pd.concat([X, pd.DataFrame(season_encoded, columns=['Autumn-Winter', 'Summer-Spring'])], axis=1)

ohe = OneHotEncoder()
promotion_encoded = ohe.fit_transform(data[["promotion_type"]])
promotion_encoded = promotion_encoded.toarray()
X = pd.concat([X, pd.DataFrame(promotion_encoded)], axis=1)
y = data["sales_amount"]


# X.to_csv('new_train.csv')
# product.to_csv('new_product.csv')


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                random_state=0, shuffle=False, train_size= 0.8)
