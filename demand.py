import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.impute import KNNImputer
import numpy as np
from statistics import mode

new_product = pd.read_csv('new_product.csv')
data = pd.read_csv('data/train.csv')
product = pd.read_csv('data/product.csv')
product_data = pd.read_csv('data/product.csv')

data.date = pd.to_datetime(data.date)


# Fill nan values
data['price']= data['price'].fillna(data['price'].mean())
data['promotion_type'] = data['promotion_type'].fillna("No promotion")


# product['color_type'] = product['color_type'].fillna(product['color_type'].mode()[0])
# product['life_style'] = product['life_style'].fillna(product['life_style'].mode()[0])
# product['fabric'] = product['fabric'].fillna(product['fabric'].mode()[0])
# product['weight_of_fabric'] = product['weight_of_fabric'].fillna(product['weight_of_fabric'].mode()[0])
# product['neck_style'] = product['neck_style'].fillna(product['neck_style'].mode()[0])
# product['form_type'] = product['form_type'].fillna(product['form_type'].mode()[0])
# product['sleeve_type'] = product['sleeve_type'].fillna(product['sleeve_type'].mode()[0])
# product['washing_style'] = product['washing_style'].fillna(product['washing_style'].mode()[0])
# product['fabric_type'] = product['fabric_type'].fillna(product['fabric_type'].mode()[0])

def removeAll(myList, valueToBeRemoved):
    try:
        while True:
            myList.remove(valueToBeRemoved)
    except ValueError:
        pass
    
    return myList

temp = []
def fillProductNanValues(column, indices):

    for index in indices:
        
        if(index <= 10 or index + 10 >= len(indices)):
            product[column][index] = product[column][index - 1]
        else:
            neighbors = product[column][index - 10: index]
            top_list = neighbors.tolist()
            bottom_neighbors = product[column][index + 1: index + 10]
            bottom_list = bottom_neighbors.tolist()
            top_list.extend(bottom_list)
            top_list = [str(i) for i in top_list]
            top_list = removeAll(top_list, "nan")
            
            product[column][index] = mode(top_list)

columns = ['color_type', 'life_style', 'fabric', 'weight_of_fabric', 
           'neck_style', 'form_type', 'sleeve_type', 'washing_style', 'fabric_type']

for column in columns:
    indices = list(np.where(product[column].isnull())[0])            
    fillProductNanValues(column, indices)

df_cat = product.drop('id', axis=1)
encoder = OrdinalEncoder()

cat_cols = df_cat.columns
product_encoded = product
def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data


#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(product_encoded[columns])

# imputer = KNNImputer(n_neighbors= 3)
# product_list = imputer.fit_transform(product_encoded)
# product_df = pd.DataFrame(product_list, columns=product.columns)
    

# def updateFloatValues(column):
#     for index, item in product_df[column].items():
#             if(item % 1 != 0):
#                 product_df[column][index] = product_df[column][index - 1]

# columns = ['color_type', 'life_style', 'fabric', 'weight_of_fabric', 
#            'neck_style', 'form_type', 'sleeve_type', 'washing_style', 'fabric_type']

# for column in columns:
#     updateFloatValues(column)    

                    

            

print(product.isna().sum())


desc = data.describe()
corr_matrix = data.corr()

dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='7D')


g = data.groupby('product_id')

first = g.get_group((list(g.groups)[0]))

p_22 = product.loc[product['id'] == 22]


first = pd.concat([first, p_22], axis=1)
print(p_22.iloc[0][1])




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



# data['category_1'] = ' '
# data['category_2'] = ' '
# data['category_3'] = ' '
# data['color_type'] = ' '
# data['life_style'] = ' '
# data['fabric'] = ' '
# data['weight_of_fabric'] = ' '
# data['neck_style'] = ' '
# data['form_type'] = ' '
# data['sleeve_type'] = ' '  
# data['washing_style'] = ' '


# data['fabric_type'] = ' '
# exact_product = product.loc[product['id'] == 22]
# data.at[0, 'category_1'] = exact_product.iloc[0][1]
 
# for index, row in data.iterrows():
#     print(index)
#     product_id = row['product_id']
#     exact_product = product.loc[product['id'] == product_id]
#     y = exact_product.iloc[0][1]
#     data.at[index, 'category_1'] = exact_product.iloc[0][1]
#     data.at[index, 'category_2'] = exact_product.iloc[0][2]
#     data.at[index, 'category_3'] = exact_product.iloc[0][3]
#     data.at[index, 'color_type'] = exact_product.iloc[0][4]
#     data.at[index, 'life_style'] = exact_product.iloc[0][5]
#     data.at[index, 'fabric'] = exact_product.iloc[0][6]
#     data.at[index, 'weight_of_fabric'] = exact_product.iloc[0][7]
#     data.at[index, 'neck_style'] = exact_product.iloc[0][8]
#     data.at[index, 'form_type'] = exact_product.iloc[0][9]
#     data.at[index, 'sleeve_type'] = exact_product.iloc[0][10]
#     data.at[index, 'washing_style'] = exact_product.iloc[0][11]
#     data.at[index, 'fabric_type'] = exact_product.iloc[0][12]
    
    
# print(data)

# data.to_csv('merged-table.csv')

product_encoded.to_csv('product_encoded.csv')
product.to_csv('product_na_filled.csv')
