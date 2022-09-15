#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd

#Loading Dataset
train_data = pd.read_csv(r'/Users/ahmetkaansever/Downloads/InventAnalyticsProject-master/new_train.csv')
a = train_data.head()
b = train_data.describe()

test_data = pd.read_csv(r'/Users/ahmetkaansever/Downloads/dac22-invent-analytics-project/test.csv')

products = pd.read_csv(r'/Users/ahmetkaansever/Downloads/InventAnalyticsProject-master/new_product.csv')
products.shape


# In[153]:


print(train_data.shape)
print(test_data.shape)


# In[154]:


get_ipython().run_line_magic('whos', '')


# In[155]:


get_ipython().run_line_magic('whos', '')


# In[156]:


test_data['category_1'] = ' '


# In[157]:


test_data['category_2'] = ' '
test_data['category_3'] = ' '
test_data['color_type'] = ' '
test_data['life_style'] = ' '
test_data['fabric'] = ' '
test_data['weight_of_fabric'] = ' '
test_data['neck_style'] = ' '
test_data['form_type'] = ' '
test_data['sleeve_type'] = ' '  
test_data['washing_style'] = ' '
test_data['fabric_type'] = ' '


# In[158]:


train_data['category_1'] = ' '
train_data['category_2'] = ' '
train_data['category_3'] = ' '
train_data['color_type'] = ' '
train_data['life_style'] = ' '
train_data['fabric'] = ' '
train_data['weight_of_fabric'] = ' '
train_data['neck_style'] = ' '
train_data['form_type'] = ' '
train_data['sleeve_type'] = ' '  
train_data['washing_style'] = ' '
train_data['fabric_type'] = ' '


# In[159]:


test_data


# In[160]:


train_data


# In[161]:


print(products.iloc[47])


# In[162]:


#prev = 0
#odd_ones = []
    #for index, row in products.iterrows():
    #if(prev + 1 != row['id']):
        #odd_ones.append(row['id'] - 1)
    #prev = row['id']
#print(odd_ones)


    
    


# In[163]:


products


# In[164]:


for index, row in train_data.iterrows():
    for odd in odd_ones:
        if(row['product_id'] == odd):
            train_data = train_data.drop(index)
train_data


# In[165]:


for index, row in test_data.iterrows():
    product_id = row['product_id']
    exact_product = products.loc[products['id'] == product_id]
    row['category_1'] = exact_product.iloc[0][1]
    row['category_2'] = exact_product.iloc[0][2]
    row['category_3'] = exact_product.iloc[0][3]
    row['color_type'] = exact_product.iloc[0][4]
    row['life_style'] = exact_product.iloc[0][5]
    row['fabric'] = exact_product.iloc[0][6]
    row['weight_of_fabric'] = exact_product.iloc[0][7]
    row['neck_style'] = exact_product.iloc[0][8]
    row['form_type'] = exact_product.iloc[0][9]
    row['sleeve_type'] = exact_product.iloc[0][10]
    row['washing_style'] = exact_product.iloc[0][11]
    row['fabric_type'] = exact_product.iloc[0][12]
    
print(test_data)


# In[ ]:




