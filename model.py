import pandas as pd
import numpy as np
from collections import Counter
import joblib
from text_preprocessing import text_process

df = pd.read_csv('sample30.csv') # reading the original dataset

vectorizer = joblib.load('tfidf_vectorizer.pkl') # loading the tf_idf vectorizer model
lr_model = joblib.load('lr_model.pkl') # loading the logistic regression model

user_based_recommender = pd.read_csv('user_final_rating.csv').set_index('reviews_username') # loading the User-based Recommender Model

# taking the username as user input
user_input = input("Enter your username: ")
if user_input not in list(user_based_recommender.index):
    print('Sorry !!! The username is not registered !!!')
else:
    # getting top 20 recommended products
    recommended_user_based_products = pd.DataFrame(user_based_recommender.loc[user_input].sort_values(ascending=False))[0:20]

    # fine-tuning the recommender model outputs (products) to select top 5 products with mostly positive reviews
    positive_products = []
    for i in range(20):
        reviews = list(df.loc[df.name == recommended_user_based_products.index[i], 'reviews_text'])
        processed_reviews = np.array([text_process(review) for review in reviews]).astype('U')
        sentiment = Counter(lr_model.predict(vectorizer.transform(processed_reviews))).most_common()[0][0]
        if sentiment == 1:
            positive_products.append(recommended_user_based_products.index[i])
        if len(positive_products) == 5:
            break
            
    # getting the 5 final recommended product details like brand, categories and manufacturer along with the name
    recommended_user_based_product_details = {}
    recommended_user_based_product_details['name'] = []
    recommended_user_based_product_details['brand'] = []
    recommended_user_based_product_details['categories'] = []
    recommended_user_based_product_details['manufacturer'] = []

    for product in positive_products:
        recommended_user_based_product_details['brand'].append(df.loc[df.name == product, 'brand'].iloc[0])
        recommended_user_based_product_details['categories'].append(df.loc[df.name == product, 'categories'].iloc[0])
        recommended_user_based_product_details['manufacturer'].append(df.loc[df.name == product, 'manufacturer'].iloc[0])
        recommended_user_based_product_details['name'].append(product)

    print('Recommended Product Details are as follows .... ')
    print(pd.DataFrame(recommended_user_based_product_details).head())