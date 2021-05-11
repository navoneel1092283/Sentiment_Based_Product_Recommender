import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# reading the original data along with the predicted sentiments by the Logistic Regression Model
lr_model_predicted_user_sentiments = pd.read_csv('sample30_final.csv')
# loading the User-based Recommender Model
user_based_recommender = pd.read_csv('user_final_rating.csv').set_index('reviews_username')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	user_input = request.form.get("name")
	if user_input not in list(user_based_recommender.index):
		return render_template('index.html', message = 'Username not registered !!!')

	else:
        # getting top 20 recommended products
	    recommended_user_based_products = pd.DataFrame(user_based_recommender.loc[user_input].sort_values(ascending=False))[0:20]
        
	    # fine-tuning the recommender model outputs (products) to select top 5 products with mostly positive reviews
	    positive_products = []
	    for product in recommended_user_based_products.index:
	        predicted_sentiments = lr_model_predicted_user_sentiments.loc[lr_model_predicted_user_sentiments.name == product, 'predicted_user_sentiment']
	        final_sentiment = Counter(predicted_sentiments).most_common()[0][0]
	        if final_sentiment == 1:
	            positive_products.append(product)
	        if len(positive_products) == 5:
	            break

	    recommended_product_details = []
	    for product in positive_products:
	        brand = lr_model_predicted_user_sentiments.loc[lr_model_predicted_user_sentiments.name == product, 'brand'].iloc[0]
	        category = lr_model_predicted_user_sentiments.loc[lr_model_predicted_user_sentiments.name == product, 'categories'].iloc[0]
	        manufacturer = lr_model_predicted_user_sentiments.loc[lr_model_predicted_user_sentiments.name == product, 'manufacturer'].iloc[0]
	        recommended_product_details.append([product, brand, category, manufacturer])


	return render_template('index.html', username = 'for ' + user_input,
	                                     product1n = recommended_product_details[0][0], product1b = recommended_product_details[0][1], product1c = recommended_product_details[0][2], product1m = recommended_product_details[0][3], 
										 product2n = recommended_product_details[1][0], product2b = recommended_product_details[1][1], product2c = recommended_product_details[1][2], product2m = recommended_product_details[1][3],
										 product3n = recommended_product_details[2][0], product3b = recommended_product_details[2][1], product3c = recommended_product_details[2][2], product3m = recommended_product_details[2][3],
										 product4n = recommended_product_details[3][0], product4b = recommended_product_details[3][1], product4c = recommended_product_details[3][2], product4m = recommended_product_details[3][3],
										 product5n = recommended_product_details[4][0], product5b = recommended_product_details[4][1], product5c = recommended_product_details[4][2], product5m = recommended_product_details[4][3])

if __name__ == "__main__":
    app.run(debug=True)