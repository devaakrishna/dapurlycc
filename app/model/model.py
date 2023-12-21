import random
import math 
import pandas as pd
import tensorflow as tf
import numpy as np
from sqlalchemy import create_engine, text

# version of model
__version__ = "2.0.0"
# change the url
def get_data_template(table_name):
    db_user = 'root'
    db_password = '12345'
    db_port = 3306
    db_host = "34.128.103.197"
    db_name = 'dapurly-db'

    # Create the connection string
    connection_string =  f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

    # Create the engine
    engine = create_engine(connection_string)

    # Write your SQL query
    sql_query = text(f'SELECT * FROM `{table_name}`')

    # Use pandas.read_sql() to read the query results into a DataFrame
    # df = pd.read_sql_query(sql_query, con = engine)
    with engine.connect().execution_options(stream_results=True) as connection:
        result = connection.execute(sql_query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df
# change the url        
data_recipe = get_data_template(table_name="all-recipe")
base_ratings = get_data_template(table_name='user-ratings')  
food_price = get_data_template(table_name='harga_clean')  
 
    
# load the model  
def load_model(): 
    # change the url (for load model use the API)
    load_model = tf.keras.models.load_model("/path/in/container/RecomendationV2.h5")
    return load_model


# load thee weight, X and bias
def load_weights_X_bias():
    # change the url
    W = tf.Variable(get_data_template(table_name='small-W-final')) 
    X  = tf.Variable(get_data_template(table_name='small-X-final'))
    bias  = tf.Variable(get_data_template(table_name='small-B-final')) 
    return W,X, bias

# this function was created to get new user's ratings
def get_new_user_ratings(recipe_dataset, base_ratings, user_country_references): 
    new_user_rating = np.zeros(base_ratings.shape[0])
    rated_by_new_user_index = np.zeros(base_ratings.shape[0])
    country_liked_selected = user_country_references
    for j in country_liked_selected:
        for i in range(len(recipe_dataset)): 
            if recipe_dataset['kategori'].iloc[i] in j:
                random_rate = np.random.randint(low = 3, high=6) /5
                rated_by_new_user_index[i] = int(i) 
                new_user_rating[i] = random_rate
    return new_user_rating, rated_by_new_user_index

# this function  is to convet based ratings dataset to numpy array
def based_ratings(based_ratings): 
    Y = []
    temp = []
    for i in range(len(based_ratings)): 
        for j in range(1, 200): 
            temp.append(based_ratings[f'user{j}'].iloc[i])
        Y.append(temp)
        temp = []
    return Y

# concat the new user's ratings and get the index
def concat_based_new_user_ratings(user_country_references_param): 
    # this function actually have user_country_references, CC team you can add the parameter with the configuration 
    get_based_ratings = based_ratings(base_ratings)
    new_user_ratings, new_user_ratings_index = get_new_user_ratings(recipe_dataset = data_recipe, base_ratings = base_ratings, user_country_references =  user_country_references_param)
    Y_concat = np.array(np.c_[new_user_ratings, get_based_ratings])
    return Y_concat
   

def retrain(country_refrences):
    user_weight,recipe_x, bias = load_weights_X_bias() 
    Y = concat_based_new_user_ratings(user_country_references_param=country_refrences)
    model = load_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics = 'mse')
    model.fit(x = np.matmul(recipe_x.numpy(), np.transpose(user_weight)) + bias.numpy(), y = Y, epochs=2)
    new_user_prediction = model.predict(np.matmul(recipe_x.numpy(), np.transpose(user_weight)) + bias.numpy())[:, 0]
    pred = tf.argsort(new_user_prediction, direction = 'DESCENDING')
    return pred
# give the final recomendation based on user preferences
def final_recomendation(prediction, bahan_yang_disukai_param, bahan_yang_tidak_disukai_param, pantangan_makan_param, budget_param, jumlah_makan_sehari_param, jumlah_dewasa_param, jumlah_anak_param):
    bahan_yang_disukai = bahan_yang_disukai_param
    bahan_yang_tidak_disukai = bahan_yang_tidak_disukai_param
    pantangan_makanan = pantangan_makan_param
    budget = budget_param
    jumlah_makan_sehari = jumlah_makan_sehari_param
    jumlah_dewasa = jumlah_dewasa_param
    jumlah_anak = jumlah_anak_param
    all_recomendation = []
    # change the url
    data_bahan = data_recipe['nama_bahan']   
    # change the url
    data_harga_bahan = food_price
    recipe_dataset = data_recipe
    for i in range(len(prediction)):   
        j = prediction[i]
        all_recomendation.append(int(j))

    recipe_filter_by_bahan = []
    for i in range(len(all_recomendation)):
        for j in range(len(bahan_yang_disukai)): 
            if bahan_yang_disukai[j] in data_bahan[i].replace("'", '').split(",") or bahan_yang_disukai[j].lower() in data_bahan[i].replace("'", '').split(","): 
                if (bahan_yang_tidak_disukai not in data_bahan[i].replace("'", '').split(",") or bahan_yang_tidak_disukai.lower() not in data_bahan[i].replace("'", '').split(",")) and pantangan_makanan not in data_bahan[i].replace("'", '').split(","): 
                    recipe_filter_by_bahan.append(int(i))

    for j in recipe_filter_by_bahan:
        for i in range(1, len(recipe_filter_by_bahan)): 
            if int(float(recipe_dataset['kandungan_nutrisi'].iloc[int(recipe_filter_by_bahan[i])].replace("'", "").replace("'", "").strip().split(',')[1])) > int(float(recipe_dataset['kandungan_nutrisi'].iloc[int(recipe_filter_by_bahan[i - 1])].replace("'", "").replace("'", "").strip().split(',')[1])): 
                temp = recipe_filter_by_bahan[i-1]
                recipe_filter_by_bahan[i -1] = recipe_filter_by_bahan[i]
                recipe_filter_by_bahan[i] = temp
    get_harga_bahan = []
    get_bahan_recipe = []
    for i in recipe_filter_by_bahan:
        get_bahan_recipe.append(recipe_dataset['nama_bahan'].iloc[i].replace("'", "").replace("'", "").split(','))
    all_satuan = []

    for i in range(len(recipe_filter_by_bahan)): 
        temp  = recipe_dataset['satuan'].iloc[i].replace("'", "").split(",")
        all_satuan.append(temp)

    for i in range(len(get_bahan_recipe)): 
        total = 0
        for j in range(len(get_bahan_recipe[i])): 
            for k in range(len(data_harga_bahan)): 
                if str(data_harga_bahan['nama_bahan'].iloc[k]).replace(" ", "") == get_bahan_recipe[i][j].replace(" ", "") :
                    if j < len(all_satuan[i]):
                        if all_satuan[i][j] == "ons":   
                            total += ((data_harga_bahan['harga'].iloc[k]) / 4)
                        elif str(all_satuan[i][j]).replace(" ", "") == 'sendokteh' or str(all_satuan[i][j]).replace(" ", "") == 'sendokmakan':   
                            total += ((data_harga_bahan['harga'].iloc[k]) / 10)
                        elif str(all_satuan[i][j].replace(" ", "")) == 'cangkir' or 'cangkir' in str(all_satuan[i][j].replace(" ", "")):   
                            total += ((data_harga_bahan['harga'].iloc[k]) / 2) 
                    else: 
                        total += (data_harga_bahan['harga'].iloc[k])
        get_harga_bahan.append(total)


    for i in range(len(get_harga_bahan)): 
        if get_harga_bahan[i] <= 50000: 
            continue
        elif get_harga_bahan[i] <= 100000 and get_harga_bahan[i] >  50000: 
            get_harga_bahan[i] = get_harga_bahan[i] / 4
        elif get_harga_bahan[i] > 100000 and get_harga_bahan[i] <= 200000: 
            get_harga_bahan[i] = get_harga_bahan[i] / 6
        else: 
            get_harga_bahan[i] = get_harga_bahan[i] / 8
        
    final_recomend = []
    for i in range(len(get_harga_bahan)): 
        temp = []
        total  = 0
        max_jump = int(len(get_harga_bahan) / (7 * jumlah_makan_sehari)) 
        jump = random.randint(a=1, b= max_jump)
        for j in range(i, len(get_harga_bahan) - jump): 
            if int(total) <= int(budget):
                total += (get_harga_bahan[j + jump] * (jumlah_dewasa + math.ceil(0.5 * jumlah_anak)))
                temp.append(recipe_filter_by_bahan[j + jump])
            else: 
                  temp = []
            
        if len(temp) >= 7 * jumlah_makan_sehari: 
            if temp not in final_recomend:
                final_recomend.append(temp)
    return final_recomend
def get_recommender(bahan_yang_disukai, bahan_yang_tidak_disukai , pantangan_makan, budget, jumlah_makan_sehari, jumlah_dewasa, jumlah_anak, country_refrences_param): 
    pred =  retrain(country_refrences = country_refrences_param)
    all = final_recomendation(prediction=pred, bahan_yang_disukai_param= bahan_yang_disukai, bahan_yang_tidak_disukai_param=bahan_yang_tidak_disukai, pantangan_makan_param= pantangan_makan, budget_param= budget, jumlah_makan_sehari_param=jumlah_makan_sehari, jumlah_anak_param = jumlah_anak,jumlah_dewasa_param=jumlah_dewasa)
    return all