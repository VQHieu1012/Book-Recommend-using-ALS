from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
import streamlit as st
import pandas as pd
import numpy as np

### Spark Setup ###
spark = SparkSession.builder.appName("Recommendation").getOrCreate()
model_path = "./model/model_1"
model = ALSModel.load(model_path)


### Prepare data


df = spark.read.csv('./data/final.csv', header=True, inferSchema=True)
pred = model.transform(df)
all_item_recommendations = model.recommendForAllItems(10)


books = pd.read_csv('./data/BX-Books.csv', on_bad_lines='skip', sep=';', encoding='latin')
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]

### Sorted highest rating books for each user
window_spec = Window.partitionBy("user_id").orderBy(col("rating").desc())

# Use the rank function to assign ranks to each item within each user's partition
ranked_df = df.withColumn("rank", rank().over(window_spec))

# Filter to get only the top 3 highest-rated items for each user
top_rated_items_df = ranked_df.filter(col("rank") <= 3).drop("rank")


### Function in ALS ###

# Recommend top k for specific user
def recommend_book_for_user(user_id, k):
    """
    user_id (int): user_id 
    k (int): top k recommended items

    Return: (ISBN, title, author, year, publisher, img_url)
    """
    user_id = int(user_id)
    userRecs = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["user_id"]), k)
    items = [row['item_id'] for row in userRecs.select('recommendations').collect()[0][0]]
    filtered_df = df.filter(df["item_id"].isin(items))
    isbn_list = filtered_df.select("ISBN").distinct().rdd.flatMap(lambda x: x).collect()
    book_data = books[books['ISBN'].isin(isbn_list)]
    result_list = [tuple(row) for _, row in book_data.iterrows()]
    return result_list

def recommend_user_for_book(book_name, k):
    """
    book_name: book title
    k: top k

    Return list(user_id, [item_id])
    """
    item_id = df.filter(df["title"] == book_name).select("item_id").first()
    user_item = list()
    subset_recommendations = all_item_recommendations.filter(all_item_recommendations["item_id"].isin([item_id[0]]))
    user_id = [row['user_id'] for row in subset_recommendations.select('recommendations').collect()[0][0]]
    for id in user_id[:k]:
        user_data = top_rated_items_df.filter(col("user_id") == id)
        item_ids = user_data.rdd.map(lambda row: row.item_id).take(3)
        user_item.append([id, item_ids])
    return user_item

def get_book_info(book_id):
    """
    book_id (list): list of book id

    Return: (ISBN, title, author, year, publisher, img_url)
    """
    filtered_df = df.filter(df["item_id"].isin(book_id))
    isbn_list = filtered_df.select("ISBN").distinct().rdd.flatMap(lambda x: x).collect()
    book_data = books[books['ISBN'].isin(isbn_list)]
    result_list = [tuple(row) for _, row in book_data.iterrows()]
    return result_list

st.header("Book recommender system")

pdf = pd.read_csv("./data/user_item_rating.csv")
book_names = pdf['title'].unique()
user_ids = pdf['user_id'].unique()

selected_books = st.selectbox(
    "Type or select a book",
    np.append(book_names, "None")
)

selected_users = st.selectbox(
    "Type or select an user",
    np.append(user_ids, "None")
)

# number = st.selectbox(
#     "Number of books",
#     np.arange(1, 6)
# )



btn1 = st.button('Show top 5 items for user id '+ selected_users+" using ALS")
if btn1:
    if selected_users != "None":
        
        result_list = recommend_book_for_user(selected_users, 5)
        col_num = len(result_list)
        col = st.columns(5)
        for i in range(col_num):
            with col[i]:
                st.text(result_list[i][1])
                st.image(result_list[i][5])

        # with col2:
        #     st.text(result_list[1][1])
        #     st.image(result_list[1][5])

        # with col3:
        #     st.text(result_list[2][1])
        #     st.image(result_list[2][5])


        # with col4:
        #     st.text(result_list[3][1])
        #     st.image(result_list[3][5])


        # with col5:
        #     st.text(result_list[4][1])
        #     st.image(result_list[4][5])
    else:
        st.write("Select user id.")

btn2 = st.button('Recommend '+ selected_books+ ' for 5 users')
if btn2:
    if selected_books != "None":
        
        user_item = recommend_user_for_book(selected_books, 5)
        users = len(user_item)
        if users != 0:
            
            for ui in range(users):
                st.write("User id: ", user_item[ui][0])
                st.write("Top 3 highest rated books rated by user:")
                result_ = get_book_info(user_item[ui][1])
                col = st.columns(3)
                for i in range(len(result_)):
                    with col[i]:
                        st.text(result_[i][1])
                        st.text(result_[i][2])
                        st.image(result_[i][5])
                    
        else:
            st.write("Nothing to show")
        
    else:
        st.write("Select book.")
    