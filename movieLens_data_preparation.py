from surprise import Dataset, Reader, SVD
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

ratings_df = pd.read_csv('./dataset/ml-10M100K/ratings.dat', sep='::', header=None, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
movies_df = pd.read_csv('./dataset/ml-10M100K/movies.dat', sep='::', header=None, names=['movieId', 'title', 'genres'], engine='python')

num_item = 200
d = 5
movie_counts = ratings_df['movieId'].value_counts()
top_n_movies = movie_counts.head(num_item).index.tolist()
top_n_df = ratings_df[ratings_df['movieId'].isin(top_n_movies)]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(top_n_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD(n_factors=d, biased=True, verbose=True)  
model.fit(trainset)
user_embeddings = model.pu 
item_embeddings = model.qi  
num_users = top_n_df['userId'].nunique() 
user_item_matrix = np.zeros((num_users, num_item))    
user_item_matrix_mask = np.zeros((num_users, num_item), dtype=bool)
sorted_movie_ids = sorted(top_n_movies)
movie_id_to_sorted_index = {movie_id: index for index, movie_id in enumerate(sorted_movie_ids)}
i = 0
for user_id in top_n_df['userId'].unique():
    user_df = top_n_df[top_n_df['userId'] == user_id]
    for index, row in user_df.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        sorted_index = movie_id_to_sorted_index.get(movie_id)
        if sorted_index is not None:
            user_item_matrix[i][sorted_index] = rating
            user_item_matrix_mask[i][sorted_index] = True
    i += 1


pred = np.matmul(user_embeddings, item_embeddings.T)

import pickle


file_name = "./dataset/MovieLens10M_{}_{}.pkl".format(num_item, d)


data_to_save = {
    "user_embeddings": user_embeddings,
    "item_embeddings": item_embeddings,
    "ratings": user_item_matrix,
    "mask": user_item_matrix_mask
}


with open(file_name, 'wb') as file:
    pickle.dump(data_to_save, file)

print(f"Data saved to {file_name}")