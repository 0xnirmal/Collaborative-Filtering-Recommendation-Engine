import sys
import math
from predictor import Predictor, compute_weighted_average, compute_mean, clean_user
import collections
import math
import numpy as np

class Cosine(Predictor):
	def __init__(self):
		self.user_data = None

	def train(self, user_data):
		self.user_data = user_data

	def predict(self, user_num, user_set, num_neighbors):
		#Make sure you hide the training data on the values in the user_set
		prediction_set = {}
		user_pre_clean = self.user_data[user_num]
		user_post_clean = clean_user(user_pre_clean, user_set)
		#user_post_clean contains all the values for a user not in our prediction set
		similarity_to_user = []
		#generate relevant subset
		for user_v in self.user_data:
			# exclude current user
			if user_v != user_num:
				similarity = self.calculate_cosine_similarity(user_post_clean, self.user_data[user_v])
				similarity_to_user.append((similarity, self.user_data[user_v]))
		similarity_to_user.sort(key=lambda x: x[0], reverse=True)
		for movie in user_set:
			prediction = compute_weighted_average(user_post_clean, similarity_to_user, movie, num_neighbors)
			if prediction > 5.0:
				prediction = 5.0
			elif prediction < 0.5:
				prediction = 0.5
			prediction_set[movie] = prediction
		return collections.OrderedDict(sorted(prediction_set.items()))

	def calculate_cosine_similarity(self, u, v):
		mean_u = compute_mean(u)
		mean_v = compute_mean(v)
		#generate intersection subset
		intersection = []
		for movie in u:
			if movie in v:
				intersection.append(movie)
		if len(intersection) == 0:
			return 0
		#calculate numerator

		movies_u = []
		movies_v = []

		for movie in intersection:
			movies_u.append(u[movie])
			movies_v.append(v[movie])

		vector_u = np.asarray(movies_u)
		vector_v = np.asarray(movies_v)

		dot_product = np.dot(vector_u, vector_v)

		norm_u = np.linalg.norm(vector_u)
		norm_v = np.linalg.norm(vector_v)

		return float(dot_product)/(norm_u*norm_v)

