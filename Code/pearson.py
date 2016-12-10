import sys
import math
from predictor import Predictor
import collections
import math
from heapq import heappush, heappop

class Pearson(Predictor):
	def __init__(self):
		self.user_data = None

	def train(self, user_data):
		self.user_data = user_data

	def predict(self, user_num, user_set, num_neighbors):
		#Make sure you hide the training data on the values in the user_set
		print(user_num)
		ordered_user_set = collections.OrderedDict(sorted(user_set.items()))
		prediction_set = {}
		user_pre_clean = self.user_data[user_num]
		user_post_clean = {}
		for movie in user_pre_clean:
			if movie not in user_set:
				user_post_clean[movie] = user_pre_clean[movie]
		#user_post_clean contains all the values for a user not in our prediction set
		similarity_to_user = []
		#generate relevant subset
		for user_v in self.user_data:
			# exclude current user
			if user_v != user_num:
				similarity = self.calculate_pearson_similarity(user_post_clean, self.user_data[user_v])
				similarity_to_user.append((similarity, self.user_data[user_v]))
		similarity_to_user.sort(key=lambda x: x[0], reverse=True)
		for movie in user_set:
			prediction = self.compute_weighted_average(user_post_clean, similarity_to_user, movie, num_neighbors)
			prediction_set[movie] = prediction

		return collections.OrderedDict(sorted(prediction_set.items()))

	def compute_weighted_average(self, u, similarity_to_user, movie, num_neighbors):
		mean_u = self.compute_mean(u)
		numerator = 0.0
		denominator = 0.0
		neighbors = []
		for similarity, user in similarity_to_user:
			if movie in user and len(neighbors) < num_neighbors:
				neighbors.append((similarity, user))
			if len(neighbors) >= num_neighbors:
				break
		#neighbors is a list of tuples similarity - user
		for i in range(len(neighbors)):
			similarity = neighbors[i][0]
			user = neighbors[i][1]
			mean_user = self.compute_mean(user)
			numerator += similarity * user[movie]
			denominator += math.fabs(similarity)

		if denominator == 0:
			return 0

		return numerator / denominator

	def calculate_pearson_similarity(self, u, v):
		mean_u = self.compute_mean(u)
		mean_v = self.compute_mean(v)
		#generate intersection subset
		intersection = []
		for movie in u:
			if movie in v:
				intersection.append(movie)
		if len(intersection) == 0:
			return 0
		#calculate numerator
		numerator = 0.0
		for movie in intersection:
			numerator += (u[movie] - mean_u) * (v[movie] - mean_v)
		#calulate denominator
		denominator = 0.0
		radical1 = 0.0
		radical2 = 0.0
		for movie in intersection:
			radical1 += math.pow(u[movie] - mean_u, 2)
			radical2 += math.pow(v[movie] - mean_v, 2)
		radical1 = math.sqrt(radical1)
		radical2 = math.sqrt(radical2)
		denominator = radical1 * radical2
		if denominator == 0:
			return 0.0
		return numerator / denominator

	def compute_mean(self, u):
		sum_u = 0.0
		for movie in u:
			sum_u += u[movie]
		return sum_u / len(u)





