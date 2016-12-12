import sys
import math
from predictor import Predictor, compute_weighted_average, compute_mean, clean_user
import collections
import math
from heapq import heappush, heappop

class Spearman(Predictor):
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
				similarity = self.calculate_spearman_similarity(user_post_clean, self.user_data[user_v])
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

	def calculate_spearman_similarity(self, u, v):
		#generate intersection subset
		intersection = []
		for movie in u:
			if movie in v:
				intersection.append(movie)
		if len(intersection) == 0:
			return 0

		movie_ranks_u = {}
		movie_ranks_v = {}

		index = 0.5
		while index <= 5.0:
			movie_ranks_u[index] = 0
			movie_ranks_v[index] = 0
			index += 0.5

		for movie in intersection:
			movie_ranks_u[u[movie]] += 1
			movie_ranks_v[v[movie]] += 1

		list_of_u_ranks = []
		list_of_v_ranks = []
		starting_rank_u = 1.0
		starting_rank_v = 1.0

		index = 5.0
		while index >= 0.5:
			in_block = movie_ranks_u[index]
			if in_block != 0:
				ending_rank = starting_rank_u + in_block
				avg_rank = (starting_rank_u + ending_rank) / in_block
				for i in range(in_block):
					list_of_u_ranks.append(avg_rank)
				starting_rank_u += in_block
			index -= 0.5

		index = 5.0
		while index >= 0.5:
			in_block = movie_ranks_v[index]
			if in_block != 0:
				ending_rank = starting_rank_v + in_block
				avg_rank = (starting_rank_v + ending_rank) / in_block
				for i in range(in_block):
					list_of_v_ranks.append(avg_rank)
				starting_rank_v += in_block
			index -= 0.5

		mean_u = compute_mean_spearman(list_of_u_ranks)
		mean_v = compute_mean_spearman(list_of_v_ranks)

		numerator = 0.0
		denominator = 0.0
		#compute numerator
		for i in range(len(intersection)):
			numerator += (list_of_u_ranks[i] - mean_u) * (list_of_v_ranks[i] - mean_v)

		#computer denominator
		term1 = 0.0
		term2 = 0.0
		for i in range(len(intersection)):
			term1 += math.pow(list_of_u_ranks[i] - mean_u, 2)
			term2 += math.pow(list_of_v_ranks[i] - mean_v, 2)
		denominator = math.sqrt(term1 * term2)

		if denominator == 0:
			return 0.0
		return numerator / denominator

def compute_mean_spearman(list_of_u_ranks):
	sum_u = 0.0
	for rank in list_of_u_ranks:
		sum_u += rank
	return sum_u / len(list_of_u_ranks)

