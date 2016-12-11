from abc import ABCMeta, abstractmethod
import math

def compute_weighted_average(u, similarity_to_user, movie, num_neighbors):
	mean_u = compute_mean(u)
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
		mean_user = compute_mean(user)
		numerator += similarity * user[movie]
		denominator += math.fabs(similarity)

	if denominator == 0:
		return 0

	return numerator / denominator

def compute_mean(u):
	sum_u = 0.0
	for movie in u:
		sum_u += u[movie]
	return sum_u / len(u)

def clean_user(user_pre_clean, user_set):
	user_post_clean = {}
	for movie in user_pre_clean:
		if movie not in user_set:
			user_post_clean[movie] = user_pre_clean[movie]
	return user_post_clean

# abstract base class for defining predictors
class Predictor:
	__metaclass__ = ABCMeta

	@abstractmethod
	def train(self, instances): pass

	@abstractmethod
	def predict(self, instance): pass

