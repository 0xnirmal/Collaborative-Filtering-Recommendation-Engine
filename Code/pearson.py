import sys
import math
from predictor import Predictor

class Pearson(Predictor):
	def __init__(self):
		self.user_data = None

	def train(self, user_data):
		self.user_data = user_data

	def predict(self, user_num, user_set):
		#TODO
		#Make sure you hide the training data on the values in the user_set
		label_set = {}
		for i in range(len(user_set)):
			label_set[i] = 0
		return label_set
