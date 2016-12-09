import sys
import math
from predictor import Predictor
import collections

class Pearson(Predictor):
	def __init__(self):
		self.user_data = None

	def train(self, user_data):
		self.user_data = user_data

	def predict(self, user_num, user_set):
		#TODO
		#Make sure you hide the training data on the values in the user_set
		label_set = {}
		od = collections.OrderedDict(sorted(user_set.items()))
		for key, value in (od.items()):
			label_set[key] = value
		return collections.OrderedDict(sorted(label_set.items()))
