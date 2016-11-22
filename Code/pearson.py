import sys
import math
from predictor import Predictor

class Pearson(Predictor):
	def __init__(self):
		self.user_data = None

	def train(self, user_data):
		self.user_data = user_data

	def predict(self, user):
		#TODO
		pass
