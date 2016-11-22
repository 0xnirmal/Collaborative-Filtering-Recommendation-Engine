import os
import argparse
import sys
import pickle
from pearson import Pearson
from predictor import Predictor

args = None

def load_data(filename):
    users = {}
    with open(filename) as reader:
    	#skip first line
    	next(reader)
        for line in reader:
            if len(line.strip()) == 0:
                continue
            # Divide the line into user, movieid, and rating
            split_line = line.split(",")
            user = int(split_line[0])
            if user not in users:
            	users[user] = {}
            user = users[user]
            movie_id = int(split_line[1])
            rating = float(split_line[2])
            user[movie_id] = rating
    return users

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="Data file.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    args = parser.parse_args()
    return args

def train(users, algorithm):
	if algorithm == 'pearson':
		model = Pearson()
		model.train(users)
		return model
	elif algorithm == 'spearmen':
		return None
	elif algorithm == 'mean_squared':
		return None
	elif algorithm == "cosine_similarity":
		return None
	else:
		print("No model found given algorithm " + algorithm)
        sys.exit(-1)

def main():
    global args
    args = get_args()
    if args.mode.lower() == "train":
        # Load the training data.
        users = load_data(args.data)
        # Train the model.
        predictor = train(users, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif args.mode.lower() == "test":
        # Load the test data.
        users = load_data(args.data)
        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
	main()
