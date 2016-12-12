#this code is a modified version of the classify.py we were provided for our other homework assigments

import os
import argparse
import sys
import pickle
from pearson import Pearson
from spearman import Spearman
from predictor import Predictor
from meansquared import Meansquared
from cosine import Cosine

args = None

def load_data(filename, exclusion = False):
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
            if exclusion:
                if len(user) < 10:
                    user[movie_id] = rating
            else:
                user[movie_id] = rating
    return users

def load_prediction_data(filename):
    training_data = load_data(filename, True)
    prediction_data = {}
    with open(filename) as reader:
        #skip first line
        next(reader)
        for line in reader:
            if len(line.strip()) == 0:
                continue
            # Divide the line into user, movieid, and rating
            split_line = line.split(",")
            user = int(split_line[0])
            if user not in prediction_data:
                prediction_data[user] = {}
            movie_id = int(split_line[1])
            rating = float(split_line[2])
            if movie_id not in training_data[user] and len(prediction_data[user]) < 10:
                prediction_data[user][movie_id] = rating
    return prediction_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="Data file.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--num-neighbors", type=int, help="The number of similar neighbors considered", default=5)
    args = parser.parse_args()
    return args

def train(users, algorithm):
    if algorithm == 'pearson':
        model = Pearson()
        model.train(users)
        return model
    elif algorithm == 'spearman':
        model = Spearman()
        model.train(users)
        return model
    elif algorithm == 'mean_squared':
        model = Meansquared()
        model.train(users)
        return model
    elif algorithm == "cosine":
        model = Cosine()
        model.train(users)
        return model
    else:
        print("No model found given algorithm " + algorithm)
        sys.exit(-1)

def write_predictions(predictor, prediction_data, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for user in prediction_data:
                labels = predictor.predict(user, prediction_data[user], args.num_neighbors)
                for key in labels:
                    writer.write(str(labels[key]))
                    writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    global args
    args = get_args()
    if args.mode.lower() == "train":
        # Load the training data.
        training_set = load_data(args.data)
        # Train the model.
        predictor = train(training_set, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif args.mode.lower() == "test":
        # Load the test data.
        test_data = load_prediction_data(args.data)
        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
        write_predictions(predictor, test_data, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()
