# Collaborative Filtering Recommendation Engine #

This project is a survey of neighborhood-based collaborative filtering techniques and their effectiveness in a movie recommendation system. Similarity metrics we are surveying include:
* Pearson Correlation Coefficient
* Spearman Rank Correlation Coefficient
* Mean-Squared Distance
* Cosine Similarity

For a detailed understanding of the study, see the writeup attached to this repo.

## Things To Do ##
- [x] Establish Data Pipeline
- [x] Setup prediction/testing environment
- [x] Write evaluation script
- [x] Write/test Pearson Correlation Coefficient
- [x] Write/test Spearman Rank Correlation Coefficient
- [x] Write/test Mean-Squared Distance
- [x] Write/test Cosine Similarity
- [x] Write final analysis

## Running the Code ##
All of these commands should be issued from the main directory of the repository.
### Setup ###
pip install -r requirements.txt

### Training ###
python Code/runner.py --mode train --algorithm insert_algorithm_here --model-file algorithm's_name.model --data Data/ratings.csv

### Predicting ###
python Code/runner.py --mode test --algorithm insert_algorithm_here --model-file algorithm's_name.model --num-neighbors default_is_five --data Data/ratings.csv --predictions-file insert_algorithm_here.predictions

### Evaluation ###
python Code/eval.py Data/ratings.csv algorithm's_name.predictions

### Running Testing Suite ###
python test.py

### Algorithm Names (for command line arguments) ###
* pearson
* spearman
* mean_squared
* cosine

