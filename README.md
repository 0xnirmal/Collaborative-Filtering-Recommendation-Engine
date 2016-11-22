#Collaborative Filtering Recommendation Engine#

This project is a survey of collaborative filtering techniques and their effectiveness in a movie recommendation system. Similarity metrics we are surveying include:
* Pearson Correlation Coefficient
* Spearman Rank Correlation Coefficient
* Mean-Squared Distance
* Cosine Similarity

For a detailed understanding of the study, see the project proposal attached to this repo.

##Things To Do##
- [x] Establish Data Pipeline
- [x] Setup prediction/testing environment
- [ ] Write evaluation script
- [ ] Write/test Pearson Correlation Coefficient
- [ ] Write/test Spearman Rank Correlation Coefficient
- [ ] Write/test Mean-Squared Distance
- [ ] Write/test Cosine Similarity
- [ ] Write final analysis

##Running the Code##
All of these commands should be issued from the main directory of the repository.
##Setup##
pip install -r requirements.txt

##Training##
python Code/runner.py --mode train --algorithm insert_algorithm_here --model-file algorithm's_name.model --data Data/ratings.csv

##Predicting###
python Code/runner.py --mode test --algorithm insert_algorithm_here --model-file algorithm's_name.model --data Data/ratings.csv --predictions-file insert_algorithm_here.predictions

##Testing##
TBD (Evaluation Script needs to be written)

