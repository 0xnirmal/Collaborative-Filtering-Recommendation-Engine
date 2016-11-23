#!/usr/bin/python

import sys
import math

if len(sys.argv) != 3:
    print 'usage: %s data predictions' % sys.argv[0]
    sys.exit()

data_file = sys.argv[1]
predictions_file = sys.argv[2]

data = open(data_file)
predictions = open(predictions_file)


true_labels = []
prev_user_id = -1
move_index_counter = 0

with open(data_file) as reader:
        next(reader)
	for line in reader:
		data_columns = line.split(",")
		current_user_id = data_columns[0]

		if current_user_id == prev_user_id:
			move_index_counter += 1

		else:
			prev_user_id = current_user_id
			move_index_counter = 0

		if move_index_counter >= 10 and move_index_counter < 20:
			true_labels.append(data_columns[2])

predicted_labels = []
for line in predictions:
    predicted_labels.append(line.strip())

if len(predicted_labels) != len(true_labels):
    print 'Number of lines in two files do not match.'
    sys.exit()

total = len(predicted_labels)
absolute_difference = 0

for i in range (len(predicted_labels)):
	absolute_difference += math.fabs(float(predicted_labels[i]) - float(true_labels[i]))

print 'Accuracy: %f' % ((float(absolute_difference)/float(total)))
