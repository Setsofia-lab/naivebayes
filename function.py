#Implement niave bayes from scratch on iris ds - a multiclass classification with 4 input features one class predicted
from math import exp, pi, sqrt


# Step 1: Separate By Class.
# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Test separating data by class
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]

# separated = separate_by_class(dataset)
# for label in separated:
# 	print(label)
# 	for row in separated[label]:
# 		print(row)

# Step 2: Summarize Dataset.
def mean(numbers):
	return sum(numbers) / float(len(numbers))


def stdev(numbers):
	avg = mean(numbers)
	variance = sum((x-avg) ** 2 for x in numbers) / len(numbers)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# summary = summarize_dataset(dataset)
# print(summary)

# devaitions = stdev(dataset[4])
# print(devaitions)

# Step 3: Summarize Data By Class.
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# summary = summarize_by_class(dataset)
# for label in summary:
# 	print(label)
# 	for row in summary[label]:
# 		print(row)

# Step 4: Gaussian Probability Density Function.

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Test Gaussian PDF
# print(calculate_probability(1.0, 1.0, 1.0))
# print(calculate_probability(2.0, 1.0, 1.0))
# print(calculate_probability(0.0, 1.0, 1.0))


# Step 5: Class Probabilities

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

summaries = summarize_by_class(dataset)
probabilities = calculate_class_probabilities(summaries, dataset[5])
print(probabilities)