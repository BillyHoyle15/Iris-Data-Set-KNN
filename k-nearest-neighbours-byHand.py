# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:56:47 2020

@author: Suat
"""


from collections import Counter
import requests
from typing import Dict,TypeVar, Tuple, NamedTuple, List
import csv
from collections import defaultdict
import random
import math

#1 Creating a Vector
Vector = List[float]
"""------------------------------------------------------------------------"""

#2: vector operations for calculating euclidean distance, we use this as 
#calculating distance between two iris data
def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be same length"
    
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))
"""------------------------------------------------------------------------"""
#3 Splitting function for train and test datas
X = TypeVar('X')  # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.


"""------------------------------------------------------------------------"""

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'


def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    
    winners = []
    for count in vote_counts.values():
        if count == winner_count:
            winners.append(count)
    num_winners = len(winners)

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest

# Tie, so look at first 4, then 'b'
assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'


class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:

    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point, new_point))

    # Find the labels for the k closest
    k_nearest_labels = []
    for lp in by_distance[:k]:
        k_nearest_labels.append(lp.label)

    # and let them vote.
    return majority_vote(k_nearest_labels)


data = requests.get(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

with open('iris.dat', 'w') as f:
    f.write(data.text)


def parse_iris_row(row: List[str]) -> LabeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = []
    for value in row[:-1]:
        measurements.append(float(value))
        
    # class is e.g. "Iris-virginica"; we just want "virginica"
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)

with open('iris.data') as f:
    reader = csv.reader(f)
    iris_data = []
    for row in reader:
#        iris_data.append(parse_iris_row(row))
# couldn't just use this becasue iris_data has one extra emty list at the end
# of the list for me. If you print iris_data you'll notice. So I get rid of it.
        iris_data.append(row)
    new_iris_data = iris_data[:-1] #last array of the iris data is empty
    iris_data_2 = []
    for flower in new_iris_data:
        iris_data_2.append(parse_iris_row(flower))
            
#To start with, let’s split the data into a test set and a training set:
random.seed(12)
iris_train, iris_test = split_data(iris_data_2, 0.70)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150 

"""
The training set will be the “neighbors” that we’ll use to classify the 
points in the test set. We just need to choose a value for k, the number 
of neighbors who get to vote. Too small (think k = 1), and we let outliers 
have too much influence; too large (think k = 105), and we just predict the 
most common class in the dataset.

In a real application (and with more data), we might create a separate 
validation set and use it to choose k. Here we’ll just use k = 5:
"""

# track how many times we see (predicted, actual)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)