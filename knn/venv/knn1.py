import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from collections import Counter
from queue import PriorityQueue
import sklearn.neighbors


class KNNClassifier:

    def __init__(self, X, y, k=5):
        """
        Initialize our custom KNN classifier
        PARAMETERS
        X - our training data features
        y - our training data answers
        k - the number of nearest neighbors to consider for classification
        """
        self._model = sklearn.neighbors.BallTree(X)
        self._y = y
        self._k = k
        self._x = X
        self._counts = self.getCounts()

    def getCounts(self):
        """
        Creates a dictionary storing the counts of each answer class found in y
        RETURNS
        counts - a dictionary of counts of answer classes
        """

        # BEGIN Workspace 1.1
        unique, count = np.unique(self._y, return_counts=True)
        counts = dict(zip(unique, count))
        # END Workspace 1.1
        return counts

    def majority(self, indices):
        """
        Given indices, report the majority label of those points.
        For a tie, report the most common label in the data set.
        PARAMETERS
        indices - an np.array, where each element is an index of a neighbor
        RETURNS
        label - the majority label of our neighbors
        """
        count_neighbor = []
        # BEGIN Workspace 1.2
        for item in indices:
            count_neighbor.append(self._y[item])
        counter = Counter(count_neighbor)
        most_common = [counter.most_common()[0][0]]
        higher = counter.most_common(0)[0][1]
        i = 1
        for item in counter.most_common():
            if counter.most_common()[i][1] == higher:
                most_common.append(counter.most_common()[i][0])
                i += 1
            else:
                break
        total_counts = self.getCounts()
        label = most_common[0]
        highest = total_counts[most_common[0]]
        j = 1
        for item in most_common:
            if total_counts[most_common[j]] > highest:
                label = most_common[j]
        # END Workspace 1.2
        return label

    def classify(self, given_point):
        """
        Given a new data point, classify it according to the training data X and our number of neighbors k into the appropriate class in our training answers y
        PARAMETERS
        point - a feature vector of our test point
        RETURNS
        ans - our predicted classification
        """
        # BEGIN Workspace 1.3
        print(given_point)
        q = PriorityQueue(self._k)
        for index, point in enumerate(self._x):
            t = 0
            for i in range(len(given_point)):
                t += (point[i] - given_point[i]) ** 2

            if not q.full():
                q.put((-t, index))
            else:
                tmp = q.get()
                if -tmp[0] < t:
                    q.put(tmp)
                else:
                    q.put((-t, index))
        res = []
        while not q.empty():
            tp = q.get()
            res.append(tp[1])

        ans = self.majority(res)
        # END Workspace 1.3
        return ans

    def confusionMatrix(self, testX, testY):
        """
        Generate a confusion matrix for the given test set
        PARAMETERS
        testX - an np.array of feature vectors of test points
        testY - the corresponding correct classifications of our test set
        RETURN
        C - an N*N np.array of counts, where N is the number of classes in our classifier
        """
        C = np.zeros((len(self._counts), len(self._counts)), dtype=int)
        # BEGIN Workspace 1.4
        i = 0
        class_number = {}
        for key, value in self._counts.items():
            class_number[key] = i
            i += 1
        i = 0
        for x in testX:
            prediction = self.classify(x)
            C[class_number[prediction]][class_number[testY[i]]] += 1
            i += 1
        # END Workspace 1.4
        return C

    def accuracy(self, C):
        """
        Generate an accuracy score for the classifier based on the confusion matrix
        PARAMETERS
        C - an np.array of counts
        RETURN
        score - an accuracy score
        """
        score = np.sum(C.diagonal()) / C.sum()
        return score


import unittest


class KNNTester(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[3, 1], [2, 8], [2, 7], [5, 2], [3, 2], [8, 2], [2, 4]])
        self.y = np.array([[1, -1, -1, 1, -1, 1, -1]])
        self.knnfive = KNNClassifier(self.x, self.y)
        self.knnthree = KNNClassifier(self.x, self.y, 3)
        self.knnone = KNNClassifier(self.x, self.y, 1)

        self.testPoints = np.array([[2, 7], [2, 6], [4, 4]])

    def testCounter(self):
        """
        Test getCounts function from knnclassifier
        """
        self.assertEqual(self.knnfive._counts[1], 3)
        self.assertEqual(self.knnfive._counts[-1], 4)

    def testKNNOne(self):
        """
        Test if the classifier returns "correct" (expected) classifications for k = 1
        """
        self.assertEqual(self.knnone.classify(self.testPoints[0]), 1)
        # BEGIN Workspace
        # Add more tests as needed
        # END Workspace

    # BEGIN Workspace
    # Add more test functions as requested
    # HINT - You'll want to make sure your
    # END Workspace


tests = KNNTester()
myTests = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner().run(myTests)