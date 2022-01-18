# MACHINE LEARNING GROUP-40 ASSIGNMENT-1

#  Neha Dalmia 19CS30055

#  Hritaban Ghosh 19CS30053

# Import the required libraries

import numpy as np
from collections import deque
import copy
import math
from graphviz import Digraph


# Accuracy


def accuracy(y_test, y_predict):
    """
    The accuracy function takes in two input parameters y_test and y_predict
    and returns the accuracy metric of y_predict over y_test.

    Inputs:
        'y_test': The ground truth of the classification values (Observed Values) - Pandas DataFrame

        'y_predict': The classification values obtained using the prediction model (Predicted Values) - List

    Outputs:
        'accuracy': The accuracy metric of y_predict over y_test
    """
    # Convert the y_test dataframe object into a list
    y_test_t = list(y_test.values)

    # Create an array which marks 1 if there is a mis-match else marks 0
    difference = np.array(list(map(lambda x, y: x != y, y_predict, y_test_t)))

    # Calculate the accuracy metric of y_predict over y_test.
    accuracy_metric = (difference.size - np.sum(difference)) / difference.size

    return accuracy_metric


def freqs(arr):
    """
    The freqs function takes in one input parameter arr which is a list of target values
    and returns a two arrays, one corresponding to the unique values in
    arr and the other corresponding to the count of each unique value.

    Inputs:
        'arr' - arr is a list of target classes like ['A', 'B', 'I', 'A', ...]

    Outputs:
        'unique, counts' - two arrays, 'unique' corresponding to the unique values in
    arr and 'counts' corresponding to the count of each unique value.
    """
    # Convert the list into an array
    x = np.array(arr)

    # Apply numpy's unique function with return_counts = True to obtain the two arrays
    unique, counts = np.unique(x, return_counts=True)

    return unique, counts


# Gini Function


def gini(y):
    """
    The gini function takes in an input parameter y which is a list of target values
    and returns the gini index of the list.
    Gini Impurity tells us what is the probability of mis-classifying an observation.

    Inputs:
        'y': y is a list of target classes like ['A', 'B', 'I', 'A', ...]

    Outputs:
        'gini': The gini index of the list
    """
    # Get the value and corresponding frequency of various classes in y
    vals, frequencies = freqs(y)

    # sum_of_freq is the total number of values in y
    sum_of_freq = np.sum(frequencies)

    # Compute the gini index
    gini_index = 0  # To store the gini index
    for x in frequencies:
        gini_index += (x / sum_of_freq) * (1 - x / sum_of_freq)

    return gini_index


def gini_attribute(X, y, attribute, par_gini, split=2):
    """
    The gini_attribute function takes in five input parameters - X, y, attribute,
    par_gini and split and returns the best possible split that can be obtained
    on that particular attribute and the corresponding gain and threshold of the split.

    'X': X is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.

    'y': y is a indexed pandas dataframe consisting of target values in a column.

    'attribute': attribute is a string containing the name of the column for which we want to find the best split.

    'par_gini': Gini Index of the y(All Target Values).

    'split': This parameter has been defaulted to 2. It indicates the number of subsets a attribute column can be split into.

    NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
            For all purposes in this algorithm a node will be split into two and two nodes only.
    """
    # indices contain the index values of X dataframe
    indices = X.index

    # attribute_data stores the [index, attribute value] for all indices
    attribute_data = [[i, X[attribute][i]] for i in indices]

    # Sort the attribute_data based on attribute values
    attribute_data = sorted(attribute_data, key=lambda x: x[1])

    # Initialise gain obtained on splitting the attribute column to be 0
    gain_attr = 0

    # Total number of indices(Total number of samples in X)
    n = len(indices)
    leftlist = []  # List of index values in left split
    rightlist = deque([attribute_data[i][0] for i in range(0, n)])  # Deque of index values in right split
    leftsplit = []  # List of index values in left split
    rightsplit = []  # List of index values in right split

    # The for loop starts from [1] and [n-1] split and keeps updating leftsplit and rightsplit whenever a better split is found.
    # The goal of this loop is to find a split with the maximum gini gain
    for i in range(0, n - 1):
        # Update the lists
        leftlist.append(attribute_data[i][0])
        rightlist.popleft()

        # If the target values do not flip at this split do not consider it to be a candidate threshold
        if y[leftlist[-1]] == y[rightlist[0]]:
            continue

        # Compute the gini values for the left and right half of the target values
        leftval = gini(y.loc[list(leftlist)].values)
        rightval = gini(y.loc[list(rightlist)].values)

        # value is a temporary which stores the gain obtained on the current split
        value = par_gini - leftval * ((i + 1) / n) - rightval * ((n - i - 1) / n)

        # Update the gain_attr and splits if and only if a better split is obtained in terms of maximizing gain
        if value >= gain_attr:
            gain_attr = copy.deepcopy(value)
            leftsplit = copy.deepcopy(leftlist)
            rightsplit = copy.deepcopy(rightlist)

    # Calculate the threshold for the best split obtained.
    # Threshold is the mean of the greatest value in the left split and the least value in the right split
    threshold = (X.loc[leftsplit[-1]][attribute] + X.loc[rightsplit[0]][attribute]) / 2

    return gain_attr, leftsplit, rightsplit, threshold


# Information Gain


def entropy(y):
    """
    The entropy function takes in an input parameter y which is a list of target values
    and returns the entropy of the list.
    Entropy, is a impurity measure of the randomness in the information being processed.
    The higher the entropy, the harder it is to draw any conclusions from that information.

    Inputs:
        'y': y is a list of target classes like ['A', 'B', 'I', 'A', ...]

    Outputs:
        'entropy_val': entropy_val is the measure of impurity of y.
    """
    # Get the value and corresponding frequency of various classes in y
    vals, frequencies = freqs(y)

    # sum_of_freq is the total number of values in y
    sum_of_freq = np.sum(frequencies)

    # Compute the entropy of the list of target values
    entropy_val = 0  # To store the entropy value
    for x in frequencies:
        entropy_val -= (x / sum_of_freq) * math.log2(x / sum_of_freq)

    return entropy_val


def igain_attribute(X, y, attribute, par_entropy, split=2):
    """
    The igain_attribute function takes in five input parameters - X, y, attribute,
    par_entropy and split and returns the best possible split that can be obtained
    on that particular attribute and the corresponding gain and threshold of the split.

    'X': X is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.

    'y': y is a indexed pandas dataframe consisting of target values in a column.

    'attribute': attribute is a string containing the name of the column for which we want to find the best split.

    'par_entropy': Entropy of y(All Target Values).

    'split': This parameter has been defaulted to 2. It indicates the number of subsets a attribute column can be split into.

    NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
            For all purposes in this algorithm a node will be split into two and two nodes only.
    """
    # indices contain the index values of X dataframe
    indices = X.index

    # attribute_data stores the [index, attribute value] for all indices
    attribute_data = [[i, X[attribute][i]] for i in indices]

    # Sort the attribute_data based on attribute values
    attribute_data = sorted(attribute_data, key=lambda x: x[1])

    # Initialise gain obtained on splitting the attribute column to be 0
    gain_attr = 0

    # Total number of indices(Total number of samples in X)
    n = len(indices)
    leftlist = []  # List of index values in left split
    rightlist = deque([attribute_data[i][0] for i in range(0, n)])  # Deque of index values in right split
    leftsplit = []  # List of index values in left split
    rightsplit = []  # List of index values in right split

    # The for loop starts from [1] and [n-1] split and keeps updating leftsplit and rightsplit whenever a better split is found.
    # The goal of this loop is to find a split with the maximum information gain
    for i in range(0, n - 1):
        # Update the lists
        leftlist.append(attribute_data[i][0])
        rightlist.popleft()

        # If the target values do not flip at this split do not consider it to be a candidate threshold
        if y[leftlist[-1]] == y[rightlist[0]]:
            continue

        # Compute the entropy values for the left and right half of the target values
        leftval = entropy(y.loc[list(leftlist)].values)
        rightval = entropy(y.loc[list(rightlist)].values)

        # value is a temporary which stores the information gain obtained on the current split
        value = par_entropy - leftval * ((i + 1) / n) - rightval * ((n - i - 1) / n)

        # Update the gain_attr and splits if and only if a better split is obtained in terms of maximizing gain
        if value >= gain_attr:
            gain_attr = copy.deepcopy(value)
            leftsplit = copy.deepcopy(leftlist)
            rightsplit = copy.deepcopy(rightlist)

    # Calculate the threshold for the best split obtained.
    # Threshold is the mean of the greatest value in the left split and the least value in the right split
    threshold = (X.loc[leftsplit[-1]][attribute] + X.loc[rightsplit[0]][attribute]) / 2

    return gain_attr, leftsplit, rightsplit, threshold


# Printing the Tree


def get(node):
    """
    This function returns a formatted string which will be used
    for printing the parameters inside the node when the graph
    is crated using graphviz.

    Inputs:
        'node': This is the node object that is to be checked for the leaf.
    """
    # Printing the decision attribute of the node if the node is not leaf
    if not node.is_leaf:
        return "node #{}, attribute ={}, vote={}".format(node.node_id, node.decision_attr, node.vote)
    return "node #{}, vote ={}".format(node.node_id, node.vote)


def print_decision_tree(dtree, filepath=""):
    """
    This function prints the decision tree graph so created
    and saves the output in the a pdf file.

    Inputs:
        'dtree':  Root node of the decision tree for which the the graph needs top be printed.
    """
    # create a new Digraph
    f = Digraph('Decision Tree', filename='decision_tree.gv')
    f.attr(rankdir='LR', size='1000,500')

    # border of the nodes is set to rectangle shape
    f.attr('node', shape='rectangle')

    if dtree.root.is_leaf:
        # If the root is a leaf, then print the node alone
        f.node("node #{} , vote ={}".format(dtree.root.node_id, dtree.root.vote))
    else:
        # Do a breadth first search and add all the edges in the output graph
        q = [dtree.root]  # queue for the breadth first search
        while len(q) > 0:
            node = q.pop(0)
            if not node.is_leaf:
                # If the node has children then add the edges from this node to to it's children
                if len(node.children) > 0:
                    f.edge(get(node), get(node.children[0]), label='<=' + str(node.threshold))
                    q.append(node.children[0])
                if len(node.children) > 1:
                    f.edge(get(node), get(node.children[1]), label='>' + str(node.threshold))
                    q.append(node.children[1])

    f.render(f'{filepath}decision_tree.gv', view=True)
