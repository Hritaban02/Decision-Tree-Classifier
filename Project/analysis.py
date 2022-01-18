# MACHINE LEARNING GROUP-40 ASSIGNMENT-1

#  Neha Dalmia 19CS30055

#  Hritaban Ghosh 19CS30053

# Import the required libraries
# Import the required libraries for manipulating datasets and visualizing the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from models import DecisionTreeClassifier
from utility import accuracy, print_decision_tree

# Import the Dataset

# Loading the data in a dataframe

# The Avila data set consists of avila_tr.txt (a training set containing 10430 samples) and avila_ts.txt (a test set containing the 10437 samples) 
# We were instructed to combine these two sets and then perform a 80/20 split of the combined set into training and test sets respectively.
# Therefore the new combined set named avila_combines.txt consists of 20867 samples

# Load the avila_combined.txt into a pandas dataframe
# The txt file stores the data in a comma separated format and contains no information about the columns
df = pd.read_csv('avila_combined.txt', delimiter=",", header=None)

# Rename the Columns Appropriately

# Renaming the columns according to the following information

# Attribute Information:

# F1: intercolumnar distance
# F2: upper margin
# F3: lower margin
# F4: exploitation
# F5: row number
# F6: modular ratio
# F7: interlinear spacing
# F8: weight
# F9: peak number
# F10: modular ratio/ interlinear spacing
# Class: A, B, C, D, E, F, G, H, I, W, X, Y

df.columns = ['intercolumnar distance', 'upper margin', 'lower margin', 'exploitation', 'row number', 'modular ratio',
              'interlinear spacing',
              'weight', 'peak number', 'modular ratio/ interlinear spacing', 'Class']

# Check for Null Values (Missing Data)

# Checking for null value in the columns

df.isnull().sum()
# No null value found


# Prompt the user for metric
user_metric = input("Enter the metric you want to use for impurity measure(gini or entropy): ")
print("You have chosen: ", user_metric)

# Query the user on reusing attributes
user_reuse_attribute = input("Enter the whether or not you want to reuse attributes at different levels in the tree(True or False): ")
print("You have chosen: ", user_reuse_attribute)

# Prompt the user for max_depth
user_max_depth = int(input("Enter the maximum depth up to which you want the tree to grow(integer): "))
print("You have entered: ", user_max_depth)

# Analysis for the best model over 10 different random train, test, validation splits
# Using gini as the measure of impurity


X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

final_model = None
max_test_accuracy = 0.0
final_X_test = None
final_y_test = None
final_X_valid = None
final_y_valid = None
final_X_train = None
final_y_train = None

accuracy_sum = 0.0

for i in range(0, 10):
    X_sub, X_test, y_sub, y_test = train_test_split(X, y, test_size=0.2, random_state=i + 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_sub, y_sub, test_size=0.25, random_state=i + 1)
    print("***********************")
    print(f"Random split #{i + 1} ")
    model = DecisionTreeClassifier(reuse_attribute=user_reuse_attribute, max_depth=user_max_depth, metric=user_metric)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    ac = accuracy(y_test, y_predict)
    accuracy_sum += ac
    print(f"Accuracy = {ac * 100}%")
    print("***********************")
    if ac > max_test_accuracy:
        max_test_accuracy = ac
        final_model = copy.deepcopy(model)
        final_X_test = copy.deepcopy(X_test)
        final_y_test = copy.deepcopy(y_test)
        final_X_valid = copy.deepcopy(X_valid)
        final_y_valid = copy.deepcopy(y_valid)
        final_X_train = copy.deepcopy(X_train)
        final_y_train = copy.deepcopy(y_train)
accuracy_sum /= 10.0

# Printing the best decision tree(not pruned)
print_decision_tree(final_model, f"{user_metric}/")
with open(f"{user_metric}/decision_tree.gv") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

y_predict = final_model.predict(final_X_test)
ac = accuracy(final_y_test, y_predict)
print(f"Maximum Accuracy = {ac * 100}%, average accuracy = {accuracy_sum * 100}%")


# Confusion matrix in sklearn

# Get the final_y_test target values in list format
y_test_values = final_y_test.values

# Confusion matrix
classes = np.unique(np.array(y_test_values))
matrix = confusion_matrix(y_test_values, y_predict, labels=classes)
print('Confusion matrix : \n', matrix)

# Outcome values order in sklearn

# Classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_test_values, y_predict, labels=classes)
print('Classification report : \n', matrix)


# Analysis for Depth of Tree
# Using gini as the measure of impurity


train_accuracies = []
test_accuracies = []
val_accuracies = []
depths = []
nodes = []

for depth in range(1, user_max_depth+1):
    y_predict = final_model.predict(final_X_test, use_depth=depth)
    ac = accuracy(final_y_test, y_predict)
    test_accuracies.append(ac)
    y_predict = final_model.predict(final_X_train, use_depth=depth)
    ac = accuracy(final_y_train, y_predict)
    train_accuracies.append(ac)
    y_predict = final_model.predict(final_X_valid, use_depth=depth)
    ac = accuracy(final_y_valid, y_predict)
    val_accuracies.append(ac)
    depths.append(depth)
    if not len(nodes):
        nodes.append(final_model.node_at_depth_count[str(depth)])
    else:
        if str(depth) in final_model.node_at_depth_count.keys():
            nodes.append(nodes[-1] + final_model.node_at_depth_count[str(depth)])
        else:
            nodes.append(nodes[-1])

plt.plot(depths, train_accuracies, label="training accuracy")
plt.plot(depths, test_accuracies, label="test accuracy")
plt.plot(depths, val_accuracies, label="validation accuracy")
plt.legend()
plt.xlabel('Depth of the Decision Tree')
plt.ylabel('Accuracy')
plt.title(user_metric)
plt.savefig(f'{user_metric}/depth_analysis.png', dpi=300, bbox_inches='tight')
plt.clf()


# Analysis for the Number of Nodes in the Tree
# Using gini as the measure of impurity


plt.plot(nodes, train_accuracies, label="training accuracy")
plt.plot(nodes, test_accuracies, label="test accuracy")
plt.plot(nodes, val_accuracies, label="validation accuracy")
plt.legend()
plt.xlabel('Number of Nodes in the Decision Tree')
plt.ylabel('Accuracy')
plt.title(user_metric)
plt.savefig(f'{user_metric}/node_analysis.png', dpi=300, bbox_inches='tight')
plt.clf()


# Analysis of the status of tree during pruning
# Using gini as the measure of impurity


model_temp = copy.deepcopy(final_model)
a, b, c, d = model_temp.prune_node_vary((final_X_train, final_y_train), (final_X_test, final_y_test),
                                        (final_X_valid, final_y_valid))
plt.plot(d, a, label="training accuracy")
plt.plot(d, b, label="test accuracy")
plt.plot(d, c, label="validation accuracy")
plt.legend()
plt.xlabel('Number of Nodes in the Decision Tree After Pruning')
plt.ylabel('Accuracy')
plt.title(user_metric)
plt.savefig(f'{user_metric}/accuracy_analysis.png', dpi=300, bbox_inches='tight')
plt.clf()


# Pruning the tree

final_model.prune(evaluation_set=(final_X_valid, final_y_valid), stopping_rounds=3)

print("Number of nodes in the pruned tree: ", len(final_model.node_dict))

# Printing the best decision tree(pruned)
print_decision_tree(final_model, f"{user_metric}/pruned_")
with open(f"{user_metric}/pruned_decision_tree.gv") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

y_predict = final_model.predict(final_X_test)
ac = accuracy(final_y_test, y_predict)
print(f"Accuracy = {ac * 100}%")

# Confusion matrix in sklearn

# Get the final_y_test target values in list format
y_test_values = final_y_test.values

# Confusion matrix after pruning
classes = np.unique(np.array(y_test_values))
matrix = confusion_matrix(y_test_values, y_predict, labels=classes)
print('Confusion matrix : \n', matrix)

# Outcome values order in sklearn

# Classification report for precision, recall f1-score and accuracy after pruning
matrix = classification_report(y_test_values, y_predict, labels=classes)
print('Classification report : \n', matrix)

# # Visualize the data
#
# x_axis = sorted(df['Class'].unique())
# vc = (df['Class'].value_counts(sort=False))
# y_axis = [vc[i] for i in x_axis]
# plt.bar(x_axis, y_axis)
# plt.xlabel('Copier id')
# plt.ylabel('Frequency')
# plt.savefig('Scatter_Plots/accuracy_analysis.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
# # Inter columnar Distance
#
# plt.scatter(X["intercolumnar distance"], y)
# plt.savefig('Scatter_Plots/intercolumnar_distance.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
# # Upper margin
#
# plt.scatter(X["upper margin"], y)
# plt.savefig('Scatter_Plots/upper_margin.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Lower margin
#
# plt.scatter(X["lower margin"], y)
# plt.savefig('Scatter_Plots/lower_margin.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Exploitation
#
# plt.scatter(X["exploitation"], y)
# plt.savefig('Scatter_Plots/exploitation.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Row Number
#
# plt.scatter(X["row number"], y)
# plt.savefig('Scatter_Plots/row_number.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
# # Modular ratio
#
# plt.scatter(X["modular ratio"], y)
# plt.savefig('Scatter_Plots/modular_ratio.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Interlinear spacing
#
# plt.scatter(X["interlinear spacing"], y)
# plt.savefig('Scatter_Plots/interlinear_spacing.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Weight
#
# plt.scatter(X["weight"], y)
# plt.savefig('Scatter_Plots/weight.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
#
# # Peak number
#
# plt.scatter(X["peak number"], y)
# plt.savefig('Scatter_Plots/peak_number.png', dpi=300, bbox_inches='tight')
# plt.clf()
#
