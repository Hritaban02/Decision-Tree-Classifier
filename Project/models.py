import numpy as np
import copy
from utility import freqs, gini, gini_attribute, entropy, igain_attribute, accuracy

# Node Class


class node:
    # class variable node_id_number keeps track of the id of the current node
    # It is incremented each time a new node is constructed and re-initialized to 1 for a new decision tree
    node_id_number = 1

    def __init__(self, X, y, attributes_left, depth_of_parent, max_depth, metric="gini", split=2):
        """
        The __init__ method of the node class takes in seven input parameters - X, y, attributes_left,
        depth_of_parent, max_depth, metric and split and constructs the node.

        Inputs:
            'X': X is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.

            'y': y is a indexed pandas dataframe consisting of target values in a column.

            'attributes_left': It is a list of attributes left(strings) to consider when making a decision at this node.

            'depth_of_parent': It is the depth level of the node's parent in the tree. If the node being created is root, then depth_of_parent = 0.

            'max_depth': max_depth is the restriction set on the maximum depth upto which the tree can grow.

            'metric': The metric to be used when comparing different attributes to be chosen for decision at this node.

            'split': This parameter has been defaulted to 2. It indicates the number of subsets a attribute column can be split into.

            NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
                    For all purposes in this algorithm a node will be split into two and two nodes only.

        Attributes:
            'node_id': node_id stores the ID of the node being created in the tree.

            'attr': attr stores the list of attributes that are to be considered for making a decision at this node.

            'metric':   metric is the impurity measure to be evaluated for splitting the sample instances at this node.
                        It takes values 'gini' or 'entropy' and is defaulted to 'gini'.

            'split': split is the maximum number of children a node can have.

            NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
                    For all purposes in this algorithm a node will be split into two and two nodes only.

            'gain': The gain in purity obtained after choosing an attribute for decision. Initialised to 0.

            'children': List of child nodes to this node. Initialised to empty list.

            'metricVal': The value of the metric for the subset of data at this . Initialised to 0.

            'left_split':   The list of indexes corresponding to the sample instances of the dataset that is to be sent to the left child of this node.
                            Initialised to None.

            'right_split':  The list of indexes corresponding to the sample instances of the dataset that is to be sent to the right child of this node.
                            Initialised to None.

            'decision_attr':The attribute on which a decision is to be made whether to take the left child or the right child as we go down the tree.
                            Initialised to None.

            'threshold':    The threshold value for comparison with attribute value for making the decision.
                            Threshold is the mean of the greatest value in the left split and the least value in the right split.
                            Initialised to None.

            'is_leaf': Boolean value which indicates whether or not the node is a leaf.

            'depth': It is the depth level of this node in the tree. For root, depth = 1.

            'vote': The classification vote of this node if it would have been a child. The target value which has the maximum frequency.
        """
        self.node_id = node.node_id_number
        node.node_id_number += 1
        self.attr = attributes_left
        self.metric = metric
        self.split = split
        self.gain = 0
        self.children = []
        self.metricVal = 0
        self.left_split = None
        self.right_split = None
        self.decision_attr = None
        self.threshold = None
        self.is_leaf = False
        self.depth = depth_of_parent+1

        # Get the unique values in y and the count of each unique value.
        vals, frequencies = freqs(y.values)
        self.vote = vals[np.argmax(frequencies)]

        # The node is a leaf node if max_depth is reached (or)
        # the list of target values is pure (or)
        # The attribute list is empty (We have looked at every attribute) (or)
        # There are no samples in the dataset.

        # If node is not leaf then consider it for making a decision, else the node is a leaf.
        if self.depth < max_depth and len(vals) > 1 and len(attributes_left) > 0 and len(X.index) > 0:
            # Use metric 'gini' for purifying nodes
            if metric == "gini":
                # Evaluate the impurity measure of the list of target values
                self.metricVal = gini(y.values)

                # Initialise gain
                gain = -1

                left_split = X.index  # Initialise left_split to indices of the complete dataset at this node
                right_split = []  # Initialise right_split to empty list

                # For every attribute left to consider, choose the one that gives the maximum gain
                for attr in attributes_left:
                    # Calculate maximum gain obtainable for this attribute, with corresponding left and right splits along with the threshold using the gini_attribute function
                    gain_new, left_split_new, right_split_new, threshold = gini_attribute(X, y, attr, self.metricVal)

                    # If gain obtained at this attribute is more than the current gain
                    # then update the gain value, decision attribute, left_split, right_split and threshold value.
                    if gain_new > gain:
                        gain = gain_new
                        self.decision_attr = attr
                        left_split = left_split_new
                        right_split = right_split_new
                        self.threshold = threshold

                # Set the gain, left_split and right_split to values corresponding to the attribute which gives maximum gain
                self.gain = gain
                self.left_split = left_split
                self.right_split = right_split

            # Use metric 'entropy' for purifying nodes
            if metric == "entropy":
                # Evaluate the impurity measure of the list of target values
                self.metricVal = entropy(y.values)

                # Initialise gain
                gain = -1

                left_split = X.index  # Initialise left_split to indices of the complete dataset at this node
                right_split = []  # Initialise right_split to empty list

                # For every attribute left to consider, choose the one that gives the maximum gain
                for attr in attributes_left:
                    # Calculate maximum gain obtainable for this attribute, with corresponding left and right splits along with the threshold using the igain_attribute function
                    gain_new, left_split_new, right_split_new, threshold = igain_attribute(X, y, attr, self.metricVal)

                    # If gain obtained at this attribute is more than the current gain
                    # then update the gain value, decision attribute, left_split, right_split and threshold value.
                    if gain_new > gain:
                        gain = gain_new
                        self.decision_attr = attr
                        left_split = left_split_new
                        right_split = right_split_new
                        self.threshold = threshold

                # Set the gain, left_split and right_split to values corresponding to the attribute which gives maximum gain
                self.gain = gain
                self.left_split = left_split
                self.right_split = right_split
        else:
            # Node is a leaf so set is_leaf to True
            self.is_leaf = True

        # Uncomment this block to see how the nodes are being generated

        # print("************************")
        # print("Node ID:", self.node_id)
        # print("Vote:", self.vote)
        # print("Decision Attribute: ", self.decision_attr)
        # print("Threshold: ", self.threshold)
        # print("Is Leaf: ", self.is_leaf)
        # print("************************")

    def test(self, x, use_depth):
        """
        The test method of the node class takes in two input parameters x and use_depth
        and returns the classification assigned to it by the subtree at this node.

        Inputs:
            'x': x is one single instance of the attributes values arranged in a single row.

            'use_depth': The depth up to which the tree should be used to make predictions.

        Outputs:
            'Class': The classification assigned to it by the subtree at this node.
        """
        # If the node's depth is greater than or equal to the depth we are using for predictions or if the node is leaf return its vote
        # else depending on the decision attribute's value move to the left child or the right child and recursively keep testing.
        if self.depth >= use_depth or self.is_leaf:
            return self.vote
        elif x[self.decision_attr] <= self.threshold:
            return self.children[0].test(x, use_depth)
        else:
            return self.children[1].test(x, use_depth)


# Decision Tree Classifier Class


class DecisionTreeClassifier:
    def __init__(self, max_depth=100, metric="gini", reuse_attribute=False, split=2):
        """
        The __init__ method of the node class takes in four input parameters - max_depth, metric, reuse_attribute and split and constructs the node.

        Inputs:
            'max_depth': The maximum depth up to which the tree should be built. It is defaulted to 100.

            'metric':   The metric to be used when comparing different attributes to be chosen for decision at this node.
                        It takes values 'gini' or 'entropy' and is defaulted to 'gini'.

            'reuse_attribute': This boolean parameter tells us whether or not we want to reuse attributes.

            'split': This parameter has been defaulted to 2. It indicates the number of subsets a attribute column can be split into.

            NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
                    For all purposes in this algorithm a node will be split into two and two nodes only.

        Attributes:
            'root': root is a node object corresponding to the the root of the Decision Tree.
                    It is initialised to None.

            'max_depth': The maximum depth up to which the tree should be built.

            'metric':   metric is the impurity measure to be evaluated for splitting the sample instances at this node.
                        It takes values 'gini' or 'entropy' and is defaulted to 'gini'.

            'reuse_attribute': This boolean parameter tells us whether or not we want to reuse attributes.

            'split': split is the maximum number of children a node can have.

            NOTE:   split > 2 is not implemented in this algorithm, this parameter is a design choice and has been mentioned for extension purposes only.
                    For all purposes in this algorithm a node will be split into two and two nodes only.

            'node_dict':    The dictionary of nodes whose key is a node_id. It contains all the nodes of the decision tree.
                            The item in the dictionary is of the form key: node_id and value: node. It is initialised to an empty dictionary.

            'node_at_depth_count': The dictionary of counts of node at a depth d. Key: depth d and Value: Number of nodes at depth d

            'X_train':  X_train is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.
                        It contains all the training samples on which the Decision Tree has to be trained.
                        It is initialised to None.

            'y_train':  y_train is a indexed pandas dataframe consisting of target values arranged a column.
                        It contains all the target values to corresponding training samples on which the Decision Tree has to be trained.
                        It is initialised to None.
        """
        node.node_id_number = 1
        self.root = None
        self.max_depth = max_depth
        self.reuse_attribute = reuse_attribute
        self.metric = metric
        self.split = split
        self.node_dict = {}
        self.node_at_depth_count = {}
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        The fit method takes in two input parameters - X and y.
        It constructs the complete decision tree until all nodes are pure or there are no attributes left.

        Inputs:
            'X': X is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.

            'y': y is a indexed pandas dataframe consisting of target values in a column.
        """
        # Store the dataset on which the decision tree is being trained as part of it
        self.X_train = X
        self.y_train = y

        # List out the attributes of the training set
        attributes_list = list(self.X_train.columns)

        # Create the root node
        root = node(X=self.X_train, y=self.y_train, attributes_left=attributes_list, depth_of_parent=0,
                    max_depth=self.max_depth, metric=self.metric, split=self.split)

        # Store the root node
        self.root = root

        # Add the root node to the dictionary of nodes
        self.node_dict[root.node_id] = root

        # Increment count of nodes at depth 1
        if '1' not in self.node_at_depth_count.keys():
            self.node_at_depth_count['1'] = 1
        else:
            self.node_at_depth_count['1'] += 1

        # Remove the decision attribute used for decision at this root node from the list if reuse_attribute is set to false
        if not self.reuse_attribute:
            attributes_list.remove(root.decision_attr)

        # Build the subtree rooted at root (Subtree rooted at node is the tree itself)
        self.build(self.root, attributes_list, root.left_split, root.right_split)

    def build(self, root, attributes_left, left_split, right_split):
        """
        The build method takes in 4 input parameters - root, attributes_left, left_split and right_split
        and constructs the subtree rooted at the node 'root'.

        Inputs:
            'root': The root is a node object at which we want to construct the subtree.

            'attributes_left': attributes_left is the list of attributes that are left to consider

            'left_split': The indices of the sample instances from self.X_train, that are sent to the left child of 'root'.

            'right_split': The indices of the sample instances from self.X_train, that are sent to the right child of 'root'.
        """
        # If root is not a leaf then create the left child node and right child node and add them as children of the root node
        if not root.is_leaf:
            left_child = node(X=self.X_train.loc[left_split], y=self.y_train.loc[left_split],
                              attributes_left=attributes_left, depth_of_parent=root.depth, max_depth=self.max_depth,
                              metric=self.metric, split=self.split)
            root.children.append(left_child)
            right_child = node(X=self.X_train.loc[right_split], y=self.y_train.loc[right_split],
                               attributes_left=attributes_left, depth_of_parent=root.depth, max_depth=self.max_depth,
                               metric=self.metric, split=self.split)
            root.children.append(right_child)

            # Add the left and right child to the node_dict dictionary
            self.node_dict[left_child.node_id] = left_child
            self.node_dict[right_child.node_id] = right_child

            # Increment by 2 count of nodes at depth root.depth + 1
            if str(root.depth + 1) not in self.node_at_depth_count.keys():
                self.node_at_depth_count[str(root.depth + 1)] = 2
            else:
                self.node_at_depth_count[str(root.depth + 1)] += 2

            # Deep copy the attributes list
            attributes_list_1 = copy.deepcopy(attributes_left)
            attributes_list_2 = copy.deepcopy(attributes_left)

            # If the left child is not a leaf then build the subtree using the left child as root and the remaining attributes
            if not left_child.is_leaf:
                if not self.reuse_attribute:
                    attributes_list_1.remove(left_child.decision_attr)
                self.build(left_child, attributes_list_1, left_child.left_split, left_child.right_split)

            # If the right child is not a leaf then build the subtree using the right child as root and the remaining attributes
            if not right_child.is_leaf:
                if not self.reuse_attribute:
                    attributes_list_2.remove(right_child.decision_attr)
                self.build(right_child, attributes_list_2, right_child.left_split, right_child.right_split)

    def predict(self, X, use_depth=100):
        """
        The predict method takes in two input parameters X and use_depth and returns the list of predictions made by the tree on X.

        Inputs:
            'X': X is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.

            'use_depth': The depth up to which the tree should be used to make predictions.

        Outputs:
            'y_predict': The list of predictions(classifications) made on each training instance of X
        """
        # Get the indices in X
        indices = X.index

        # Initialise the list of predictions as an empty list
        y_predict = []

        # For each training instance, classify it and add it to the prediction list
        for i in indices:
            y_predict.append(self.root.test(X.loc[i], use_depth))

        return y_predict

    def prune(self, evaluation_set, stopping_rounds=0):
        """
        The prune method takes in two input parameters- evaluation_set and stopping_rounds
        and prunes the nodes based on the evaluation set and uses the stopping rounds to decide when to stop pruning.

        'evaluation_set':   It is a tuple of X_valid and y_valid.
                            X_valid is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.
                            y_valid is a indexed pandas dataframe consisting of target values arranged a column.
                            The validation set on which we decide whether further pruning of the tree will benefit us or not.

        'stopping_rounds': The number of rounds for which we prune further and see whether the pruning improves the accuracy or not
                            even if in the initial rounds the accuracy might have decreased.
        """
        # Get the X_valid and y_valid from the tuple
        (X_valid, y_valid) = evaluation_set

        # Until pruning is completed keep looping
        # Pruning is completed when there is no node which upon pruning would give a better accuracy than the current tree.
        done = False
        while not done:
            # Compute the accuracy of the current decision tree before pruning
            accuracy_before_pruning = accuracy(y_valid, self.predict(X_valid))

            # accuracy_dict is a dictionary of key: node_id and value: accuracy after node with 'key' as id is pruned
            accuracy_dict = {}

            # For every node in the tree, if the node is not a leaf then it is candidate for pruning.
            # So set the node as leaf(prune it) and then compute the accuracy on the pruned tree
            # Store this accuracy in accuracy_dict with the node's ID as key
            # Set the node back to being not a leaf
            for node_i in self.node_dict.values():
                if not node_i.is_leaf:
                    node_i.is_leaf = True
                    accuracy_after_pruning = accuracy(y_valid, self.predict(X_valid))
                    accuracy_dict[node_i.node_id] = accuracy_after_pruning
                    node_i.is_leaf = False

            # Stop the pruning when there is no node in accuracy_dict
            if len(accuracy_dict) == 0:
                break

            # Store the ID of the node which has the maximum accuracy upon pruning
            max_accuracy_node_id = max(accuracy_dict, key=accuracy_dict.get)

            # If the maximum accuracy obtained after pruning is less than the accuracy before pruning then we are done
            # Else if the accuracy improves then prune the node.
            # For pruning the node, we set it as leaf and remove its children form the node_dict dictionary
            if accuracy_dict[max_accuracy_node_id] < accuracy_before_pruning:
                done = True
            else:
                pruned_node = self.node_dict[max_accuracy_node_id]
                # Uncomment the below block to see which nodes are being pruned
                # print("Pruned Node with ID: ", pruned_node.node_id)
                pruned_node.is_leaf = True
                self.remove_children(pruned_node.node_id)

        # The following code checks to see if further pruning has any scope of improvement
        # It performs the pruning for a number of rounds as mentioned in 'stopping_rounds' even if the accuracy may decrease a little
        # Then it checks whether or not the accuracy improves in the last round (slope is positive)
        # This means that there is further scope of improvement
        # So we prune the tree again until we see that no improvement is possible

        accuracy_before_last_round = None  # Store the accuracy just before the last round of pruning
        accuracy_after_last_round = None  # Store the accuracy just after the last round of pruning

        # If stopping_rounds is greater than 0 then perform further pruning
        if stopping_rounds > 0:

            # Perform further pruning for 'stopping_rounds' number of rounds
            for i in range(0, stopping_rounds):
                # Compute the accuracy of the current decision tree before pruning
                accuracy_before_pruning = accuracy(y_valid, self.predict(X_valid))

                # # Store the accuracy just before the last round of pruning
                accuracy_before_last_round = accuracy_before_pruning

                # accuracy_dict is a dictionary of key: node_id and value: accuracy after node with 'key' as id is pruned
                accuracy_dict = {}

                # For every node in the tree, if the node is not a leaf then it is candidate for pruning.
                # So set the node as leaf(prune it) and then compute the accuracy on the pruned tree
                # Store this accuracy in accuracy_dict with the node's ID as key
                # Set the node back to being not a leaf
                for node_i in self.node_dict.values():
                    if not node_i.is_leaf:
                        node_i.is_leaf = True
                        accuracy_after_pruning = accuracy(y_valid, self.predict(X_valid))
                        accuracy_after_last_round = accuracy_after_pruning
                        accuracy_dict[node_i.node_id] = accuracy_after_pruning
                        node_i.is_leaf = False

                # Stop the pruning when there is no node in accuracy_dict
                if len(accuracy_dict) == 0:
                    break

                # Store the ID of the node which has the maximum accuracy upon pruning
                max_accuracy_node_id = max(accuracy_dict, key=accuracy_dict.get)

                # Store the node to be pruned in pruned_node, then we set it as leaf and remove its children form the node_dict dictionary
                pruned_node = self.node_dict[max_accuracy_node_id]
                # Uncomment the below block to see which nodes are being pruned
                # print("Pruning On Stopping Rounds: Pruned Node with ID: ", pruned_node.node_id)
                pruned_node.is_leaf = True
                self.remove_children(pruned_node.node_id)

            # If the accuracy improves in the last round of pruning,
            if accuracy_after_last_round > accuracy_before_last_round:
                self.prune(evaluation_set, stopping_rounds=0)

    def prune_node_vary(self, evaluation_set_train, evaluation_set_test, evaluation_set_val):
        """
        The prune_node_vary method takes in three input parameters- evaluation_set_train, evaluation_set_test, evaluation_set_val
        and prunes the nodes based on the evaluation_set_val. This is a utility method to help us monitor the status and goodness of the tree.

        'evaluation_set_train': It is a tuple of X_train and y_train.
                                X_train is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.
                                y_train is a indexed pandas dataframe consisting of target values arranged a column.
                                To test the accuracy on the training set while pruning.

        'evaluation_set_test':  It is a tuple of X_valid and y_valid.
                                X_valid is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.
                                y_valid is a indexed pandas dataframe consisting of target values arranged a column.
                                To test the accuracy on the test set while pruning.

        'evaluation_set_val':   It is a tuple of X_valid and y_valid.
                                X_valid is a indexed pandas dataframe consisting of attribute values arranged in rows and columns.
                                y_valid is a indexed pandas dataframe consisting of target values arranged a column.
                                The validation set on which we decide whether further pruning of the tree will benefit us or not.
        """
        # The list of accuracies obtained on various stages of pruning on train, test and validation set
        train_accuracies = []
        test_accuracies = []
        val_accuracies = []

        # prunes is list of the number of nodes in the tree during various stages of pruning
        prunes = []

        # Get the attributes dataframe and target value dataframe from training set, test set and validation set
        (X_train, y_train) = evaluation_set_train
        (X_test, y_test) = evaluation_set_test
        (X_valid, y_valid) = evaluation_set_val

        # Initialised pruned to 0
        pruned = 0

        # Until pruning is completed keep looping
        # Pruning is completed when there is no node left for pruning
        while len(self.node_dict) > 1:
            # Compute the accuracy of the current decision tree before pruning
            accuracy_before_pruning = accuracy(y_valid, self.predict(X_valid))

            # Get the accuracies on the test set and the training set
            train_acc = accuracy(y_train, self.predict(X_train))
            test_acc = accuracy(y_test, self.predict(X_test))

            # Store the accuracies in the corresponding lists
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            val_accuracies.append(accuracy_before_pruning)

            # Store the current number of nodes in the list
            prunes.append(len(self.node_dict))

            # accuracy_dict is a dictionary of key: node_id and value: accuracy after node with 'key' as id is pruned
            accuracy_dict = {}

            # For every node in the tree, if the node is not a leaf then it is candidate for pruning.
            # So set the node as leaf(prune it) and then compute the accuracy on the pruned tree
            # Store this accuracy in accuracy_dict with the node's ID as key
            # Set the node back to being not a leaf
            for node_i in self.node_dict.values():
                if not node_i.is_leaf:
                    node_i.is_leaf = True
                    accuracy_after_pruning = accuracy(y_valid, self.predict(X_valid))
                    accuracy_dict[node_i.node_id] = accuracy_after_pruning
                    node_i.is_leaf = False

            # Stop the pruning when there is no node in accuracy_dict
            if len(accuracy_dict) == 0:
                break

            # Store the ID of the node which has the maximum accuracy upon pruning
            max_accuracy_node_id = max(accuracy_dict, key=accuracy_dict.get)

            # Prune the node which gives the maximum accuracy after pruning
            # For pruning the node, we set it as leaf and remove its children form the node_dict dictionary

            pruned_node = self.node_dict[max_accuracy_node_id]
            # Uncomment the below block to see which nodes are being pruned
            # print("Pruned Node with ID: ", pruned_node.node_id)
            pruned_node.is_leaf = True
            self.remove_children(pruned_node.node_id)

            # Increment the number of nodes pruned
            pruned = pruned + 1

        return train_accuracies, test_accuracies, val_accuracies, prunes

    def remove_children(self, node_id):
        """
        The remove_children method takes in one input parameter-node_id and removes the subtree rooted
        at node with node_id from the node_dict dictionary.

        Inputs:
            'node_id': ID of the node whose subtree we want to remove.
        """
        # If the node_id is not in node_dict then return
        if node_id not in self.node_dict:
            return

        # If the node_id does not have children then return
        if len(self.node_dict[node_id].children) == 0:
            return
        else:
            # Else if the node_id has children
            # Then for every children remove their subtree
            # And remove the child from node_dict if it has not already been removed
            for child in self.node_dict[node_id].children:
                self.remove_children(child.node_id)
                if child.node_id in self.node_dict:
                    self.node_dict.pop(child.node_id)
