from dt_learning import DecisionTree
import dataset
import performances

from termcolor import colored
import matplotlib.pyplot as plt
import random
import copy

probability = [0, 0.1, 0.2, 0.5]
datasets = ['tic-tac-toe.data', 'cmc.data', 'OBS-Network-DataSet_2_Aug27.data']

for dts in datasets:
    dts_name = dataset.get_dts_name(dts)
    attributes, target_pos = dataset.get_attributes(dts)
    examples, classification = dataset.get_examples(dts, target_pos)
    examples_indexes = [x for x in range(len(examples))]
    attributes_indexes = [x for x in range(len(attributes))]
    attributes_val = dataset.get_attributes_values(examples, attributes_indexes)
    target_val = dataset.get_target_values(classification)
    print("\n", colored("-" * 25, "red"), colored(dts_name, "red"), colored("DATASET", "red"), colored("-" * 25, "red"))
    print(colored("\nAttributes' name:\n", "green"), attributes)
    print(colored("Attributes values:\n", "green"), attributes_val)
    print(colored("Target values:\n", "green"), target_val, "\n")
    training_set, validation_set, test_set = [], [], []  # 60% TRN - 20% VAL - 20% TST
    for _ in range(int(len(examples) / 5) * 3):
        ind = random.choice(examples_indexes)
        training_set.append(examples[ind] + [classification[ind]])
        examples_indexes.remove(ind)
    for _ in range(int(len(examples_indexes) / 2)):
        ind = random.choice(examples_indexes)
        test_set.append(examples[ind] + [classification[ind]])
        examples_indexes.remove(ind)
    for i in examples_indexes:
        validation_set.append(examples[i] + [classification[i]])
    accuracy_prob = []
    for p in probability:
        print(colored("\nProbability =", "blue"), p)
        trn_set_r = dataset.remove_data(copy.deepcopy(training_set), attributes_indexes, p)
        wgt = [1 for _ in trn_set_r]
        trees, accuracy_hgt = [], []
        dtl = DecisionTree(copy.deepcopy(attributes_indexes), len(trn_set_r[0]) - 1)
        original_tree = dtl.dt_learning(trn_set_r, trn_set_r, attributes, attributes_val, wgt, wgt, 0, len(attributes))
        trees.append(original_tree)
        accuracy_hgt.append(performances.accuracy(copy.deepcopy(validation_set), original_tree, attributes, target_val))
        max_height = dtl.get_max_height()
        for h in range(1, max_height):  # GRID SEARCH CROSS VALIDATION
            dt = DecisionTree(copy.deepcopy(attributes_indexes), len(trn_set_r[0]) - 1)
            tree = dt.dt_learning(trn_set_r, trn_set_r, attributes, attributes_val, wgt, wgt, 0, h)
            accuracy_hgt.append(performances.accuracy(copy.deepcopy(validation_set), tree, attributes, target_val))
            trees.append(tree)
        max_accuracy_index = accuracy_hgt.index(max(accuracy_hgt))
        best_tree = trees[max_accuracy_index]
        accuracy = performances.accuracy(copy.deepcopy(test_set), best_tree, attributes, target_val)
        print(colored("Accuracy =", "cyan"), "%.2f" % accuracy, "%\n")
        accuracy_prob.append(accuracy)
        '''
        # delete comment to show the variation in accuracy with respect to tree's depth
        print(colored("Max height:", "cyan"), max_height)
        print(colored("Best accuracy height:", "cyan"), max_accuracy_index + 1)
        plt.plot(range(1, max_height + 1), accuracy_hgt)
        plt.title('{} Dataset'.format(dataset.get_dts_name(dts)))
        plt.xlabel('Height')
        plt.ylabel('Accuracy over validation set')
        plt.show()
        '''
    plt.plot(probability, accuracy_prob)
    plt.title('{} Dataset'.format(dataset.get_dts_name(dts)))
    plt.xlabel('Delete probability')
    plt.ylabel('Accuracy over test set')
    plt.show()
