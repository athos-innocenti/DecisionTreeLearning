from dt_learning import DecisionTree
import dataset
import performances

from termcolor import colored
import matplotlib.pyplot as plt
import random
import copy
import statistics

probability = [0, 0.1, 0.2, 0.5]
datasets = ['tic-tac-toe.data', 'balance-scale.data', 'kr-vs-kp.data']

for dts in datasets:
    dts_name = dataset.get_dts_name(dts)
    attributes, target_pos = dataset.get_attributes(dts)
    examples, classification = dataset.get_examples(dts, target_pos)
    examples_indexes = [x for x in range(len(examples))]
    attributes_indexes = [x for x in range(len(attributes))]
    attributes_val = dataset.get_attributes_values(examples, attributes_indexes)
    target_val = dataset.get_target_values(classification)
    data, test_set = [], []  # 80% TRN+VAL 20% TST
    for _ in range(int(len(examples) / 5)):
        ind = random.choice(examples_indexes)
        test_set.append(examples[ind] + [classification[ind]])
        examples_indexes.remove(ind)
    for i in examples_indexes:
        data.append(examples[i] + [classification[i]])
    accuracy_prob = []
    for p in probability:
        print(colored("\nProbability =", "blue"), p)
        best_height = []
        data_o = copy.deepcopy(data)
        data_r = dataset.remove_data(copy.deepcopy(data), attributes_indexes, p)
        data_indexes = [x for x in range(len(data_o))]
        training_blocks, validation_blocks = [], []
        for _ in range(5):  # GRID SEARCH CROSS VALIDATION
            block_t, block_v = [], []
            for i in range(int(len(data_r) / 5)):   # 1 BLOCK VALIDATION + 4 BLOCKS TRAINING
                ind = random.choice(data_indexes)
                block_t.append(data_r[ind])
                block_v.append(data_o[ind])
                data_indexes.remove(ind)
            training_blocks.append(block_t)
            validation_blocks.append(block_v)
        for section in range(len(training_blocks)):
            vld_set = validation_blocks[section]
            trn_set = []
            for i in range(len(training_blocks)):
                if i != section:
                    trn_set.extend(training_blocks[i])
            accuracy_hgt = []
            wgt = [1 for _ in range(len(trn_set))]
            dtl = DecisionTree(copy.deepcopy(attributes_indexes), len(trn_set[0]) - 1)
            original_tree = dtl.dt_learning(trn_set, trn_set, attributes, attributes_val, wgt, wgt, 0, len(attributes))
            accuracy_hgt.append(performances.accuracy(copy.deepcopy(vld_set), original_tree, attributes, target_val))
            max_height = dtl.get_max_height()
            for h in range(1, max_height):  # FIND HEIGHT FOR BEST ACCURACY OVER VLD
                wgt = [1 for _ in range(len(trn_set))]
                dt = DecisionTree(copy.deepcopy(attributes_indexes), len(trn_set[0]) - 1)
                tree = dt.dt_learning(trn_set, trn_set, attributes, attributes_val, wgt, wgt, 0, h)
                accuracy_hgt.append(performances.accuracy(copy.deepcopy(vld_set), tree, attributes, target_val))
            best_height.append(accuracy_hgt.index(max(accuracy_hgt)))
        mode_best_height = statistics.mode(best_height)
        dtl = DecisionTree(copy.deepcopy(attributes_indexes), len(data_r[0]) - 1)
        wgt = [1 for _ in range(len(data_r))]
        best_tree = dtl.dt_learning(data_r, data_r, attributes, attributes_val, wgt, wgt, 0, mode_best_height)
        accuracy = performances.accuracy(copy.deepcopy(test_set), best_tree, attributes, target_val)
        print(colored("Accuracy =", "cyan"), "%.2f" % accuracy, "%\n")
        accuracy_prob.append(accuracy)
    plt.plot(probability, accuracy_prob)
    plt.title('{} Dataset'.format(dts_name))
    plt.xlabel('Delete probability')
    plt.ylabel('Accuracy over test set')
    plt.show()
