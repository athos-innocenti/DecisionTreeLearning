from dt_learning import DecisionTree

import random


def remove_data(examples, attributes_indexes, p):
    for ex in examples:
        for attr in attributes_indexes:
            if random.uniform(0, 1) <= p:
                ex[attr] = ''
    return examples


def get_dts_name(dataset):
    return dataset.replace(".data", "").upper()


def get_attributes(dataset):
    f = open('datasets/names/{}'.format(dataset.replace(".data", ".names")))
    attributes = f.readline().split(',')  # read first line of data file containing attributes' name
    f.close()
    if attributes[0] != "Class":  # example's classification is named Class in .name file
        target_pos = len(attributes) - 1
        return attributes[:target_pos], target_pos  # classification is at the end
    else:
        target_pos = 0
        return attributes[target_pos + 1:], target_pos  # classification is at the beginning


def get_examples(dataset, target_position):
    f = open('datasets/{}'.format(dataset))
    examples = f.readlines()
    f.close()
    classification = []
    for i in range(len(examples)):
        examples[i] = examples[i].strip()  # return a copy of the string where whitespaces are removed
        examples[i] = examples[i].split(',')  # return a list of words of the string where , is used as separator
        classification.append(examples[i][target_position])
        if target_position != 0:
            examples[i] = examples[i][:target_position]  # classification is at the end
        else:
            examples[i] = examples[i][target_position + 1:]  # classification is at the beginning
    return examples, classification


def get_attributes_values(examples, attributes_indexes):
    return [DecisionTree.get_values(examples, attr) for attr in attributes_indexes]


def get_target_values(classification):
    values = []
    for c in classification:
        if c not in values:
            values.append(c)
    return values
