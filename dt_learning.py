import math


class Node:
    def __init__(self):
        self.value = None
        self.height = None
        self.branch = []
        self.children = []


class DecisionTree:
    def __init__(self, attributes_indexes, target_position):
        self.attributes_indexes = attributes_indexes
        self.target_position = target_position
        self.height = 0

    def get_max_height(self):
        return self.height

    def height_update(self, h):
        if h > self.height:
            self.height = h

    def same_classification(self, examples):
        classification = examples[0][self.target_position]
        for ex in examples:
            if ex[self.target_position] != classification:
                return False
        return True

    @staticmethod
    def check_missing(examples, index):
        for ex in examples:
            if ex[index] == '':
                return True
        return False

    @staticmethod
    def get_values(examples, index):
        attr_values = []
        for ex in examples:
            value = ex[index]
            if value not in attr_values and value != '':
                attr_values.append(value)
        return attr_values

    def plurality_value(self, examples, weights):
        target_values = self.get_values(examples, self.target_position)
        target_values_occur = self.get_weighted_occur(weights, target_values, examples, self.target_position)
        return target_values[target_values_occur.index(max(target_values_occur))]

    @staticmethod
    def get_weighted_occur(weights, attr_values, examples, index):
        weights_per_val = []
        for val in attr_values:
            wgt = 0
            for ex in examples:
                if ex[index] == val:
                    wgt += weights[examples.index(ex)]
            weights_per_val.append(wgt)
        return weights_per_val

    @staticmethod
    def get_entropy(values, values_occurrence, total_weight):
        entropy = 0
        for i in range(len(values)):
            if values_occurrence[i] != 0:
                p_i = values_occurrence[i] / total_weight
                entropy -= p_i * math.log(p_i, 2)
        return entropy

    def gain(self, entropy_s, examples, attributes_values, weights):
        gain_per_attribute, prob = [], []
        s = sum(weights)
        for attr in self.attributes_indexes:
            missing = self.check_missing(examples, attr)
            if missing:
                known_examples, known_weights = [], []
                for ex in examples:
                    if ex[attr] != '':
                        known_examples.append(ex)
                        known_weights.append(weights[examples.index(ex)])
                weighted_occur = self.get_weighted_occur(known_weights, attributes_values[attr], known_examples, attr)
                total_weight = sum(known_weights)
                if total_weight != 0:
                    prob.append([weighted_occur[attributes_values[attr].index(val)] / total_weight for val in
                                 attributes_values[attr]])
                else:
                    prob.append([1 for _ in attributes_values[attr]])
            else:
                prob.append([1 for _ in attributes_values[attr]])
            remainder = 0
            for v in range(len(attributes_values[attr])):
                examples_per_v, weights_per_v = [], []
                for ex in examples:
                    if ex[attr] == attributes_values[attr][v] or ex[attr] == '':
                        examples_per_v.append(ex)
                        weights_per_v.append(weights[examples.index(ex)] * prob[self.attributes_indexes.index(attr)][v])
                if len(examples_per_v) != 0 and s != 0:
                    target_values_per_v = self.get_values(examples_per_v, self.target_position)
                    target_values_occur_per_v = self.get_weighted_occur(weights_per_v, target_values_per_v,
                                                                        examples_per_v, self.target_position)
                    s_v = sum(weights_per_v)
                    remainder += (s_v / s) * self.get_entropy(target_values_per_v, target_values_occur_per_v, s_v)
            gain_per_attribute.append(entropy_s - remainder)
        max_gain_index = gain_per_attribute.index(max(gain_per_attribute))
        return self.attributes_indexes[max_gain_index], prob[max_gain_index]

    def dt_learning(self, examples, parent_examples, attributes, attributes_values, weights, parent_wgt, height, max_h):
        root = Node()
        root.height = height
        if root.height == max_h:
            if len(examples) == 0:
                root.value = self.plurality_value(parent_examples, parent_wgt)
                return root
            elif self.same_classification(examples):
                root.value = examples[0][self.target_position]
                return root
            else:
                root.value = self.plurality_value(examples, weights)
                return root
        if len(examples) == 0:
            self.height_update(height)
            root.value = self.plurality_value(parent_examples, parent_wgt)
            return root
        elif len(self.attributes_indexes) == 0:
            self.height_update(height)
            root.value = self.plurality_value(examples, weights)
            return root
        elif self.same_classification(examples):
            self.height_update(height)
            root.value = examples[0][self.target_position]
            return root
        else:
            target_values = self.get_values(examples, self.target_position)
            target_values_occur = self.get_weighted_occur(weights, target_values, examples, self.target_position)
            entropy_s = self.get_entropy(target_values, target_values_occur, sum(weights))
            index, prob = self.gain(entropy_s, examples, attributes_values, weights)
            root.value = attributes[index]
            for value in attributes_values[index]:
                root.branch.append(value)
                exs, wgt = [], []
                for ex in examples:
                    if ex[index] == value or ex[index] == '':
                        exs.append(ex)
                        wgt.append(weights[examples.index(ex)] * prob[attributes_values[index].index(value)])
                if len(self.attributes_indexes) > 0 and index in self.attributes_indexes:
                    self.attributes_indexes.pop(self.attributes_indexes.index(index))
                root.children.append(
                    self.dt_learning(exs, examples, attributes, attributes_values, wgt, weights, height + 1, max_h))
        return root
