from dt_learning import DecisionTree

from scipy.stats import chi2
import numpy as np


class TreePruning:
    def __init__(self, examples, attributes, target_values):
        self.examples = examples
        self.attributes = attributes
        self.target_values = target_values
        self.table = chi2.isf(np.array([0.05]), np.array(range(1, 30)))

    @staticmethod
    def get_occurrence(attr_values, examples, index):
        values_occurrence = []
        for val in attr_values:
            count_occurrence = 0
            for ex in examples:
                if ex[index] == val:
                    count_occurrence += 1
            values_occurrence.append(count_occurrence)
        return values_occurrence

    def plurality_value(self, examples):
        target_values = DecisionTree.get_values(examples, len(examples[0]) - 1)
        target_values_occur = self.get_occurrence(target_values, examples, len(examples[0]) - 1)
        return target_values[target_values_occur.index(max(target_values_occur))]

    def delta(self, root, examples):
        target_values = DecisionTree.get_values(examples, len(examples[0]) - 1)
        target_values_occurrence = TreePruning.get_occurrence(target_values, examples, len(examples[0]) - 1)
        attr_values = DecisionTree.get_values(examples, self.attributes.index(root.value))
        delta_values = []
        for value in attr_values:
            exs = []
            for ex in examples:
                if ex[self.attributes.index(root.value)] == value:
                    exs.append(ex)
            target_values_occurrence_per_v = TreePruning.get_occurrence(target_values, exs, len(examples[0]) - 1)
            expected = []
            for val in target_values:
                for tar in target_values:
                    if tar == val:
                        expected.append(target_values_occurrence[target_values.index(tar)] * (
                            (sum(target_values_occurrence_per_v)) / len(examples)))
            arg = 0
            for ex in expected:
                arg += (((target_values_occurrence_per_v[expected.index(ex)] - ex) ** 2) / ex)
            delta_values.append(arg)
        return sum(delta_values)

    def leaf_descendants(self, node):
        if len(node.children) > 0:
            for child in node.children:
                if child.value not in self.target_values:
                    return False
            return True
        else:
            return False

    def chi_squared_pruning(self, root, examples, count):
        for c in range(len(root.children)):
            exs = []
            if self.leaf_descendants(root.children[c]):
                for ex in examples:
                    if ex[self.attributes.index(root.value)] == root.branch[c]:
                        exs.append(ex)
                if len(exs) > 0:
                    delta = self.delta(root.children[c], exs)
                    dof = len(DecisionTree.get_values(exs, self.attributes.index(root.children[c].value))) - 1
                    chi2_alpha = self.table[dof - 1]
                    if delta < chi2_alpha:
                        count += 1
                        for i in range(len(root.children[c].children)):
                            root.children[c].children.pop()
                            root.children[c].branch.pop()
                        root.children[c].value = self.plurality_value(exs)
            else:
                for ex in examples:
                    if ex[self.attributes.index(root.value)] == root.branch[root.children.index(root.children[c])]:
                        exs.append(ex)
                self.chi_squared_pruning(root.children[c], exs, count)
        return root, count

    def pruning(self, root):
        root, count_pruned_nodes = self.chi_squared_pruning(root, self.examples, 0)
        while count_pruned_nodes > 0:
            root, count_pruned_nodes = self.chi_squared_pruning(root, self.examples, 0)
            if count_pruned_nodes == 0:
                break
        return root
