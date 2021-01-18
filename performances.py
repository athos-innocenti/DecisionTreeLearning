def test(example, root, attributes, target_values):
    while root.value not in target_values:
        branch = example[attributes.index(root.value)]
        root = root.children[root.branch.index(branch)]
    target_value = root.value
    if target_value == example[len(example) - 1]:
        return True
    return False


def accuracy(test_set, root, attributes, target_values):
    correct = 0
    for ex in test_set:
        if test(ex, root, attributes, target_values):
            correct += 1
    return correct / float(len(test_set)) * 100


'''
def print_tree(root, count):
    print(" " * (count + 1), root.value)
    for branch in root.branch:
        print(" " * (count + 2), "=", branch)
        print_tree(root.children[root.branch.index(branch)], count + 4)
'''
