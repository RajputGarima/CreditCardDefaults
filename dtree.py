import pandas as pd
import numpy as np
from params import *
from matplotlib import pyplot as plt
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
validation_file = sys.argv[3]
mode = int(sys.argv[4])

df = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_val = pd.read_csv(validation_file)


# --dropping ID because it is unique for every example
df.drop(['X0'], axis=1, inplace=True)
df_test.drop(['X0'], axis=1, inplace=True)
df_val.drop(['X0'], axis=1, inplace=True)


D = np.asarray(df[1:], dtype=float)
nodes = 0
a = 0
x = 15000
count = 0
h = 25
node_arr = []
train_arr = []
test_arr = []
valid_arr = []
# print(D[:, 0]) --- prints first column of D


# to convert continuous values to boolean
def preprocess():
    medians = np.median(D, axis=0)
    continuous = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for i in continuous:
        D[:, i][D[:, i] <= medians[i]] = 0
        D[:, i][D[:, i] > medians[i]] = 1


preprocess()
# print(np.unique(D[:, -1])) -- array of unique values


def pure_leaves(D):
    if len(np.unique(D[:, -1])) == 1:
        return True
    else:
        return False


def classify_data(D):
    classes, counts = np.unique(D[:, -1], return_counts=True)
    classified_class = classes[counts.argmax()]
    return classified_class


def split_data_bool(D, split_col):
    false_data = D[D[:, split_col] == 0.0]
    true_data = D[D[:, split_col] == 1.0]
    return false_data, true_data


def split_data_bool_b(D, split_col):
    false_data = D[D[:, split_col] == 2.0]
    true_data = D[D[:, split_col] == 1.0]
    return false_data, true_data


def split_data_categorical(D, split_col, limit):
    cat_list = []
    if limit == 7 or limit == 4:
        for i in range(limit):
            cat_list.append(D[D[:, split_col] == i])
            #print(len(cat_list[i]))
    else:
        for i in range(limit):
            cat_list.append(D[D[:, split_col] == i])
            #print(len(cat_list[i]))
        cat_list.append(D[D[:, split_col] == -1])
        cat_list.append(D[D[:, split_col] == -2])
    return np.asarray(cat_list)


def entropy_calc(D):
    _, counts = np.unique(D[:, -1], return_counts=True)
    counts = counts[counts != 0]
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


def weighted_entropy(true_data, false_data):
    te = entropy_calc(true_data)
    fe = entropy_calc(false_data)
    total = len(true_data) + len(false_data)
    total_weighted_entropy = (len(true_data)/total) * te + (len(false_data)/total) * fe
    return total_weighted_entropy


def weighted_entropy_cat(cat_list):
    n = len(cat_list)
    total_length = 0
    total_etp = 0.0
    etp = np.zeros(n)
    for i in range(n):
        etp[i] = entropy_calc(cat_list[i])
        total_length += len(cat_list[i])
    for i in range(n):
        total_etp += (len(cat_list[i])/total_length) * etp[i]
    return total_etp


def choose_best_attribute(D):
    best = None
    bool_list = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for i in bool_list:
        false_data, true_data = split_data_bool(D, i)
        we = weighted_entropy(true_data, false_data)
        if best is None or we < best:
            best = we
            best_attribute = i

    # --- binary attr
    false_data, true_data = split_data_bool_b(D, 1)
    we = weighted_entropy(true_data, false_data)
    if best is None or we < best:
        best = we
        best_attribute = 1

    # -- categorical attr
    for i in [2, 3]:
        limit = 7 if i == 2 else 4
        cat_list = split_data_categorical(D, i, limit)
        we = weighted_entropy_cat(cat_list)
        if we < best:
            best = we
            best_attribute = i

    categorical_list = [5, 6, 7, 8, 9, 10]
    limit = 10
    for i in categorical_list:
        cat_list = split_data_categorical(D, i, limit)
        we = weighted_entropy_cat(cat_list)
        if we < best:
            best = we
            best_attribute = i

    return best_attribute


def decision_tree(D, step, redundant=False, height=0):
    global nodes
    if pure_leaves(D) or redundant or height > h or nodes > x:
        return classify_data(D)
    else:
        height += 1
        attr = choose_best_attribute(D)
        step += 1
        if step % 20 == 0:
            node_arr.append(nodes)

        if attr in [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            nodes += 2
            false_data, true_data = split_data_bool(D, attr)
        elif attr == 1:
            nodes += 2
            false_data, true_data = split_data_bool_b(D, attr)
        elif attr == 2:
            nodes += 7
            cat_list = split_data_categorical(D, attr, 7)
        elif attr == 3:
            nodes += 4
            cat_list = split_data_categorical(D, attr, 4)
        elif attr in [5, 6, 7, 8, 9, 10]:
            nodes += 12
            cat_list = split_data_categorical(D, attr, 10)

        # print("True", len(true_data))
        # print("False", len(false_data))
        if attr in [0, 1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            if len(true_data) == 0:
                question = "{}".format(attr)
                sub_tree = {question: []}
                no_answer = decision_tree(D, True, height)
                yes_answer = 0.0 if no_answer == 1 else 1.0
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
                return sub_tree
            elif len(false_data) == 0:
                question = "{}".format(attr)
                sub_tree = {question: []}
                yes_answer = decision_tree(D, True, height)
                no_answer = 0.0 if yes_answer == 1 else 1.0
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
                return sub_tree
            else:
                question = "{}".format(attr)
                sub_tree = {question: []}
                yes_answer = decision_tree(true_data, step, False, height)
                no_answer = decision_tree(false_data, step, False, height)
                if yes_answer == no_answer:
                    sub_tree = yes_answer
                else:
                    sub_tree[question].append(yes_answer)
                    sub_tree[question].append(no_answer)

                return sub_tree
        else:
            question = "{}".format(attr)
            sub_tree = {question: []}
            n = len(cat_list)
            for i in range(n):
                if len(cat_list[i]) > 0:
                    a = decision_tree(cat_list[i], step, False, height)
                    sub_tree[question].append(a)
                else:
                    a = decision_tree(D, step, True, height)
                    sub_tree[question].append(a)
            return sub_tree


def predict(input, tree, node_count):
    global count, a
    count += 1
    if count >= node_count:
        print(count)
        return a
    question = list(tree.keys())[0]
    attr = question.split()
    attribute = int(attr[0])
    if attribute in [0, 1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        if input[int(attr[0])] == 1.0:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    elif attribute in [5, 6, 7, 8, 9, 10]:
        for i in range(10):
            if input[int(attr[0])] == i:
                answer = tree[question][i]
        if input[int(attr[0])] == -1:
            answer = tree[question][10]
        if input[int(attr[0])] == -2:
            answer = tree[question][11]

    elif attribute == 2:
        for i in range(7):
            if input[int(attr[0])] == i:
                answer = tree[question][i]

    else:
        for i in range(4):
            if input[int(attr[0])] == i:
                answer = tree[question][i]

    if not isinstance(answer, dict):
        a = answer
        return answer
    else:
        return predict(input, answer, node_count)


def accuracy(D, tree, node_count):
    correct = 0
    d_size = len(D)
    for i in range(d_size):
        global count
        count = 0
        temp = predict(D[i], tree, node_count)
        if temp == D[i][23]:
            correct += 1
    return correct/d_size


tree = decision_tree(D, step=20, redundant=False)


def datasets_prediction(tree):
    global D
    # -- training set accuracy
    D = np.asarray(df[1:], dtype=float)
    preprocess()
    train_acc = accuracy(D, tree, 20000)
    print("Training data accuracy:", train_acc)
    # -- testing set accuracy
    D = np.asarray(df_test[1:], dtype=float)
    preprocess()
    test_acc = accuracy(D, tree, 20000)
    print("Testing data accuracy:", test_acc)
    # -- validation set accuracy
    D = np.asarray(df_val[1:], dtype=float)
    preprocess()
    val_acc = accuracy(D, tree, 20000)
    print("Validation set accuracy: ", val_acc)


def plot_curve():
    D = np.asarray(df[1:], dtype=float)
    preprocess()
    for i in node_arr:
        train_arr.append(accuracy(D, tree, i))
    D = np.asarray(df_test[1:], dtype=float)
    preprocess()
    for i in node_arr:
        test_arr.append(accuracy(D, tree, i))
    D = np.asarray(df_val[1:], dtype=float)
    preprocess()
    for i in node_arr:
        valid_arr.append(accuracy(D, tree, i))


def nodes_vs_accuracy_part_a():
    plot_curve()
    train_arr = mylista['train_acc1']
    test_arr = mylista['test_acc1']
    valid_arr = mylista['valid_acc1']
    nodes = na
    line1, = plt.plot(nodes, train_arr, "b",
                      label="Training accuracy")
    line2, = plt.plot(nodes, test_arr, "r", label="Testing accuracy ")
    line3, = plt.plot(nodes, valid_arr, "g",
                      label="Validation Accuracy")
    plt.legend(handles=[line1, line2, line3])
    plt.ylabel('Accuracy')
    plt.xlabel("Number of nodes")
    plt.title("Number of nodes v/s Accuracy")
    plt.show()


if mode == 1:
    datasets_prediction(tree)
    nodes_vs_accuracy_part_a()


# --- pruning the tree


def can_prune(node):
    for i in node.values():
        for j in i:
            if isinstance(j, dict):
                return True
    return False


def is_accuracy_improved(new_tree):
    D = np.asarray(df_val[1:], dtype=float)
    preprocess()
    a1 = accuracy(D, tree)
    a2 = accuracy(D, new_tree)
    if a1 > a2:
        return False
    return True


def prune_tree(t):
    for i in tree.values():
        #print(tree.keys())
        for j in reversed(i):
            if isinstance(j, dict):
                prune_tree(j)
                if can_prune(j):
                    if is_accuracy_improved(tree):
                        del t[hash(str(j))]
                #     else:
                #         #print("not to be pruned")
                # else:
                #     # print("leaf node")


def nodes_vs_accuracy_part_b():
    plot_curve()
    plot_curve()
    train_arr = mylistb['train']
    test_arr = mylistb['test']
    valid_arr = mylistb['valid']
    nodes = nb
    line1, = plt.plot(nodes, train_arr, "b",
                      label="Training accuracy")
    line2, = plt.plot(nodes, test_arr, "r", label="Testing accuracy ")
    line3, = plt.plot(nodes, valid_arr, "g",
                      label="Validation Accuracy")
    plt.gca().invert_xaxis()
    plt.legend(handles=[line1, line2, line3])
    plt.ylabel('Accuracy')
    plt.xlabel("Number of nodes")
    plt.title("Number of nodes v/s Accuracy(with pruning)")
    plt.show()


if mode == 2:
    tree = decision_tree(D, 20, redundant=False)
    plot_curve()
    datasets_prediction(tree)
    nodes_vs_accuracy_part_b()
