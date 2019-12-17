import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
validation_file = sys.argv[3]


df = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_val = pd.read_csv(validation_file)

X_train = np.asarray(df[1:], dtype=float)
y_train = X_train[:, 24]
X_train = np.delete(X_train, -1, axis=1)

X_test = np.asarray(df_test[1:], dtype=float)
y_test = X_test[:, 24]
X_test = np.delete(X_test, -1, axis=1)

X_val = np.asarray(df_val[1:], dtype=float)
y_val = X_val[:, 24]
X_val = np.delete(X_val, -1, axis=1)


# clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=500, max_depth=None, min_samples_leaf=None)
# clf.fit(X_train, y_train)
#
# print(np.sum(clf.predict(X_train) == y_train)/len(X_train))
# print(np.sum(clf.predict(X_test) == y_test)/len(X_test))
# print(np.sum(clf.predict(X_val) == y_val)/len(X_val))


# -- best parameter setting

clf1 = DecisionTreeClassifier(criterion='entropy', min_samples_split=5, max_depth=7, min_samples_leaf=6)
clf1.fit(X_train, y_train)


print("Train", np.sum(clf1.predict(X_train) == y_train)/len(X_train))
print("Test", np.sum(clf1.predict(X_test) == y_test)/len(X_test))
print("Valid", np.sum(clf1.predict(X_val) == y_val)/len(X_val))
