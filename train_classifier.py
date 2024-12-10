import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

count = 0
# Iterate through each item in the dataset's "dataset" list
for i in dataset["dataset"]:
    count += 1
    # Check if the length of the current item is not equal to 42 (expected length for hand landmarks)
    if len(i) != 42:
        print(len(i))
        # Find the index of the current item in the dataset
        index = dataset["dataset"].index(i)

        # Remove the item from both the "dataset" and "labels" lists at the found index
        dataset["dataset"].pop(index)
        dataset["labels"].pop(index)

        print(len(i))

data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)


place = []
place.extend(np.where(y_train == "1")[0])
place.extend(np.where(y_train == "2")[0])
place.extend(np.where(y_train == "3")[0])
place.extend(np.where(y_train == "4")[0])
print (place)
x_missing = []
y_missing = []
for i in place:
    x_missing.append(X_train[i])
    y_missing.append(y_train[i])
x_missing = np.asarray(x_missing)
y_missing = np.asarray(y_missing)





###############################################################################################

with open("./ASL1.pickle", "rb") as f:
    dataset = pickle.load(f)

count = 0
# Iterate through each item in the dataset's "dataset" list
for i in dataset["dataset"]:
    count += 1
    # Check if the length of the current item is not equal to 42 (expected length for hand landmarks)
    if len(i) != 42:
        print(len(i))
        # Find the index of the current item in the dataset
        index = dataset["dataset"].index(i)

        # Remove the item from both the "dataset" and "labels" lists at the found index
        dataset["dataset"].pop(index)
        dataset["labels"].pop(index)

        print(len(i))

data1 = np.asarray(dataset["dataset"])
labels1 = np.asarray(dataset["labels"])

X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, labels1, test_size=0.2, shuffle=True, stratify=labels1, random_state=42)

X_train1 = np.concatenate((X_train1, x_missing))
y_train1 = np.concatenate((y_train1, y_missing))

model = RandomForestClassifier()

model.fit(X_train1, y_train1)

y_pred1 = model.predict(X_test1)

score = accuracy_score(y_pred1, y_test1)
score = accuracy_score(y_train1, y_test1)

print(score)
with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model":model}, f)