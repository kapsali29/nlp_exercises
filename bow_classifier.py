from sklearn.ensemble import RandomForestClassifier

from GloveClassifier import GloveClassifier


def read_train_data():
    labels = []
    contents = []
    with open("large_files/r8-train-all-terms.txt") as file:
        lines = file.readlines()
        for line in lines:
            data = line.split('\t')
            labels.append(data[0])
            contents.append(data[1])
    return labels, contents


def read_test_data():
    xtest = []
    ytest = []
    with open("large_files/r8-test-all-terms.txt") as file:
        lines = file.readlines()
        for line in lines:
            data = line.split('\t')
            ytest.append(data[0])
            xtest.append(data[1])
    return ytest, xtest



labels, contents = read_train_data()
test_labels, test_contents = read_test_data()

clf = GloveClassifier("large_files/glove.6B.50d.txt")
Xtrain = clf.fit_transform(contents)
Xtest = clf.fit_transform(test_contents)

model = RandomForestClassifier(n_estimators=200)
model.fit(Xtrain, labels)
print("train score:", model.score(Xtrain, labels))
print("test score:", model.score(Xtest, test_labels))