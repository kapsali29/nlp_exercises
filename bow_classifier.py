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


read_train_data()
