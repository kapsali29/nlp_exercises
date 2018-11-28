import numpy as np

print("Load pre trained vectors")

word2vec = {}
idx2word = []
embeddings = []
with open("large_files/glove.6B.50d.txt") as file:
    lines = file.readlines()
    for line in lines:
        splitted_line = line.split()
        word = splitted_line[0]
        vector = np.asarray(splitted_line[1:], dtype='float32')
        idx2word.append(word)
        embeddings.append(vector)
        word2vec[word] = vector

embeddings_matrix = np.vstack(embeddings)
D = len(embeddings)


def dist1(a, b):
    return np.linalg.norm(a - b)


def find_analogies(w1, w2, w3):
    if w1 not in word2vec.keys() and w2 not in word2vec.keys() and w3 not in word2vec.keys():
        print("the set of words are not in word2vec")
        pass
    else:
        king = word2vec[w1]
        man = word2vec[w2]
        woman = word2vec[w3]
        king_index = idx2word.index(w1)
        man_index = idx2word.index(w2)
        woman_index = idx2word.index(w3)
        v0 = king - man + woman
        dist = 100
        pos = 0
        for i in range(0, D):
            if i == king_index or i == man_index or i == woman_index:
                continue
            else:
                new_dist = dist1(v0, embeddings_matrix[i, :])
                if new_dist <= dist:
                    dist = new_dist
                    pos = idx2word[i]
        print("{} - {} = {} - {}".format(w3, w2, pos, w1))
        pass


find_analogies('king', 'man', 'woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english')
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('japan', 'japanese', 'australian')
find_analogies('december', 'november', 'june')
find_analogies('miami', 'florida', 'texas')
find_analogies('einstein', 'scientist', 'painter')
find_analogies('china', 'rice', 'bread')
find_analogies('man', 'woman', 'she')
find_analogies('man', 'woman', 'aunt')
find_analogies('man', 'woman', 'sister')
find_analogies('man', 'woman', 'wife')
find_analogies('man', 'woman', 'actress')
find_analogies('man', 'woman', 'mother')
find_analogies('heir', 'heiress', 'princess')
find_analogies('nephew', 'niece', 'aunt')
find_analogies('france', 'paris', 'tokyo')
find_analogies('france', 'paris', 'beijing')
find_analogies('february', 'january', 'november')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
