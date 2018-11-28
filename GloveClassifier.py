import numpy as np


class GloveClassifier(object):

    def __init__(self, source):
        self.source = source
        self.embeddings, self.word2vec, self.idx2word = self.read_glove_source()

    def read_glove_source(self):
        """
        The following function reads the Glove pre trained embeddings from a given source/file
        and returns the embeddings, the words and their correspondence
        :return:
        """
        embeddings = []
        word2vec = {}
        idx2word = []
        with open(self.source) as file:
            lines = file.readlines()
            for line in lines:
                data = line.split()
                word = data[0]
                vector = np.asarray(data[1:], dtype='float32')
                embeddings.append(vector)
                idx2word.append(word)
                word2vec[word] = vector
        return embeddings, word2vec, idx2word

    def fit_transform(self, data_list):
        """
        The following function receives the data, the data format is list of lists and transform them
         to tf idf bow model.

        :param data: list of lists
        :return: bow
        """
        bow = []
        for sentence in data_list:
            words = sentence.split()
            sentence_bow = [self.word2vec[word] for word in words if word in self.word2vec.keys()]
            sentence_score = sum(sentence_bow)/len(sentence)
            bow.append(sentence_score)
        return bow
