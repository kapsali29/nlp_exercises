from math import log
from random import randint, sample

import nltk
from nltk.corpus import brown

from collections import Counter


class BigramModel(object):

    def __init__(self):
        self.corpus = [word.lower() for word in brown.words()]
        self.unique_words = Counter(self.corpus)
        self.V = len(self.unique_words)
        self.bi_grams = list(nltk.bigrams(self.corpus))
        self.v_bi_gram = Counter(self.bi_grams)

    def unigram_probability(self, word):
        """
        The following function is used to calculate unigram probabilities e.g p(A)=count(A)+1/V

        >>> unigram_probability("flower")
        :param word: a string
        :return: probability
        """
        prob = (self.unique_words[word] + 1) / (self.V)
        return prob

    def bigram_probability(self, bigram):
        """
        The following function receives a tuple of 2 words: bigram and calculates the
        joint probability of that bigram based to brown corpus e.g
        p(B|A) = count(A->B)+1/count(A)+V

        Example Usage
        >>> bigram_probability(('the','flower'))

        :param bigram: bigram as tuple
        :return: smoothed probability
        """
        prob = (self.v_bi_gram[bigram] + 1) / (self.unique_words[bigram[0]] + self.V)
        return prob

    def sentence_probability(self, sentence):
        """
        The following function calculates the sentence probability
        log(p(A,B,C,D,E))/T=log(p(A)p(B|A)p(C|B)p(D|C)p(E|D))/T

        :param sentence: list of sentence tokens
        :return:
        """
        sentence_corpus = [word.lower() for word in sentence]
        first_word = sentence_corpus[0]
        sentence_length = len(sentence_corpus)
        sentence_bigrams = list(nltk.bigrams(sentence_corpus))
        sentence_prob = log(self.unigram_probability(first_word))
        for bigram in sentence_bigrams:
            sentence_prob += log(self.bigram_probability(bigram))
        return sentence_prob / sentence_length

    def random_words_test(self):
        """
        The following function compares the probability of a real sentence from brown corpus
        against a fake sentence (random generated words)

        :return: pass
        """
        sents = brown.sents()
        brown_sent = sents[randint(0, len(sents))]
        random_words = sample(self.unique_words.keys(), len(brown_sent))
        print(" Random sentence probability is {}".format(self.sentence_probability(random_words)))
        print(" Brown sentence probability is {}".format(self.sentence_probability(brown_sent)))
        pass

    def test_input(self):
        """
        The following function is used to test a sentence you type against a real sentece from the Brown corpus

        :return: pass
        """
        sents = brown.sents()
        brown_sent = sents[randint(0, len(sents))]
        my_sentence = input(" Please type a sentence: ")
        my_sentence_corpus = my_sentence.split()
        print(" My sentence probability is {}".format(self.sentence_probability(my_sentence_corpus)))
        print(" Brown sentence probability is {}".format(self.sentence_probability(brown_sent)))
        pass


sents = brown.sents()
obj = BigramModel()
obj.test_input()
