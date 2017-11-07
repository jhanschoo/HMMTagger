from __future__ import print_function
import random
import sys

import HMMTagger


class BuildTagger():
    pairs = []

    def __init__(self, model_file, test_file):
        self.model_file = model_file
        self.test_file = test_file

    def parse_tokens(self, filename):
        with open(filename) as t:
            # corpus is sufficiently small that we can just hold it
            # (and model in memory)
            c = t.read()
            cl = c.split()
            return tuple(tuple(wt.rsplit("/", 1)) for wt in cl)

    def train_model(self, model):
        ngram = ()
        i = 0
        l = len(self.pairs)
        for token, tag in self.pairs:
            print("training ", i, " of ", l)
            i += 1
            ngram = ngram[-model.max_n + 1:] + (tag,)
            model.learn(ngram, token)

    def parse_test_data(self, filename):
        with open(filename) as tf:
            test_pairs = tf.read().split()
            return [test_pair.rsplit("/", 1) for test_pair in test_pairs]

    def test(self, model, test_pairs):
        words = []
        tags = []
        for w, t in test_pairs:
            words.append(w)
            tags.append(t)
        predicted = self.tag_sequence(model, tuple(words))
        error = 0
        for i in range(len(tags)):
            if tags[i] != predicted[i]:
                error += 1
        return float(error) / len(tags)

    def tag_sequence(self, model, sequence):
        return model.tag(sequence)


if __name__ == "__main__":
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    bt = BuildTagger(model_file, test_file)
    hmm = HMMTagger.HMMTagger(2)
    hmm.read_model(model_file)
    test_tags = bt.parse_test_data(test_file)
    print(bt.test(hmm, test_tags))
