from __future__ import print_function
import random
import sys
import cProfile

import NaiveTagger
import HMMTagger


class BuildTagger():
    pairs = []

    def __init__(self, train_file, dev_test_file, model_file):
        self.train_file = sys.argv[1]
        self.dev_test_file = sys.argv[2]
        self.model_file = sys.argv[3]
        self.pairs = self.parse_tokens(self.train_file)

    def parse_tokens(self, filename):
        with open(filename) as t:
            # corpus is sufficiently small that we can just hold it
            # (and model in memory)
            c = t.read()
            cl = c.split()
            return tuple(tuple(wt.rsplit("/", 1)) for wt in cl)

    def train_naive_model(self, model):
        for token, tag in self.pairs:
            model.learn(tag, token)

    def train_model(self, model, pairs):
        ngram = ()
        i = 0
        l = len(pairs)
        for token, tag in pairs:
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
    train_file = sys.argv[1]
    dev_test_file = sys.argv[2]
    model_file = sys.argv[3]
    bt = BuildTagger(train_file, dev_test_file, model_file)
    test_tags = bt.parse_test_data(train_file)
    with open("kfold", "w+") as f:
        for i in range(10):
            hmm = HMMTagger.HMMTagger(2, 0.0000000000001, 0.0000000000001)
            test_start, test_end = len(bt.pairs) * \
                i // 10, len(bt.pairs) * (i + 1) // 10
            bt.train_model(
                hmm, bt.pairs[0:test_start] + bt.pairs[test_end:len(bt.pairs)])
            error = bt.test(hmm, bt.pairs[test_start:test_end])
            print("k: %d ,accuracy: %.8f" %
                  (i, 1 - error), file=f)
            f.flush()
            print("k: %d ,accuracy: %.8f" %
                  (i, 1 - error))
            sys.stdout.flush()
