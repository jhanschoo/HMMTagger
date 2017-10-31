from __future__ import print_function
import random
import sys

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

    """
    def train_naive_model(self, model):
        for token, tag in self.pairs:
            model.learn(tag, token)
    """

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
    train_file = sys.argv[1]
    dev_test_file = sys.argv[2]
    model_file = sys.argv[3]
    bt = BuildTagger(train_file, dev_test_file, model_file)
    hmm = HMMTagger.HMMTagger(2)
    params = [0.000000000001, 0.00000001,
              0.0001, 0.01, 0.25, 0.5, 0.75, 0.87, 0.93]
    test_tags = bt.parse_test_data(dev_test_file)
    bt.train_model(hmm)
    sampled = [test_tags[i:i + 100]
               for i in [random.randint(0, len(test_tags) - 100) for _ in range(10)]]
    sampled = [item for sublist in sampled for item in sublist]  # flatten
    max_emit = max_transit = 0.000000000001
    min_error = 1
    for emit_d in params:
        for transit_d in params:
            hmm.transit_discount = transit_d
            hmm.emit_discount = emit_d
            error = bt.test(hmm, sampled)
            if error < min_error:
                min_error = error
                max_transit = transit_d
                max_emit = emit_d
    hmm.transit_discount = max_transit
    hmm.emit_discount = max_emit
    hmm.write_model(model_file)
