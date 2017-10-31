from __future__ import print_function
import random
import sys

import HMMTagger


class RunTagger():

    def __init__(self, train_file, dev_test_file, model_file):
        self.test_file = sys.argv[1]
        self.model_file = sys.argv[2]
        self.out_file = sys.argv[3]

    def parse_test_data(self, filename):
        with open(filename) as tf:
            return tf.read().split()

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

    def predict(self, model, token_seq):
        predicted = self.tag_sequence(model, tuple(token_seq))
        tagged = []
        for i in range(len(token_seq)):
            tagged.append(token_seq[i] + "/" + predicted[i])
        return " ".join(tagged)

    def tag_sequence(self, model, sequence):
        return model.tag(sequence)


if __name__ == "__main__":
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    rt = RunTagger(test_file, model_file, out_file)
    hmm = HMMTagger.HMMTagger(2)
    hmm.read_model(model_file)

    test_tokens = rt.parse_test_data(test_file)
    with open(out_file, "w+") as of:
        predicted = rt.predict(hmm, test_tokens)
        of.write(predicted)
