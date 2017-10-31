from __future__ import print_function

import math
import pickle

# We use enumerated arrays instead of a dict to key statistics for
# better access times. Note that for anything but small corpora this
# strategy is infeasible due to the amount of memory required.
INF = float("Inf")
COUNT = 0
SUM_TRANSIT_COUNT = 1
SUM_TRANSIT_NUM_FOLLOWS = 2
NUM_TRANSIT_WITH_FOLLOWS = 3
NUM_TRANSIT = 4
NUM_FOLLOWS = 5
TRANSIT = 6
FOLLOWS = 7

EMIT_COUNT = 0
SUM_EMIT_NUM_FOLLOWS = 1
SUM_EMIT = 2
NUM_EMIT = 3
EMIT = 4

INV_COUNT = 0
INV_NUM_FOLLOWS = 1
INV_FOLLOWS = 2

class HMMTagger:
    """

    """

    def __init__(self, n, transit_discount=0.75, emit_discount=0.5):
        self.ngrams = {}
        self.emits = {}
        self.inverse_emits = {}
        self.max_n = n
        self.transit_discount = transit_discount
        self.emit_discount = emit_discount
        self.tag = self.viterbi

        # closure in unknown context and n-grams
        #unkl = [self._get_stats(("<UNK>",) * i) for i in range(n)]

    def _get_stats(self, ngram):
        """
        Return the transition statistics for 'ngram', or initialize it
        and return it if not present
        """
        if self.ngrams.get(ngram) is None:
            self.ngrams[ngram] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {
            }, {}]
        return self.ngrams[ngram]

    def _get_emit_stats(self, tag):
        """
        Return the emission statistics for tag, or initialize it and
        return it if not already present.
        """
        if self.emits.get(tag) is None:
            self.emits[tag] = [
                0.0,
                0.0,
                0.0,
                0.0,
                {}
            ]
        return self.emits[tag]

    def _get_inverse_emit_stats(self, token):
        """
        Return the statistics record for token, or initialize it and
        return it if not already present.
        """
        if self.inverse_emits.get(token) is None:
            self.inverse_emits[token] = [0.0,0.0,{}]
        return self.inverse_emits[token]

    def _learn_ngram(self, ngram):
        """
        Record an observation of a transition to a given tag after
        given context, where tag is the last element of the 'ngram'
        and the given context are the other elements.
        """
        context, tag = ngram[:-1], ngram[-1]
        context_stats = self._get_stats(context)
        ngram_stats = self._get_stats(ngram)

        # record ngram counts
        ngram_stats[COUNT] += 1.0
        context_stats[SUM_TRANSIT_COUNT] += 1.0

        # record transit counts
        if context_stats[TRANSIT].get(tag) is None:
            context_stats[NUM_TRANSIT] += 1.0
            context_stats[TRANSIT][tag] = 0.0
        context_stats[TRANSIT][tag] += 1.0

        # record follows for continuation counts (for the ngram
        # with the original ngram's first tag removed)
        pre, ngram = ngram[0], ngram[1:]
        if not ngram:
            return
        context, tag = ngram[:-1], ngram[-1]
        # ngram has at least one element
        ngram_stats = self._get_stats(ngram)
        context_stats = self._get_stats(context)
        if not ngram_stats[FOLLOWS]:
            context_stats[NUM_TRANSIT_WITH_FOLLOWS] += 1.0
        if ngram_stats[FOLLOWS].get(pre) is None:
            ngram_stats[NUM_FOLLOWS] += 1.0
            context_stats[SUM_TRANSIT_NUM_FOLLOWS] += 1.0
            ngram_stats[FOLLOWS][pre] = 0.0
        ngram_stats[FOLLOWS][pre] += 1.0

    def _learn_emit(self, tag, token):
        """
        Record an emission observation of token by tag.
        """
        token_stats = self._get_inverse_emit_stats(token)
        token_stats[INV_COUNT] += 1.0
        is_new_tag = False
        if token_stats[INV_FOLLOWS].get(tag) is None:
            token_stats[INV_NUM_FOLLOWS] += 1.0
            token_stats[INV_FOLLOWS][tag] = 0.0
            is_new_tag = True
        token_stats[INV_FOLLOWS][tag] += 1.0
        for stats in [self._get_emit_stats(tag), self._get_emit_stats(())]:
            stats[EMIT_COUNT] += 1.0
            if stats[EMIT].get(token) is None:
                stats[NUM_EMIT] += 1.0
                stats[EMIT][token] = 0.0
            stats[EMIT][token] += 1.0
            stats[SUM_EMIT] += 1.0
            if is_new_tag:
                stats[SUM_EMIT_NUM_FOLLOWS] += 1.0

    def learn(self, state_ngram, token):
        """
        Observe a given tag appearing in a given context emitting
        'token', where the given tag is the last element of
        'state_ngram' and the given context are the other elements.
        """
        for i in range(len(state_ngram) - 1, -1, -1):
            self._learn_ngram(state_ngram[i:])
        self._learn_emit(state_ngram[-1], token)

    def mle_count(self, ngram):
        """
        Returns the number of times 'ngram' was observed.
        """
        return self._get_stats(ngram)[COUNT]

    def mle_count_normalizer(self, context):
        """
        Returns the number of observed N-grams that have 'context' as
        their (N-1)-gram prefix.
        """
        return self._get_stats(context)[SUM_TRANSIT_COUNT]

    def types_discounted_under_mle_count(self, context):
        """
        Returns the number of distinct ngrams observed that have
        'context' as their (N-1)-gram prefix.
        """
        return self._get_stats(context)[NUM_TRANSIT]

    def mle_transit_probability(self, context_and_tag_ngram):
        """
        Returns the probability of a given tag occuring in a given
        context, using the MLE model on the observations. The given tag
        is the final element of 'context_and_tag_ngram' and the given
        context is are the preceding elements.
        """
        # should not happen, but included for consistency
        if context_and_tag_ngram == ():
            return 1.0 / (self._get_stats(())[NUM_TRANSIT] + 1.0)
        context = context_and_tag_ngram[:-1]
        normalizer = self.mle_count_normalizer(context)
        if normalizer <= 0.0:
            return self.mle_transit_probability(context_and_tag_ngram[1:])
        return self.mle_count(context_and_tag_ngram) / normalizer

    def cont_count(self, ngram):
        """
        Returns the distinct number of single-gram contexts 'ngram'
        was observed to follow.
        """
        return self._get_stats(ngram)[NUM_FOLLOWS]

    def cont_count_normalizer(self, context):
        """
        Returns the sum of the 'cont_count's of all N-grams that have
        'context' as their (N-1)-gram prefix.
        """
        return self._get_stats(context)[SUM_TRANSIT_NUM_FOLLOWS]

    def types_discounted_under_cont_count(self, context):
        """
        Returns the number of distinct ngrams that follows some
        single-gram context and that observed that have
        'context' as their (N-1)-gram prefix.
        """
        return self._get_stats(context)[NUM_TRANSIT_WITH_FOLLOWS]

    def kn_count(self, ngram, size):
        """
        Returns 'mle_count' if 'size' is the N-gram size of the HMM,
        else returns 'cont_count'.
        """
        if size == self.max_n:
            return self.mle_count(ngram)
        return self.cont_count(ngram)

    def kn_count_normalizer(self, context, size):
        """
        Returns 'mle_count_normalizer' if 'size' is the N-gram size of
        the HMM, else returns 'cont_count_normalizer'.
        """
        if size == self.max_n:
            return self.mle_count_normalizer(context)
        return self.cont_count_normalizer(context)

    def kn_types_discounted(self, context, size):
        """
        Returns 'types_discounted_under_mle_count' or
        'types_discounted_under_cont_count'
        based on 'size'
        """
        if size == self.max_n:
            return self.types_discounted_under_mle_count(context)
        return self.types_discounted_under_cont_count(context)

    def kn_transit_probability(self, context_and_tag_ngram):
        """
        Returns the Kneser-Ney smoothed probability of a given tag
        occuring in a given context. The given tag is the final element
        of 'context_and_tag_ngram' and the given context is are the
        preceding elements.
        """
        if context_and_tag_ngram == ():
            # add 1 for <UNK>
            return 1.0 / (self._get_stats(())[NUM_TRANSIT] + 1.0)
        context, size = context_and_tag_ngram[:-1], len(context_and_tag_ngram)
        normalizer = self.kn_count_normalizer(context, size)
        if normalizer <= 0.0:
            return self.kn_transit_probability(context_and_tag_ngram[1:])
        discounted_count = max(self.kn_count(
            context_and_tag_ngram, size) - self.transit_discount, 0)
        l_weight = self.transit_discount * self.kn_types_discounted(context, size)
        return (discounted_count + l_weight * self.kn_transit_probability(context_and_tag_ngram[1:])) / normalizer

    def words_discounted(self, tag):
        """
        Returns the number of distinct words observed that have
        'tag' as their prefix.
        """
        return self._get_emit_stats(tag)[NUM_EMIT]

    def emit_mle_count(self, tag, token):
        """
        Returns the number of tokens of type 'token' observed to
        be emitted by 'tag'.
        """
        return self._get_emit_stats(tag)[EMIT].get(token, 0.0)

    def emit_mle_count_normalizer(self, tag):
        """
        Returns the total number of tokens observed to be emitted by
        'tag'.
        """
        return self._get_emit_stats(tag)[EMIT_COUNT]

    def emit_token_mle_count(self, token):
        """
        Returns the total number of tokens of type 'token' observed
        to be emitted.
        """
        return self._get_inverse_emit_stats(token)[INV_COUNT]

    def emit_token_mle_count_normalizer(self):
        """
        Returns the total number of tokens emitted.
        """
        return self._get_emit_stats(())[SUM_EMIT]

    def emit_mle_probability(self, tag, token):
        """
        Returns the MLE probability of 'token' being emitted by 'tag'.
        """
        normalizer = self.emit_mle_count_normalizer(tag)
        if normalizer <= 0.0:
            token_normalizer = self.emit_token_mle_count_normalizer()
            # should not happen, but included for consistency
            if token_normalizer <= 0.0:
                return 1.0 / (self.words_discounted(()) + 1.0)
            return self.emit_token_mle_count(token) / self.emit_token_mle_count_normalizer()
        return self.emit_mle_count(tag, token) / self.emit_mle_count_normalizer(tag)

    def emit_token_cont_count(self, token):
        """
        Returns the number of distinct tags that 'token' is observed
        to follow.
        """
        return self._get_inverse_emit_stats(token)[INV_NUM_FOLLOWS]

    def emit_token_cont_count_normalizer(self):
        """
        Returns the sum of the number of distinct tags that we have
        observed each token to be emitted by.
        """
        return self._get_emit_stats(())[SUM_EMIT_NUM_FOLLOWS]

    def emit_kn_probability(self, tag, token):
        """
        Returns the Kneser-Ney smoothed count of 'token' being
        emitted by 'tag'
        """
        # configure uniform interpolation
        uniform = 1.0 / (self.words_discounted(()) + 1.0)
        # configure zeroth order terms on continuation counts
        zeroth_normalizer = self.emit_token_cont_count_normalizer()
        if zeroth_normalizer <= 0.0:
            zeroth_prob = uniform
        else:
            zeroth_term = max(self.emit_token_cont_count(token) -
                              self.emit_discount, 0)
            zeroth_weight = self.emit_discount * self.words_discounted(())
            zeroth_prob = (zeroth_term + zeroth_weight * uniform) / zeroth_normalizer
        # configure first order Markov chain on MLE counts
        first_normalizer = self.emit_mle_count_normalizer(tag)
        if first_normalizer <= 0.0:
            return zeroth_prob
        else:
            first_term = max(self.emit_mle_count(tag, token) -
                             self.emit_discount, 0)
            first_weight = self.emit_discount * self.words_discounted(tag)
        return (first_term + first_weight * zeroth_prob) / first_normalizer

    def viterbi(self, token_seq):
        context_probs = {(): 0.0}
        context_backtrace = {(): []}
        i = 0
        l = len(token_seq)
        # advance by one transition each token
        for token in token_seq:
            i+=1
            print(i, "/", l)
            probs = {}
            backtrace = {}
            # consider all contexts currently being considered
            for context, context_prob in context_probs.items():
                dummy_context = context
                # closure over unseen context; regress to smaller context
                while dummy_context and self._get_stats(dummy_context)[NUM_TRANSIT] == 0:
                    dummy_context = dummy_context[1:]
                # evaluate the logprob associated with each observed transition from the context
                # currently being considered, and remember the max of them
                for candidate_tag in self.ngrams[dummy_context][TRANSIT].keys():
                    candidate_prob = math.log(self.emit_kn_probability(candidate_tag, token), 2) + math.log(self.kn_transit_probability(context+(candidate_tag,)), 2) + context_prob
                    # slide back up the context window
                    candidate_context = (context + (candidate_tag,))[-(self.max_n - 1):]
                    self._get_stats(candidate_context)
                    if probs.get(candidate_context) is None or probs[candidate_context] < candidate_prob:
                        probs[candidate_context] = candidate_prob
                        # we "append" by constructing a new list with a reference so that this is
                        # O(1) in the length of the cain, not O(n).
                        backtrace[candidate_context] = [context_backtrace[context], candidate_tag]
            context_probs = probs
            context_backtrace = backtrace

        # in the final step we choose the most probable chain
        max_candidate = ((), -INF)
        for context, prob in context_probs.items():
            if prob > max_candidate[1]:
                max_candidate = (context, prob)
        # we flatten the backtrace into a shallow list
        backtrace_list = []
        backtrace = context_backtrace[max_candidate[0]]
        while len(backtrace) > 0:
            backtrace_list.append(backtrace[1])
            backtrace = backtrace[0]
        backtrace_list.reverse()
        return backtrace_list

    def write_model(self, filename):
        with open(filename, 'w+') as file:
            pickle.dump([
                self.ngrams,
                self.emits,
                self.inverse_emits,
                self.max_n,
                self.transit_discount,
                self.emit_discount
            ], file)

    def read_model(self, filename):
        with open(filename, 'r') as file:
            [
                self.ngrams,
                self.emits,
                self.inverse_emits,
                self.max_n,
                self.transit_discount,
                self.emit_discount
            ] = pickle.load(file)

# Ad-hoc self-tests
if __name__ == '__main__':
    def should_be(a, b, s=""):
        """Helper function to assert state on tests"""
        if a == b:
            return
        print("'", a, "' should be '", b, "' ", s, sep='')

    # zero tags test

    unk = "UNK"
    A = "A"
    y = "y"
    z = "z"
    unkt = (unk,)
    unktt = (unk,) * 2
    unkttt = (unk,) * 3
    unka = (unk, A)
    At = (A,)
    Ay = (A, y)

    hmm = HMMTagger(3, 0.75, 0.5)
    should_be(hmm.mle_count(unkt), 0.0, "hmm.mle_count(unkt)")
    should_be(hmm.mle_count_normalizer(unkt), 0.0, "hmm.mle_count_normalizer(unkt)")
    should_be(hmm.types_discounted_under_mle_count(unkt), 0.0, "hmm.types_discounted_under_mle_count(unkt)")
    should_be(hmm.mle_transit_probability(unkt), 1.0, "hmm.mle_transit_probability(unkt)")
    should_be(hmm.cont_count(unkt), 0.0, "hmm.cont_count(unkt)")
    should_be(hmm.cont_count_normalizer(unkt), 0.0, "hmm.cont_count_normalizer(unkt)")
    should_be(hmm.types_discounted_under_cont_count(unkt), 0.0, "hmm.types_discounted_under_cont_count(unkt)")
    should_be(hmm.kn_transit_weight((), 1), 1.0, "hmm.kn_transit_weight((), 1)")
    should_be(hmm.kn_transit_weight(unkt, 2), 1.0, "hmm.kn_transit_weight(unkt, 2)")
    should_be(hmm.kn_transit_probability(unkt), 1.0, "hmm.kn_transit_probability(unkt)")
    should_be(hmm.words_discounted(unk), 0.0, "hmm.words_discounted(unk)")
    should_be(hmm.emit_mle_count(unk, z), 0.0, "hmm.emit_mle_count(unk, z)")
    should_be(hmm.emit_mle_count_normalizer(unk), 0.0, "hmm.emit_mle_count_normalizer(unk)")
    should_be(hmm.emit_mle_probability(unk, z), 1.0, "hmm.emit_mle_probability(unk, z)")
    should_be(hmm.emit_token_mle_count(z), 0.0, "hmm.emit_token_mle_count(z)")
    should_be(hmm.emit_token_mle_count_normalizer(), 0.0, "hmm.emit_token_mle_count_normalizer()")
    should_be(hmm.emit_token_cont_count(z), 0.0, "hmm.emit_token_cont_count(z)")
    should_be(hmm.emit_token_cont_count_normalizer(), 0.0, "hmm.emit_token_cont_count_normalizer()")
    should_be(hmm.kn_emit_probability(unk, z), 1.0, "hmm.kn_emit_probability(unk, z)")
    #print(hmm.viterbi(("n", "n", "n",)))

    # single count test
    hmm.learn(At, y)
    should_be(hmm.mle_count(unkt), 0.0, "hmm.mle_count(unkt)")
    should_be(hmm.mle_count(At), 1.0, "hmm.mle_count(At)")
    should_be(hmm.mle_count_normalizer(unkt), 0.0, "hmm.mle_count_normalizer(unkt)")
    should_be(hmm.mle_count_normalizer(()), 1.0, "hmm.mle_count_normalizer(()))")
    should_be(hmm.types_discounted_under_mle_count(unkt), 0.0, "hmm.types_discounted_under_mle_count(unkt)")
    should_be(hmm.types_discounted_under_mle_count(()), 1.0, "hmm.types_discounted_under_mle_count(())")
    should_be(hmm.mle_transit_probability(unkt), 0.0, "hmm.mle_transit_probability(unkt)")
    should_be(hmm.mle_transit_probability(At), 1.0, "hmm.mle_transit_probability(At)")
    should_be(hmm.cont_count(unkt), 0.0, "hmm.cont_count(unkt)")
    should_be(hmm.cont_count(At), 0.0, "hmm.cont_count(At)")
    should_be(hmm.cont_count_normalizer(unkt), 0.0, "hmm.cont_count_normalizer(unkt)")
    should_be(hmm.cont_count_normalizer(()), 0.0, "hmm.cont_count_normalizer(())")
    should_be(hmm.types_discounted_under_cont_count(unkt), 0.0, "hmm.types_discounted_under_cont_count(unkt)")
    should_be(hmm.types_discounted_under_cont_count(()), 0.0, "hmm.types_discounted_under_cont_count()")
    should_be(hmm.kn_transit_weight((), 1), 1.0, "hmm.kn_transit_weight((), 1)")
    should_be(hmm.kn_transit_weight(unkt, 2), 1.0, "hmm.kn_transit_weight(unkt, 2)")
    should_be(hmm.kn_transit_probability(unkt), 0.5, "hmm.kn_transit_probability(unkt)")
    should_be(hmm.kn_transit_probability(At), 0.5, "hmm.kn_transit_probability(At)")
    should_be(hmm.words_discounted(unk), 0.0, "hmm.words_discounted(unk)")
    should_be(hmm.words_discounted(A), 1.0, "hmm.words_discounted(A)")
    should_be(hmm.emit_mle_count(unk, z), 0.0, "hmm.emit_mle_count(unk, z)")
    should_be(hmm.emit_mle_count(A, z), 0.0, "hmm.emit_mle_count(A, z)")
    should_be(hmm.emit_mle_count(A, y), 1.0, "hmm.emit_mle_count(A, y)")
    should_be(hmm.emit_mle_count_normalizer(unk), 0.0, "hmm.emit_mle_count_normalizer(unk)")
    should_be(hmm.emit_mle_count_normalizer(A), 1.0, "hmm.emit_mle_count_normalizer(A)")
    should_be(hmm.emit_mle_probability(unk, z), 0.0, "hmm.emit_mle_probability(unk, z)")
    should_be(hmm.emit_mle_probability(unk, y), 1.0, "hmm.emit_mle_probability(unk, y)")
    should_be(hmm.emit_mle_probability(A, z), 0.0, "hmm.emit_mle_probability(A, z)")
    should_be(hmm.emit_mle_probability(A, y), 1.0, "hmm.emit_mle_probability(A, y)")
    should_be(hmm.emit_token_mle_count(z), 0.0, "hmm.emit_token_mle_count(z)")
    should_be(hmm.emit_token_mle_count(y), 1.0, "hmm.emit_token_mle_count(y)")
    should_be(hmm.emit_token_mle_count_normalizer(), 1.0, "hmm.emit_token_mle_count_normalizer()")
    should_be(hmm.emit_token_cont_count(z), 0.0, "hmm.emit_token_cont_count(z)")
    should_be(hmm.emit_token_cont_count(y), 1.0, "hmm.emit_token_cont_count(z)")
    should_be(hmm.emit_token_cont_count_normalizer(), 1.0, "hmm.emit_token_cont_count_normalizer()")
    # 0 + 0.5*0.5
    should_be(hmm.kn_emit_probability(unk, z), 0.25, "hmm.kn_emit_probability(unk, z)")
    # 0.5 + 0.5*0.5
    should_be(hmm.kn_emit_probability(unk, y), 0.75, "hmm.kn_emit_probability(unk, y)")
    # 0.5*0.5*0.5
    should_be(hmm.kn_emit_probability(A, z), 0.5*0.5*0.5, "hmm.kn_emit_probability(A, z)")
    # 0.5+0.5*(0.5+0.5*0.5)
    should_be(hmm.kn_emit_probability(A, y), 0.5+0.5*(0.5+0.5*0.5), "hmm.kn_emit_probability(A, y)")
    print(hmm.viterbi(("y", "y", "n")))
    #print(hmm.viterbi(("y", "y", "n")))

    hmm.learn(("A", "B"), "y")
    hmm.learn(("A", "B", "B"), "z")
    hmm.learn(("B", "B","A"), "y")
    should_be(hmm.kn_emit_probability("A", "y")+hmm.kn_emit_probability("A", "z")+hmm.kn_emit_probability("A", "n"), 1.0)
    should_be(hmm.kn_transit_probability(("UNK", "UNK", "A")), hmm.kn_transit_probability(("A",)))
    should_be(hmm.kn_transit_probability(("B", "UNK", "A")), hmm.kn_transit_probability(("A",)))
    should_be(hmm.kn_transit_probability(("UNK", "B", "A")), hmm.kn_transit_probability(("B", "A",)))
    print("===============")
    print("context", "('UNK',)", "('A',)", "('B')")
    tags = ["UNK", "A", "B"]
    context = [(), ("A",), ("B",), ("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    for c in context:
        probs = [hmm.kn_transit_probability(c+(t,)) for t in tags]
        print(c, probs, sum(probs))
    words = ["n", "y", "z"]
    print("tag", "n", "y", "z")
    for t in tags:
        probs = [hmm.kn_emit_probability(t, w) for w in words]
        print(t, probs, sum(probs))
    print(hmm.viterbi(("y","z","n","z","y","y")))

    hmm.write_model("temp.model")
    hmm2 = HMMTagger(0.75, 0.75)
    hmm2.read_model("temp.model")
    print(hmm2.viterbi(("y","z","n","z","y","y")))
