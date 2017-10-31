

class NaiveTagger:
    """
    The naive tagger behaves by the following strategy:
    given a token, tag it with the tag that has most often emitted
    that token. If no tags have emitted that token, tag the word with
    the most frequent tag.
    """
    tag_counts = {}
    token_maps = {}
    max_tag = ""
    max_tag_count = -float("Inf")

    def learn(self, tag, token):
        self.tag_counts[tag] = 1 + self.tag_counts.get(tag, 0)
        if self.tag_counts[tag] > self.max_tag_count:
            self.max_tag = tag
            self.max_tag_count = self.tag_counts[tag]

        self.token_maps[token] = self.token_maps.get(token, {})
        self.token_maps[token][tag] = 1 + self.token_maps[token].get(tag, 0)
        token_tag_counts = self.token_maps[token][tag]
        if token_tag_counts > self.token_maps[token].get("", (tag, -float("Inf")))[1]:
            self.token_maps[token][""] = (tag, token_tag_counts)

    def tag_one(self, token):
        if self.token_maps.get(token):
            return self.token_maps[token][""][0]
        else:
            return self.max_tag

    def tag(self, token_seq):
        return tuple(self.tag_one(token) for token in token_seq)
