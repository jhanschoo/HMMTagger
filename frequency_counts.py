#!/usr/bin/env python2

from __future__ import print_function
import sys

if __name__ == "__main__":

    train_file = sys.argv[1]
    dev_test_file = sys.argv[2]

    train_transit_pairs = {}
    train_emit_pairs = {}
    dev_transit_pairs = {}
    dev_emit_pairs = {}

    transit_map = {}
    emit_map = {}

    t = open(train_file)
    trl = tuple(tuple(wt.rsplit("/", 1)) for wt in t.read().split())
    t.close()
    train_length = len(trl)
    dev = open(dev_test_file)
    devl = tuple(tuple(wt.rsplit("/", 1)) for wt in dev.read().split())
    test_length = len(devl)
    print(train_length, test_length)
    prev_tag = None
    for token, tag in trl:
        transit_pair = (prev_tag, tag)
        train_transit_pairs[transit_pair] = train_transit_pairs.get(transit_pair, 0) + 1
        prev_tag = tag

        emit_pair = (tag, token)
        train_emit_pairs[emit_pair] = train_emit_pairs.get(emit_pair, 0) + 1
    prev_tag = None
    for token, tag in devl:
        transit_pair = (prev_tag, tag)
        dev_transit_pairs[transit_pair] = dev_transit_pairs.get(transit_pair, 0) + 1
        prev_tag = tag

        emit_pair = (tag, token)
        dev_emit_pairs[emit_pair] = dev_emit_pairs.get(emit_pair, 0) + 1

    for t_pair, count in train_transit_pairs.items():
        if transit_map.get(count) is None:
            transit_map[count] = [0.0, 0.0]
        transit_map[count][1] += 1
        if dev_transit_pairs.get(t_pair):
            transit_map[count][0] += dev_transit_pairs[t_pair]
            del dev_transit_pairs[t_pair]
    transit_map[0] = [sum(dev_transit_pairs.values()), len(dev_transit_pairs.values())]

    for e_pair, count in train_emit_pairs.items():
        if emit_map.get(count) is None:
            emit_map[count] = [0.0, 0.0]
        emit_map[count][1] += 1
        if dev_emit_pairs.get(e_pair):
            emit_map[count][0] += dev_emit_pairs[e_pair]
            del dev_emit_pairs[e_pair]
    emit_map[0] = [sum(dev_emit_pairs.values()), len(dev_emit_pairs.values())]

    print("transit counts:")
    sum_a = sum_b = 0
    transit_discount = 0
    for count in sorted(transit_map.keys()):
        dev_count = transit_map[count][0]/transit_map[count][1] * train_length / test_length
        sum_a += count
        sum_b += dev_count
        print(count, dev_count, transit_map[count][1])
    sum_c = sum_d = 0
    print("emit counts:")
    for count in sorted(emit_map.keys()):
        dev_count = emit_map[count][0]/emit_map[count][1] * train_length / test_length
        sum_c += count
        sum_d += dev_count
        print(count, dev_count, emit_map[count][1])
    print(sum_a, sum_b)
    print(sum_c, sum_b)
    print(sum_a-sum_b, sum_c-sum_d)
