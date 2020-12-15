from collections import defaultdict


def calc_is_one_cluster(t1, t2, clusters, k2ind):
    st1 = set(t1)
    st2 = set(t2)
    k1 = tuple(sorted(st1))
    k2 = tuple(sorted(st2))
    c1 = k2ind.get(k1)
    c2 = k2ind.get(k2)
    if c1 is not None and c2 is not None and c1 == c2:
        return 1
    return 0


def calc_cnt_negative(t1, t2, clusters, k2ind, cl2cl_neg):
    st1 = set(t1)
    st2 = set(t2)
    k1 = tuple(sorted(st1))
    k2 = tuple(sorted(st2))
    c1 = k2ind.get(k1)
    c2 = k2ind.get(k2)
    if c1 is not None and c2 is not None:
        return cl2cl_neg.get((c1, c2), 0)
    return -1


def get_cluster_size(t1, clusters, k2ind):
    st1 = set(t1)
    k1 = tuple(sorted(st1))
    c1 = k2ind.get(k1)
    return len(clusters[c1]) if c1 is not None else 0


def prepare(train, use_simple=False):
    clusters = list()
    k2ind = dict()

    freq = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    cl2cl_neg = defaultdict(int)
    for t1, t1s, t2, t2s, y in zip(train['name_1_tokens'], train['name_1_tokens_simple'],
                                   train['name_2_tokens'], train['name_2_tokens_simple'],
                                   train['is_duplicate']):
        if not use_simple:
            st1 = set(t1)
            st2 = set(t2)
        else:
            st1 = t1s
            st2 = t2s

        k1 = tuple(sorted(st1))
        k2 = tuple(sorted(st2))
        c1 = k2ind.get(k1)
        c2 = k2ind.get(k2)
        if y == 1:
            if c1 is not None:
                if c2 is not None:
                    if c1 != c2:
                        clusters[c1] |= clusters[c2]
                        for e in clusters[c2]:
                            k2ind[e] = c1
                        clusters[c2] = set()
                else:
                    clusters[c1] |= set([k2])
                    k2ind[k2] = c1
            else:
                if c2 is not None:
                    clusters[c2] |= set([k1])
                    k2ind[k1] = c2
                else:
                    clusters.append({k1, k2})
                    k2ind[k1] = len(clusters) - 1
                    k2ind[k2] = len(clusters) - 1

        for w1 in t1:
            freq[w1][0] += 1
            if w1 not in t2:
                freq[w1][1 + y] += 1
            else:
                freq[w1][3 + y] += 1
        for w2 in t2:
            freq[w2][0] += 1
            if w2 not in t1:
                freq[w2][1 + y] += 1
            else:
                freq[w2][3 + y] += 1

    for t1, t1s, t2, t2s, y in zip(train['name_1_tokens'], train['name_1_tokens_simple'],
                                   train['name_2_tokens'], train['name_2_tokens_simple'],
                                   train['is_duplicate']):
        if not use_simple:
            st1 = set(t1)
            st2 = set(t2)
        else:
            st1 = t1s
            st2 = t2s
        k1 = tuple(sorted(st1))
        k2 = tuple(sorted(st2))
        c1 = k2ind.get(k1)
        c2 = k2ind.get(k2)
        if y == 0:
            cl2cl_neg[(c1, c2)] += 1
            cl2cl_neg[(c2, c1)] += 1

    for c in clusters:
        cc = list(c)
        for i1 in range(len(c)):
            for i2 in range(i1 + 1, len(c)):
                for t1, t2 in [(cc[i1], cc[i2]), (cc[i2], cc[i1])]:
                    for w1 in t1:
                        if w1 not in t2:
                            freq[w1][5] += 1
                        else:
                            freq[w1][6] += 1

    return clusters, k2ind, freq, cl2cl_neg
