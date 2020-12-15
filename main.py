import numpy as np
import scipy as sp
import pandas as pd
import csv

from collections import *

import json
import pathlib
import re
import sys

import spacy

import unidecode
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from translit import translit
from utils import (
    load_legal_countries0,
    simple_transform,
    get_geo,
    multi_str_replace,
    calc_lcs,
    expand_tokens
)
from cluster import prepare, calc_is_one_cluster, calc_cnt_negative, get_cluster_size


###############################################################################
################     STAGE 1: DICTS   #########################################
###############################################################################

DATA_DIR = pathlib.Path("app/data")
train = pd.read_csv(DATA_DIR.joinpath('train.csv'), index_col="pair_id")

print(train.head())
train.head().to_csv('output/foo.csv')


print(translit('блиндер'))
non_alphanum_regex = re.compile("[^0-9a-zA-Zа-яА-ЯёЁ ]+")

legal, countries0, legal_tokens = load_legal_countries0(DATA_DIR)
print('legal', len(legal))
print('legal_tokens', len(legal_tokens))

countries, cities, cities_alt = get_geo(countries0)

legal_re = multi_str_replace([rf"{entity}" for entity in legal | legal_tokens if len(entity) > 1], debug=False)
countries_re = multi_str_replace([rf"{entity}" for entity in countries], debug=False)

t = r'sibur    gmbh inc. gmbh inc pvgmbh  b.v. bova inc.'
t = legal_re.sub('', t)
print(t)


# import en_core_web_md
# nlp_en = en_core_web_md.load()

import en_core_web_lg
nlp_en = en_core_web_lg.load()

docs = dict()
for i, s in enumerate(set(train['name_1']) | set(train['name_2'])):
    docs[s] = nlp_en(s)
    if i % 1000 == 0:
        print(i)

all_tokens = Counter()
for e in docs:
    all_tokens.update(simple_transform(e).split())
len(all_tokens)

docs_tokens = dict()
for i, token in enumerate(all_tokens):
    docs_tokens[token] = nlp_en(token)
    if i % 1000 == 0:
        print(i, len(all_tokens))

geo_add = set()
kk = 0
for token in docs_tokens:
    doc = docs_tokens[token]
    for e in doc.ents:
        if e.label_ == 'GPE':
            geo_add.add(token)
            if token not in cities_alt and token not in countries:
                #print(token)
                kk += 1

print(len(geo_add), kk)
geo_re = multi_str_replace([rf"{entity}" for entity in geo_add], debug=False)


###############################################################################
################     STAGE 2: TOKENIZATION   ##################################
###############################################################################


global_s2tokens = dict()
global_s2tokens[(0, 0, 0)] = dict()
global_s2tokens[(1, 0, 0)] = dict()
global_s2tokens[(1, 0, 1)] = dict()

tokens_freq = Counter()


def tokenize(s, del_brackets=True, only_org=False, use_freq=False):
    h = (int(del_brackets), int(only_org), int(use_freq))
    if s not in global_s2tokens[h]:
        s1 = simple_transform(s, del_brackets=del_brackets)
        s2 = legal_re.sub('', s1)
        s3 = countries_re.sub('', s2)
        s4 = geo_re.sub('', s3)

        arr = s4.split()

        arr_new = []
        i = 0
        while i < len(arr) - 1:
            ss = arr[i] + ' ' + arr[i + 1]
            ss1 = arr[i] + arr[i + 1]
            if ss in cities_alt:  # or ss1 in legal_tokens and len(ss1) > 2:
                i += 1
            else:
                arr_new.append(arr[i])
            i += 1

        if i == len(arr) - 1:
            arr_new.append(arr[-1])
        arr = list(arr_new)

        s5 = ' '.join(arr)

        s6 = ''
        last = None
        for ch in s5:
            if last != ch:
                last = ch
                s6 += ch

        if use_freq and len(arr) > 0:
            arr = s6.split()
            arr = sorted(arr, key=lambda x: -tokens_freq.get(x, 0))[:1]
            s6 = ' '.join(arr)

        # s1 = list(translit(s1, translitMapRuEn))[0]

        global_s2tokens[h][s] = None
        if only_org:
            global_s2tokens[h][s] = s6.split()
        else:
            for ss in [s6, s5, s4, s3, s2, s1, s]:
                res = ss.split()
                if len(res) != 0:
                    global_s2tokens[h][s] = res
                    break
        if global_s2tokens[h][s] is None:
            print(s)
            raise
    return global_s2tokens[h][s]

for e in set(train['name_1']):
    tokens_freq.update(tokenize(e)[:2])
for e in set(train['name_2']):
    tokens_freq.update(tokenize(e)[:2])


for col in ['name_1', 'name_2']:
    print(col)
    train[f'{col}_tokens'] = [tokenize(e) for e in train[col]]
    train[f'{col}_tokens_with_br'] = [tokenize(e, del_brackets=False) for e in train[col]]
    train[f'{col}_tokens_simple'] = [simple_transform(e).split() for e in train[col]]
#     train[f'{col}_tokens_org'] = [tokenize(e, del_brackets=True, only_org=True) for e in train[col]]
    train[f'{col}_tokens_freq'] = [tokenize(e, del_brackets=True, only_org=False, use_freq=True) for e in train[col]]


###############################################################################
################     STAGE 3: PREPARE TRAIN    ################################
###############################################################################


def op(a, b, agg):
    if agg == 'min':
        return min(a, b)
    if agg == 'max':
        return max(a, b)
    if agg == 'sum':
        return a + b
    raise

def calc_features(df_te,
                  clusters_list, k2ind_list, freq_list, cl2cl_neg_list,
                  # clusters_simple_list, k2ind_simple_list
                  ):
    res = df_te.copy()

    features_te = defaultdict(list)

    for i, df, df_features in [
        (0, df_te, features_te)
    ]:
        print(i)
        #        for s1, t1, t1s, t1b, t1o, t1f, s2, t2, t2s, t2b, t2o, t2f in (
        for s1, t1, t1s, t1b, t1f, s2, t2, t2s, t2b, t2f in (
            zip(df['name_1'], df['name_1_tokens'], df['name_1_tokens_simple'],
                df['name_1_tokens_with_br'], df['name_1_tokens_freq'],
                df['name_2'], df['name_2_tokens'], df['name_2_tokens_simple'],
                df['name_2_tokens_with_br'], df['name_2_tokens_freq']
                )):
            df_features['len_max'].append(max(len(s1), len(s2)))
            df_features['len_min'].append(min(len(s1), len(s2)))
            df_features['len_diff'].append(abs(len(s1) - len(s2)))
            df_features['len_diff_rel'].append(abs(len(s1) - len(s2)) / (len(s1) + len(s2)))

            df_features['len_t_max'].append(max(len(t1), len(t2)))
            df_features['len_t_min'].append(min(len(t1), len(t2)))
            df_features['len_t_diff'].append(abs(len(t1) - len(t2)))
            df_features['len_t_diff_rel'].append(abs(len(t1) - len(t2)) / (len(t1) + len(t2)))

            st1 = set(t1)
            st2 = set(t2)
            num, den = len(st1 & st2), len(st1 | st2)
            df_features['sim_tokens_num'].append(num)
            df_features['sim_tokens_den'].append(den)
            df_features['sim_tokens'].append(num / den)

            st1_exp = expand_tokens(t1) | set(t1b)
            st2_exp = expand_tokens(t2) | set(t2b)
            num1, den1 = len(st1 & st2_exp), len(st1)
            num2, den2 = len(st2 & st1_exp), len(st2)
            df_features['sim_tokens_exp_min'].append(min(num1 / den1, num2 / den2))
            df_features['sim_tokens_exp_max'].append(max(num1 / den1, num2 / den2))

            e1, e2 = ''.join(t1), ''.join(t2)
            ll = calc_lcs(e1, e2)
            try:
                df_features['lcs'].append(ll)
                df_features['lcs_norm'].append(ll / min(len(e1), len(e2)))
                df_features['lcs_norm_max'].append(ll / max(len(e1), len(e2)))
                df_features['lcs_norm_sum'].append(ll / (len(e1) + len(e2)))
            except:
                print(e1, e2, t1, t2)
                raise

            for i in range(1, 4):
                e1, e2 = ''.join(t1[:i]), ''.join(t2[:i])
                ll = calc_lcs(e1, e2)
                df_features[f'lcs{i}_norm'].append(ll / min(len(e1), len(e2)))

            e1, e2 = simple_transform(s1), simple_transform(s2)
            ll = calc_lcs(e1, e2)
            df_features['lcs_raw_norm'].append(ll / min(len(e1), len(e2)))

            #             e1, e2 = ''.join(t1o), ''.join(t2o)
            #             ll = calc_lcs(e1, e2)
            #             df_features['lcs_org_norm'].append((ll + 0.01) / (min(len(e1), len(e2)) + 0.1))

            e1, e2 = ''.join(t1f), ''.join(t2f)
            ll = calc_lcs(e1, e2)
            df_features['lcs_freq_norm'].append((ll + 0.01) / (min(len(e1), len(e2)) + 0.1))

            c_features = defaultdict(list)
            for clusters, k2ind, freq, cl2cl_neg in zip(
                #            for clusters, k2ind, freq, cl2cl_neg, clusters_simple, k2ind_simple in zip(
                clusters_list, k2ind_list, freq_list, cl2cl_neg_list  # , clusters_simple_list, k2ind_simple_list
            ):
                # c_features['is_one_cluster_simple'].append(calc_is_one_cluster(t1s, t2s, clusters_simple, k2ind_simple))

                c_features['is_one_cluster'].append(calc_is_one_cluster(t1, t2, clusters, k2ind))

                c_features['cnt_neg_cluster'].append(calc_cnt_negative(t1, t2, clusters, k2ind, cl2cl_neg))
                c_features['cluster_size_max'].append(max(get_cluster_size(t1, clusters, k2ind),
                                                          get_cluster_size(t2, clusters, k2ind)))

                for ind in range(1, 7):
                    for agg in ['sum', 'min', 'max']:
                        score_u = score_i = score_d = 0
                        if agg == 'min':
                            score_u = score_i = score_d = 1e6
                        for word in st1 & st2:
                            score_i = op(score_i, (freq[word][ind] + 0.01) / (freq[word][0] + 1), agg)
                        for word in st1 | st2:
                            score_u = op(score_u, (freq[word][ind] + 0.01) / (freq[word][0] + 1), agg)
                        for word in st1 ^ st2:
                            score_d = op(score_d, (freq[word][ind] + 0.01) / (freq[word][0] + 1), agg)

                        c_features[f'words_freq_sim_{agg}_{ind}_num'].append(score_i)
                        c_features[f'words_freq_sim_{agg}_{ind}_den'].append(score_u)
                        c_features[f'words_freq_sim_{agg}_{ind}_diff'].append(score_d)
                        c_features[f'words_freq_sim_{agg}_{ind}_rel1'].append(score_i / score_u)
                        c_features[f'words_freq_sim_{agg}_{ind}_rel2'].append(score_d / score_u)
            for k, v in c_features.items():
                if k in {'is_one_cluster', 'cluster_size_max'}:
                    df_features[k].append(max(v))
                else:
                    df_features[k].append(sorted(v)[len(clusters_list) // 2])

    features = []
    for k, v in sorted(features_te.items()):
        features.append(k)
        res[k] = v
    print(features)
    return res, features




clusters, k2ind, freq, cl2cl_neg = prepare(train)
clusters_simple, k2ind_simple, freq_simple, _ = prepare(train, use_simple=True)


train['cv'] = -np.arange(1, len(train) + 1)
ind_positive = (
    train['name_1_tokens_simple'].apply(lambda x: k2ind_simple.get(tuple(sorted(set(x))))) ==
    train['name_2_tokens_simple'].apply(lambda x: k2ind_simple.get(tuple(sorted(set(x)))))
)
train.loc[ind_positive, 'cv'] = train.loc[ind_positive, 'name_1_tokens_simple'] \
    .apply(lambda x: k2ind_simple.get(tuple(sorted(set(x)))))

print(train['cv'].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

tr_clusters = []
tr_k2ind = []
tr_freq = []
tr_cl2cl_neg_list = []
# tr_clusters_simple = []
# tr_k2ind_simple = []

df_gr = []

features = None
group_kfold = GroupKFold(n_splits=3)
N = len(train)
for train_index, test_index in group_kfold.split(train.head(N), groups=train.head(N)['cv']):
    print(len(train_index), len(test_index))
    tr = train.iloc[train_index, :].reset_index(drop=True)
    te = train.iloc[test_index, :].reset_index(drop=True)

    clusters, k2ind, freq, cl2cl_neg = prepare(tr)
    # clusters_simple, k2ind_simple, freq_simple, _ = prepare(tr, use_simple=True)

    te['foo'] = [calc_is_one_cluster(e1, e2, clusters, k2ind) for e1, e2 in
                 zip(te['name_1_tokens'], te['name_2_tokens'])]

    print(pd.crosstab(te['foo'], te['is_duplicate']))

    df_te, features = calc_features(te, [clusters], [k2ind], [freq], [cl2cl_neg])
    # [clusters_simple], [k2ind_simple])
    df_gr.append(df_te)

    print(pd.crosstab(df_te['is_one_cluster'], df_te['is_duplicate']))

    tr_clusters.append(clusters)
    tr_k2ind.append(k2ind)
    tr_freq.append(freq)
    tr_cl2cl_neg_list.append(cl2cl_neg)

    # tr_clusters_simple.append(clusters_simple)
    # tr_k2ind_simple.append(k2ind_simple)


df_tr_all = pd.concat(df_gr)
df_tr_all = df_tr_all[(df_tr_all['is_one_cluster'] == 0)].reset_index(drop=True)
print(pd.crosstab(df_tr_all['is_one_cluster'], df_tr_all['is_duplicate']))


###############################################################################
################     STAGE 4: FIT MODEL    ####################################
###############################################################################


features_final = [e for e in features if e not in {
    'is_one_cluster', 'cluster_size_max'
}]
print('number of features', len(features_final))

params = {
    "iterations": 100,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "rsm": 0.9,
    "border_count": 10,
    "max_ctr_complexity": 2,
    "random_strength": 1.0,
    "bagging_temperature": 100.0,
    "grow_policy": "SymmetricTree",
    "min_data_in_leaf": 5,
    "langevin": True,
    "diffusion_temperature": 100000,
    "auto_class_weights": 'SqrtBalanced',
    "random_seed": 777
}

cbs = dict()
for i in range(10):
    print(i)
    params['random_seed'] = i
    cb = CatBoostClassifier(**params, verbose=True, eval_metric='F1')
    cb.fit(df_tr_all[features_final], df_tr_all['is_duplicate'], metric_period=10)
    cbs[i] = cb


###############################################################################
################     STAGE 5: READ TEST SAMPLE AND PREDICT   ##################
###############################################################################

test = pd.read_csv(DATA_DIR.joinpath('test.csv'), index_col="pair_id")

k = 0
for i, s in enumerate(set(test['name_1']) | set(test['name_2'])):
    if s not in docs:
        docs[s] = nlp_en(s)
        k += 1
print(k)

new_tokens = set()
for e in docs:
    for t in simple_transform(e).split():
        if t not in all_tokens:
            new_tokens.add(t)
print(len(new_tokens))
for token in new_tokens:
    docs_tokens[token] = nlp_en(token)


geo_add = set()
kk = 0
for token in docs_tokens:
    doc = docs_tokens[token]
    for e in doc.ents:
        if e.label_ == 'GPE':
            geo_add.add(token)
            if token not in cities_alt and token not in countries:
                #print(token)
                kk += 1

print(len(geo_add), kk)
geo_re = multi_str_replace([rf"{entity}" for entity in geo_add], debug=False)


for col in ['name_1', 'name_2']:
    print(col)
    test[f'{col}_tokens'] = [tokenize(e) for e in test[col]]
    test[f'{col}_tokens_with_br'] = [tokenize(e, del_brackets=False) for e in test[col]]
    test[f'{col}_tokens_simple'] = [simple_transform(e).split() for e in test[col]]
#     test[f'{col}_tokens_org'] = [tokenize(e, del_brackets=True, only_org=True) for e in test[col]]
    test[f'{col}_tokens_freq'] = [tokenize(e, del_brackets=True, only_org=False, use_freq=True) for e in test[col]]


df_te_all_all, _ = calc_features(test, tr_clusters, tr_k2ind, tr_freq, tr_cl2cl_neg_list)


df_te_all_all['pred'] = 0
k = 0
for i in cbs:
    pred = cbs[i].predict_proba(df_te_all_all[features_final], ntree_end=80)[:, 1]
    df_te_all_all[f'pred{i}'] = pred
    df_te_all_all['pred'] += pred / len(cbs)
    k += 1

#df_te_all_all['pred_final'] =  df_te_all_all['pred']
df_te_all_all['pred_final'] = df_te_all_all[[f'pred{i}' for i in range(k)]].min(axis=1)

topn = 1600

col = 'pred_final'
df_te_all_all.loc[df_te_all_all['is_one_cluster'] == 1, col] = 1
thr_topn = df_te_all_all.sort_values(col)[col].values[::-1][topn]
df_te_all_all['is_duplicate'] = (df_te_all_all[col] > thr_topn).astype(int)

df_te_all_all[['is_duplicate']].to_csv('output/subm_final.csv')
