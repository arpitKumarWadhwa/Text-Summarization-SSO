import os, numpy as np
from collections import Counter
from utils import cosine, get_TF, get_TFIDF, sentences_from_document

f = open('propernoun_edit.txt').read()

propernoun = []
for i in f.splitlines():
    a, b = i.split(' >> ')
    if b == '1':
        propernoun.append(a)

path_inp = './1.clean/'
path_out = './2.feature/'

for file in os.listdir(path_inp):
    f = open(path_inp + file).read()

    T, label, sents = sentences_from_document(f)

    F1, F2, F3, F4, F5, F6, F7, F8 = [], [], [], [], [], [], [], []
    # F1: Feature Title
    for Si in sents:
        intersect = np.intersect1d(Si, T)
        F1.append(len(intersect) / len(T))

    # F2: Sentence Length
    S_longest = sents[np.argmax([len(j) for j in sents])]
    for Si in sents:
        F2.append(len(Si) / len(S_longest))

    # F3: Sentence Position
    for i, Si in enumerate(sents):
        i += 1
        F3.append(1 / i)

    # F4: Numerical Data
    for Si in sents:
        Si_numerical = 0
        for token in Si:
            if token.isnumeric():
                Si_numerical += 1
        F4.append(Si_numerical / len(Si))

    # F5: Thematic Words
    N = 10
    flat = np.concatenate(sents)
    freq = Counter(flat)

    thematic_words = [i for i, j in freq.most_common(N)]
    max_thematic = max([len(np.intersect1d(i, thematic_words)) for i in sents])

    for Si in sents:
        Si_thematic = np.intersect1d(Si, thematic_words)
        F5.append(len(Si_thematic) / max_thematic)

    # F6: Proper Noun
    for Si in sents:
        Si_propnouns = np.intersect1d(Si, propernoun)
        F6.append(len(Si_propnouns) / len(Si))

    # F7: Similarities Between Sentences
    vocab = sorted(set(flat))

    TF = get_TF(sents, vocab)

    sim_SiSj = []
    for i, Si in enumerate(TF):
        temp = []
        for j, Sj in enumerate(TF):
            if i == j: continue
            temp.append(cosine(Si, Sj))
        sim_SiSj.append(sum(temp))
    max_simSiSj = max(sim_SiSj)

    for sim_Si in sim_SiSj:
        F7.append(sim_Si / max_simSiSj)

    # F8: Term Weight
    TFIDF = get_TFIDF(sents, vocab)

    sum_TFIDF = []
    for tfidf in TFIDF:
        sum_TFIDF.append(sum(tfidf))
    max_sum_TFIDF = max(sum_TFIDF)

    for sum_tfidf in sum_TFIDF:
        F8.append(sum_tfidf / max_sum_TFIDF)

    feature = np.round(np.vstack((F1, F2, F3, F4, F5, F6, F7, F8)).T, 7)

    f = open(path_out + file, 'w')
    feature = []
    for x, row in enumerate(feature):
        temp = ''
        for col in row:
            space = ' ' * (15 - len(str(col)))
            temp += str(col) + space
        feature.append(label[x] + ' ' * 10 + temp)
    f.write('\n'.join(feature))
    f.close()
