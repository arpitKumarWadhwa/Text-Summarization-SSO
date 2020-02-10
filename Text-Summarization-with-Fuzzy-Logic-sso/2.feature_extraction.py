import re, os, numpy as np
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
    # F1: Fitur Title : Feature Title
    for Si in sents:
        irisan = np.intersect1d(Si, T)
        F1.append(len(irisan) / len(T))

    # F2: Panjang Kalimat : Sentence Length
    S_longest = sents[np.argmax([len(j) for j in sents])]
    for Si in sents:
        F2.append(len(Si) / len(S_longest))

    # F3: Posisi Kalimat : Sentence Position
    for i, Si in enumerate(sents):
        i += 1
        F3.append(1 / i)

    # F4: Data Numerik
    for Si in sents:
        Si_numerik = 0
        for token in Si:
            if token.isnumeric():
                Si_numerik += 1
        F4.append(Si_numerik / len(Si))

    # F5: Kata Tematik : Thematic Words
    N = 10
    flat = np.concatenate(sents)
    freq = Counter(flat)

    kata_tematik = [i for i, j in freq.most_common(N)]
    maks_tematik = max([len(np.intersect1d(i, kata_tematik)) for i in sents])

    for Si in sents:
        Si_tematik = np.intersect1d(Si, kata_tematik)
        F5.append(len(Si_tematik) / maks_tematik)

    # F6: ProperNoun : Proper Noun
    for Si in sents:
        Si_propnouns = np.intersect1d(Si, propernoun)
        F6.append(len(Si_propnouns) / len(Si))

    # F7: Kemiripan Antar Kalimat : Similarities Between Sentences
    vocab = sorted(set(flat))

    TF = get_TF(sents, vocab)

    sim_SiSj = []
    for i, Si in enumerate(TF):
        temp = []
        for j, Sj in enumerate(TF):
            if i == j: continue
            temp.append(cosine(Si, Sj))
        sim_SiSj.append(sum(temp))
    maks_simSiSj = max(sim_SiSj)

    for sim_Si in sim_SiSj:
        F7.append(sim_Si / maks_simSiSj)

    # F8: Bobot Term : Term Weight
    TFIDF = get_TFIDF(sents, vocab)

    sum_TFIDF = []
    for tfidf in TFIDF:
        sum_TFIDF.append(sum(tfidf))
    maks_sum_TFIDF = max(sum_TFIDF)

    for sum_tfidf in sum_TFIDF:
        F8.append(sum_tfidf / maks_sum_TFIDF)

    feature = np.round(np.vstack((F1, F2, F3, F4, F5, F6, F7, F8)).T, 7)

    f = open(path_out + file, 'w')
    fitur = []
    for x, row in enumerate(feature):
        temp = ''
        for col in row:
            space = ' ' * (15 - len(str(col)))
            temp += str(col) + space
        fitur.append(label[x] + ' ' * 10 + temp)
    f.write('\n'.join(fitur))
    f.close()
