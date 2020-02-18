import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation as punct
from nltk.tokenize import word_tokenize

def get_TF(sents, vocab):
    N = len(sents)
    V = len(vocab)

    TF = []
    for Si in sents:
        for token in vocab:
            TF.append(Si.count(token) / len(Si))
    TF = np.asarray(TF).reshape(N, V)

    return TF

def get_IDF(sents, vocab):
    N = len(vocab)
    IDF = []
    for token in vocab:
        count = 0
        for Si in sents:
            if token in Si:
                count += 1
        IDF.append(np.log10(N / count))
    IDF = np.asarray(IDF)

    return IDF

def get_TFIDF(sents, vocab):
    TF = get_TF(sents, vocab)
    IDF = get_IDF(sents, vocab)
    TFIDF = np.asarray([tf * IDF for tf in TF])

    return TFIDF


def generate_vocab(sents):
    flat = np.concatenate(sents)
    vocab = sorted(set(flat))

    return vocab


def sentences_from_document(doc):
    T = doc.splitlines()[0].split()

    label, sents = [], []
    for i in doc.splitlines()[1:]:
        a, b = i.split(' | ')
        b = b.split()
        if len(b) < 5: continue
        label.append(a)
        sents.append(b)

    return T, label, sents


def cosine(x, y):
    v1 = np.array(x)
    v2 = np.array(y)

    if((np.sqrt(np.sum(v1 ** 2))) == 0):
        return 0
    if ((np.sqrt(np.sum(v2 ** 2))) == 0):
        return 0
    return np.dot(v1, v2) / ((np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2))))

def fitness(sys_tf, ref_tf):
    fitness = 0.0

    for i, Si in enumerate(sys_tf):
        for j, Sj in enumerate(ref_tf):
            fitness += cosine(Si, Sj)

    return fitness


def fitness_gradient(sys_tf, ref_tf):

    grad = 0.0

    for i, Si in enumerate(sys_tf):
        for j, Sj in enumerate(ref_tf):
            a = np.array(Si)
            b = np.array(Sj)
            a_mod = np.sqrt(np.sum(a ** 2))
            b_mod = np.sqrt(np.sum(b ** 2))
            if (a_mod == 0):
                grad += 0
            elif (b_mod == 0):
                grad += 0
            else:
                for iter in range(0, len(a)):
                    grad += ((b[iter] / (a_mod * b_mod)) - (
                                cosine(Si, Sj) * (a[iter] / np.sum(a ** 2))))/len(a)

    return grad


#CLEANING---------------------------------------------
def punct_except(p):
    return ''.join(set(punct)-set(p))

def remove_punc(match):
    word = match.group(0)
    neww = word[1:-1]
    for p in '-/':
        neww = neww.replace(p,' ')
    return ' '+neww+' '

def remove_dash(match):
    word = match.group(0)
    return word[1:]

def clean(raw):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    flat = []
    docs = []
    doc = []

    raw = raw.replace('\n', ' \n ').lower()

    for p in '”“':
        raw = raw.replace(p, '"')

    for p in '’‘':
        raw = raw.replace(p, "'")

    for p in '—–':
        raw = raw.replace(p, '-')

    for p in punct_except("'-_"):
        raw = raw.replace(p, ' ')

    raw = re.sub("'", '', raw)
    raw = re.sub('-\W|\W-', ' ', raw)

    raw = re.sub('[ ][ ]+', ' ', raw)
    raw = re.sub('\W\d+[/-]\d+\W', remove_punc, raw)
    raw = re.sub('[ ][ ]+', ' ', raw)

    raw = re.sub('_comma_', '.', raw)
    raw = re.sub('_per_', '/', raw)

    raw = re.sub('-[km]u\W|-nya\W|-lah\W', remove_dash, raw)
    raw = re.sub('[ ][ ]+', ' ', raw)

    while raw.endswith(('\n', ' ')):
        raw = raw[:-1]

    tempdoc = []
    for no, sent in enumerate(raw.splitlines()):

        if no == 0:
            no = 'title   | '
        else:
            no = '0'*(2-len(str(no-1)))+str(no)
            no = 'sent_'+no+' | '

        tempsent = []
        for word in word_tokenize(sent):

            if word in stop_words:
                continue

            flat.append(word)
            tempsent.append(word)

        tempdoc.append(no + ' '.join(tempsent))

    docs.append('\n'.join(tempdoc))
    doc = docs[0]

    vocab = sorted(set(flat))
    f = open('vocabulary.txt', 'w')
    f.write(' '.join(vocab))
    f.close()

    stem = open('dict_stem.txt').read()
    stem = dict([i.split(' >> ') for i in stem.splitlines()])

    V = len(vocab)
    f = open('dict_stem.txt', 'w')
    for i in vocab:
        try:
            f.write(i + ' >> ' + stem[i] + '\n')
        except:
            f.write(i + ' >> ' + stemmer.stem(i) + '\n')
    f.close()

    word_edit = open('word_edit.txt').read()
    word_edit = dict([i.split(' >> ') for i in word_edit.splitlines()])

    f = open('word_edit.txt', 'w')
    for i in vocab:
        try:
            f.write(i + ' >> ' + word_edit[i] + '\n')
        except:
            # successfully added to word_edit
            word_edit[i] = i
            f.write(i + ' >> ' + i + '\n')
    f.close()

    stem_edit = open('stem_edit.txt').read()
    stem_edit = dict([i.split(' >> ') for i in stem_edit.splitlines()])

    f = open('stem_edit.txt', 'w')
    for i in vocab:
        for token in word_edit[i].split('_'):
            if token in stop_words: continue
            try:
                f.write(token + ' >> ' + stem_edit[token] + '\n')
            except:
                # successfully added to stem_edit
                stemtok = stemmer.stem(token)
                stem_edit[token] = stemtok
                f.write(token + ' >> ' + stemtok + '\n')
    f.close()

    prop_edit = open('propernoun_edit.txt').read()
    prop_edit = dict([i.split(' >> ') for i in prop_edit.splitlines()])

    f = open('propernoun_edit.txt', 'w')
    for i in vocab:
        try:
            f.write(i + ' >> ' + prop_edit[i] + '\n')
        except:
            # successfully added to prop_edit
            prop_edit[i] = '0'
            f.write(i + ' >> 0\n')
    f.close()

    temp2 = []
    for sent in doc.splitlines():
        try:
            labs, sent = sent.split(' | ')
        except:
            continue
        temp1 = []
        for word in sent.split():
            for word in word_edit[word].split('_'):
                if word in stop_words: continue
                word = stem_edit[word]
                temp1.append(word)
        temp2.append(labs + ' | ' + ' '.join(temp1))
    data = '\n'.join(temp2)

    return data




