import sys
import json
import nltk
# import scikit
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def translate_non_alphanumerics(to_translate, translate_to=u''):
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)


def getTokens(sentence):
    lowers = sentence.lower()
    lowers = translate_non_alphanumerics(lowers)
    # print lowers
    # no_punctuation = lowers.translate(string.punctuation)

    tokens = nltk.word_tokenize(lowers)

    return tokens


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

with open(sys.argv[1]) as rd:
  data = json.load(rd)

stemmer = PorterStemmer()
train, test = [], []
for i in range(len(data)):
  tokenized = getTokens(data[i]['text'])
  filtered = [w for w in tokenized if w not in stopwords.words('english')]
  stemmed = stem_tokens(filtered, stemmer)
  stemmed = ' '.join(stemmed)
  data[i]['text'] = stemmed

  if i < 70000:
    train.append(data[i])
  else:
    test.append(data[i])

with open('train.json', 'w') as fout:
    json.dump(train, fout)

with open('test.json', 'w') as fout:
    json.dump(test, fout)
