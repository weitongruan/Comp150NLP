import sys
from collections import defaultdict
from math import log, exp
import nltk
from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence.
from nltk.corpus.reader import xmldocs
import re

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.
# end_unicode =

def readCorpus(root):
    corpus_character = xmldocs.XMLCorpusReader(root, 'LCMC_C_character.XML')
    corpus_pinyin = xmldocs.XMLCorpusReader(root, 'LCMC_C_pinyin.xml')
    # print corpus_pinyin.words()
    # characterPrint(corpus_character.words()[30:40])
    character = corpus_character.words()[116:]
    pinyin = corpus_pinyin.words()[130:]
    # print corpus_character.words()[116]
    # print corpus_pinyin.words()[130:]

    characterPrint(corpus_character.words()[116:284])
    #
    # print len(corpus_character.words())
    #
    # print len(corpus_pinyin.words())

    return character, pinyin

def tokenizeCharacter(corpus_character):
    list = []
    for index in xrange(len(corpus_character)):
        for word in corpus_character[index]:
            list.append(word)
    return list
    # print list
    # characterPrint(list)

def tokenizePinyin(corpus_pinyin):
    list = []
    for index in xrange(len(corpus_pinyin)):
        temp = corpus_pinyin[index]
        temp1 = re.split('\d', temp)
        if len(temp1) > 1:
            list = list + re.split('\d', temp)[0:-1]
        else:
            for word in temp:
                list.append(word)
    # print list
    return list

def characterPrint(list):
    newstr = str()
    for character in list:
        newstr = newstr + character
    print newstr

    # for character in list:
    #     print character,

def main():
    character, pinyin = readCorpus('C:\Users\wruan02\Documents\GitHub\Comp150NLP')


if __name__ == "__main__": 
    main()