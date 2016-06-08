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
end_unicode = [u'\u3002', u'\uFF01', u'\uFF1F']

def readCorpus(root):
    corpus_character = xmldocs.XMLCorpusReader(root, 'LCMC_C_character.XML')
    corpus_pinyin = xmldocs.XMLCorpusReader(root, 'LCMC_C_pinyin.xml')
    character = corpus_character.words()[116:]
    pinyin = corpus_pinyin.words()[130:]
    # characterPrint(corpus_character.words()[116:284])

    # corpus_character = xmldocs.XMLCorpusReader(root, 'LCMC_J_character.XML')
    # corpus_pinyin = xmldocs.XMLCorpusReader(root, 'LCMC_J_pinyin.xml')
    # character = corpus_character.words()[116:]
    # pinyin = corpus_pinyin.words()[130:]
    # # characterPrint(corpus_character.words()[116:140])

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
    newlist = []
    # temp = corpus_pinyin[0:30]
    # print temp
    for index in xrange(len(corpus_pinyin)):
        temp = corpus_pinyin[index]
        temp1 = re.split('\d', temp)
        if len(temp1) > 1:
            newlist = newlist + re.split('\d', temp)[0:-1]
        else:
            for word in temp:
                newlist.append(word)
    # print newlist
    return newlist

def characterPrint(list):
    newstr = str()
    for character in list:
        newstr = newstr + character
    print newstr


def organizeSentence(list_char, list_pinyin):
    listOfSent = []
    tempSent = []
    if len(list_char) == len(list_pinyin):
        for index in xrange(len(list_char)):
            tempSent.append((list_pinyin[index], list_char[index]))
            if list_pinyin[index] in end_unicode:
                listOfSent.append(tempSent)
                tempSent = []
    else:
        print "Charster list and pinyin list doesn't match"
    return listOfSent

# Find vocabulary from training set
def getVoc(dataset):
    dic = defaultdict(int)
    voc = set()
    labels = set()
    for line in dataset:
        for tup in line:
            labels.add(tup[1])
            if tup[0] in dic:
                voc.add(tup[0])
            dic[tup[0]] += 1
    voc.add(start_token)
    voc.add(end_token)
    voc.add(unknown_token)
    labels.add(start_token)
    labels.add(end_token)
    return voc, labels

def PreprocessText(dataset, voc):
    prepList = []
    for line in dataset:
        for i in xrange(len(line)):
            if line[i][0] not in voc:
                tup = line[i]
                line[i] = unknown_token, tup[1]
        line.append((end_token, end_token))
        line.insert(0, (start_token, start_token))
        prepList.append(line)
    return prepList

class BigramHMM:
    def __init__(self, voc, labels):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        # self.transitions = defaultdict(float)
        # self.emissions = defaultdict(float)
        self.dictionary = defaultdict(set)  # maps words to tags
        self.train_set_size = 0
        self.vocabulary = voc  # a set of all possible pinyin
        self.labelset = labels  # a set of all possible labels(characters)
        self.transitions = defaultdict(dict)
        self.emissions = defaultdict(dict)


    def Train(self, training_set):
        """
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary
        """
        temp_tag_counter = defaultdict(int)     # label counter
        temp_bigram_counter = defaultdict(int)  # transition couner
        temp_pair_counter = defaultdict(int)    # emission counter
        token_counter = 0

        # Build a counter dictionary from training data
        for line in training_set:
            token_counter += len(line)
            for jdx in xrange(len(line)-1):
                temp_pair_counter[line[jdx]] += 1
                temp_tag_counter[line[jdx][1]] += 1
                self.dictionary[line[jdx][0]].add(line[jdx][1])
                temp_bigram_counter[(line[jdx][1], line[jdx+1][1])] += 1

            self.dictionary[end_token].add(end_token)
        self.train_set_size = token_counter

        # Estimate emission probabilities
        for word in self.vocabulary:
            for tag in self.labelset:
                if (word, tag) in temp_pair_counter:
                    self.emissions[word][tag] = log(float((temp_pair_counter[(word, tag)]+1)) /
                                                    (temp_tag_counter[tag]+len(self.labelset)))
                else:
                    self.emissions[word][tag] = log(1.0 / len(self.vocabulary))

        # Estimate transition probabilities
        for tag1 in self.labelset:
            for tag2 in self.labelset:
                if (tag1, tag2) in temp_bigram_counter:
                    self.transitions[tag1][tag2] = log(float((temp_bigram_counter[(tag1, tag2)]+1)) /
                                                       (temp_tag_counter[tag1]+len(self.labelset)))
                else:
                    self.transitions[tag1][tag2] = log(1.0 / len(self.labelset))

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        ambiguity_counter = 0
        token_counter = 0
        for line in data_set:
            for tup in line:
                token_counter += 1
                if len(self.dictionary[tup[0]]) > 1:
                    ambiguity_counter += 1

        ambiguity = float(ambiguity_counter)/(token_counter-2*len(data_set))*100
        return ambiguity

    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        joint_prob = 0.0
        for idx in xrange(len(sent)-1):
            joint_prob += self.emissions[sent[idx][0]][sent[idx][1]] + self.transitions[sent[idx][1]][sent[idx+1][1]]

        return exp(joint_prob)

    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        sent_tagged = []
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        tag_list = [end_token]

        viterbi['0'] = 1.0

        # Initialization: for the first non-start token
        for state in self.labelset:
            viterbi[str(1)][state] = self.transitions[start_token][state] + self.emissions[sent[1][0]][state]
            backpointer[str(1)][state] = start_token

        # Iteration
        for idx in xrange(2,len(sent)):
            for state in self.labelset:
                max_value = -float('inf')
                max_loc = []
                for pre_state in self.labelset:
                    temp = viterbi[str(idx-1)][pre_state] + self.transitions[pre_state][state]
                    if temp >= max_value:
                        max_value = temp
                        max_loc = pre_state
                viterbi[str(idx)][state] = max_value + self.emissions[sent[idx][0]][state]
                backpointer[str(idx)][state] = max_loc

        temp = end_token
        for idx in xrange(1,len(sent)):
            temp = backpointer[str(len(sent)-idx)][temp]
            tag_list.append(temp)

        for tup in sent:
            sent_tagged.append((tup[0], tag_list.pop()))

        # print sent_tagged
        return sent_tagged

    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        test_retagged = []
        for sent in test_set:
            test_retagged.append(self.Viterbi(sent))

        return test_retagged

def MostCommonClassBaseline(training_set, test_set, vocabulary, labelset):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    common_dict = defaultdict(dict)
    max_dict = dict()
    test_tagged = []

    # Learning from training set
    # Build a dictionary of counter
    for line in training_set:
        for tup in line:
            if tup[0] not in common_dict:
                common_dict[tup[0]] = defaultdict(int)
            common_dict[tup[0]][tup[1]] += 1

    # Build a dictionary of most common class
    for element in common_dict:
        value_ini = 0
        key_ini = str()
        for keys in common_dict[element]:
            if common_dict[element][keys] > value_ini:
                value_ini = common_dict[element][keys]
                key_ini = keys
        max_dict[element] = key_ini

    # Extra step: for a word never seen in training set, randomly pick a label
    for word in vocabulary:
        if word not in max_dict:
            max_dict[word] = labelset.pop()

    # Tagging
    for line in test_set:
        temp_line = []
        for i in xrange(len(line)):
            if line[i][0] not in max_dict:
                print "Error!", line[i][0], "not in max_dict!"
            else:
                temp_line.append((line[i][0], max_dict[line[i][0]]))
        test_tagged.append(temp_line)

    return test_tagged

def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
    try:
        len(test_set) == len(test_set_predicted)
    except ValueError:
        print "The size of test set and tagged test set are not the same!"

    total_counter = 0  # counter for total number of tokens
    mistake_counter = 0  # counter for incorrectly tagged tokens
    lineMistake_counter = 0  # counter for imperfect tagged lines
    for idx in xrange(len(test_set)):
        if test_set[idx] != test_set_predicted[idx]:  # If this line is perfectly tagged, no mistake for each token
            lineMistake_counter += 1
            for jdx in xrange(len(test_set[idx])):
                total_counter += 1
                if test_set[idx][jdx] != test_set_predicted[idx][jdx]:
                    mistake_counter += 1
        else:
            total_counter += len(test_set[idx])

    # minus is used for excluding sentence boundary tokens
    accuracy_tagging = 1 - float(mistake_counter)/(total_counter - 2*len(test_set))
    accuracy_sentence = 1 - float(lineMistake_counter)/len(test_set)

    print "The sentence accuracy is: ", accuracy_sentence
    print "The tagging accuracy is: ", accuracy_tagging


def main():
    character, pinyin = readCorpus('C:\Users\wruan02\Documents\GitHub\Comp150NLP')
    token_char = tokenizeCharacter(character[0:])
    token_pinyin = tokenizePinyin(pinyin[0:])
    dataset = organizeSentence(token_char, token_pinyin)

    training_set = dataset[0:500]
    test_set = dataset[501:]

    # get vocabulary first!
    vocabulary, labelset = getVoc(training_set_prep)
    print len(vocabulary)
    print len(labelset)

    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    """ Print the first sentence of each data set.
        """
    # print training_set_prep[0]
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))
    print test_set_prep[0]

    bigram_hmm = BigramHMM(vocabulary, labelset)
    bigram_hmm.Train(training_set_prep)

    """ Implement the most common class baseline. Report accuracy of the predicted tags.
        """
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep, vocabulary, labelset)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    print test_set_prep[0]
    print test_set_predicted_baseline[0]

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)

if __name__ == "__main__": 
    main()