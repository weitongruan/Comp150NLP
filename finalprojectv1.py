import sys
from collections import defaultdict
from math import log, exp
import nltk
from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence.
from nltk.corpus.reader import xmldocs
import re
import copy

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

def displayTag(list):
    newlist = []
    for tup in list:
        newlist.append(tup[1])
    characterPrint(newlist)

# Form a one to one dataset
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

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(set)  # maps words to tags
        self.train_set_size = 0
        self.max_dict = dict()   # maps words to most probable tags
        self.tag_counter = defaultdict(int)

    @staticmethod
    def Train(training_set, smoothing):
        """
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary
        """
        temp_tag_counter = defaultdict(int)
        temp_bigram_counter = defaultdict(int)
        temp_pair_counter = defaultdict(int)
        token_counter = 0
        temp_dictionary = defaultdict(set)  # corresponds to self.dictionary
        temp_transitions = defaultdict(float)  # self.transitions
        temp_emissions = defaultdict(float)  # self.emissions


        # Build a counter dictionary from training data
        for line in training_set:
            token_counter += len(line)
            for jdx in xrange(len(line)-1):
                temp_pair_counter[line[jdx]] += 1
                temp_tag_counter[line[jdx][1]] += 1
                temp_dictionary[line[jdx][0]].add(line[jdx][1])
                temp_bigram_counter[(line[jdx][1], line[jdx+1][1])] += 1
                temp_dictionary[end_token].add(end_token)

        # Estimate emission probabilities
        for pair in temp_pair_counter:
            temp_emissions[pair] = log(float(temp_pair_counter[pair])/temp_tag_counter[pair[1]])

        # Estimate transition probabilities
        for tup in temp_bigram_counter:
            if temp_tag_counter[tup[0]] == 0:
                print tup, tup[0], 'count is zero'
                return
            if smoothing == "no smoothing":
                temp_transitions[tup] = log(float(temp_bigram_counter[tup])/temp_tag_counter[tup[0]])
            if smoothing == "Laplace":
                temp_transitions[tup] = log(float(temp_bigram_counter[tup]+1) /
                                            (temp_tag_counter[tup[0]]+len(temp_tag_counter)))
        return temp_transitions, temp_emissions, temp_dictionary, temp_tag_counter, token_counter

    # supervised learning
    def supervisedLearning(self,training_set_with_tag, smoothing):
        self.transitions, self.emissions, self.dictionary, self.tag_counter, self.train_set_size = \
                                                            BigramHMM.Train(training_set_with_tag, smoothing)
    # Semi-supervised learning
    def semisupervisedLearning(self, training_set_with_tag, training_set_no_tag, smoothing):

        # use supervised learning on a tagged set
        self.transitions, self.emissions, self.dictionary, self.tag_counter, self.train_set_size = \
                                                            BigramHMM.Train(training_set_with_tag, smoothing)
        # unsupervised forward-backward
        for sent_no_tag in training_set_no_tag:
            self.transitions, self.emissions = BigramHMM.fwdbkwd(sent_no_tag, self.dictionary,
                                                    self.transitions, self.emissions, self.tag_counter, smoothing)


    @staticmethod
    def fwdbkwd(sent_no_tag, dictionary, transitions, emissions, tag_counter, smoothing):
        alpha_fwd = 1.0
        alpha_fwd_prev = 0.0

        while isclose(alpha_fwd_prev, alpha_fwd) != 1:
            temp_transitions = defaultdict(float)
            temp_emissions = defaultdict(float)
            temp_tag_index = defaultdict(set)  # key: tag; value: possible t; Used in M-step
            temp_pair_index = defaultdict(set)  # key: tag pair; value: possible t; Used in M-step
            temp_tagtotag = defaultdict(set)  # key: tag; value: next possible tag; Used in M-step

            temp_emissions[end_token, end_token] = 0.0

            alpha = defaultdict(dict)
            beta = defaultdict(dict)
            gamma = defaultdict(dict)
            sigma = defaultdict(dict)
            alpha_fwd_prev = alpha_fwd

            if smoothing == "Laplace":
                for t, o_t in enumerate(sent_no_tag):
                    if t == 0:
                        alpha[t][start_token] = 0.0
                    else:
                        for st in dictionary[o_t]:
                            prev_alpha_sum = 0.0
                            for pre_st in dictionary[sent_no_tag[t-1]]:
                                if (pre_st, st) in transitions:
                                    prev_alpha_sum += exp(alpha[t-1][pre_st]+transitions[(pre_st, st)])
                                else:
                                    prev_alpha_sum += exp(alpha[t-1][pre_st]+log(1.0/(tag_counter[pre_st]+len(tag_counter))))
                            alpha[t][st] = emissions[(o_t, st)]+log(prev_alpha_sum)

                alpha_fwd = alpha[len(sent_no_tag)-1][end_token]

                for idx in reversed(xrange(len(sent_no_tag))):
                    if idx == len(sent_no_tag)-1:
                        beta[idx][end_token] = 0.0
                    else:
                        for st in dictionary[sent_no_tag[idx]]:
                            # beta[idx][st] = 0.0
                            sum_temp = 0.0
                            for next_st in dictionary[sent_no_tag[idx+1]]:
                                if(st, next_st) in transitions:
                                    sum_temp += exp(transitions[(st, next_st)]+\
                                                     emissions[(sent_no_tag[idx+1], next_st)]+beta[idx+1][next_st])
                                else:
                                    sum_temp += exp(log(1.0/(tag_counter[st]+len(tag_counter)))+\
                                                     emissions[(sent_no_tag[idx + 1], next_st)] + beta[idx + 1][next_st])
                            beta[idx][st] = log(sum_temp)

                beta_bkwd = beta[0][start_token]

                assert isclose(alpha_fwd, beta_bkwd)   # make sure alpha_fwd == beta_bkwd

            ''' Expectation step
            '''
            for idx in alpha:
                for st in alpha[idx]:
                    gamma[idx][st] = alpha[idx][st]+beta[idx][st]-alpha_fwd
                    temp_tag_index[st].add(idx)
                    if idx != len(sent_no_tag)-1:
                        for next_st in dictionary[sent_no_tag[idx+1]]:
                            if (st, next_st) in transitions:
                                sigma[idx][(st, next_st)] = alpha[idx][st]+transitions[st,next_st]+\
                                                           emissions[(sent_no_tag[idx+1], next_st)]+beta[idx+1][next_st]-\
                                                            alpha_fwd
                                temp_pair_index[(st, next_st)].add(idx)
                                temp_tagtotag[st].add(next_st)

            ''' Maximization step
            '''
            for tup in transitions:
                if tup in temp_pair_index:
                    numerator = log(sum(exp(sigma[index][tup]) for index in temp_pair_index[tup]))
                    denominator = 0.0
                    for next_st in temp_tagtotag[tup[0]]:
                        denominator = log(sum(exp(sigma[index][(tup[0], next_st)]) for index in temp_pair_index[(tup[0], next_st)]))
                    temp_transitions[tup] = numerator-denominator
                else:
                    temp_transitions[tup] = transitions[tup]

            for word, tag in emissions:
                numerator = 0.0
                denominator = 0.0
                if tag in temp_tag_index:
                    numerator = sum(exp(gamma[index][tag]) for index in temp_tag_index[tag])
                    for idx in xrange(len(sent_no_tag)):
                        # if (sent_no_tag[idx] == word) & (tag in gamma[idx]):
                        #     numerator += exp(gamma[idx][tag])
                        if tag in gamma[idx]:
                            denominator += exp(gamma[idx][tag])
                    temp_emissions[word, tag] = log(numerator) - log(denominator)
                else:
                    temp_emissions[word, tag] = emissions[word, tag]

            emissions = temp_emissions
            transitions = temp_transitions

        return temp_transitions, temp_emissions

    def getMostCommon(self, training_set):
        common_dict = defaultdict(dict)

        # Learing from training set
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
            self.max_dict[element] = key_ini

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        ambiguity_counter = 0
        token_counter = 0
        for line in data_set:
            for tup in line:
                token_counter += 1
                if len(self.dictionary[tup[0]]) > 1:
                    ambiguity_counter += 1

        ambiguity = float(ambiguity_counter)/token_counter*100
        return ambiguity

    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        joint_prob = 0.0
        for idx in xrange(len(sent)-1):
            joint_prob += self.emissions[sent[idx]] + self.transitions[(sent[idx][1], sent[idx+1][1])]

        return exp(joint_prob)

    def Viterbi(self, sent, interpolation):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        sent_tagged = []
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        tag_list = [end_token]

        viterbi['0'] = 1.0

        if interpolation == "Most Common":
            # Initialization: for the first non-start token
            for state in self.dictionary[sent[1][0]]:
                if (start_token, state) in self.transitions:
                    viterbi[str(1)][state] = self.transitions[(start_token, state)] + self.emissions[(sent[1][0], state)]
                else:
                    viterbi[str(1)][state] = -float('inf')
                backpointer[str(1)][state] = start_token

            # Iteration
            for idx in xrange(2,len(sent)):
                max_check = -float("inf")
                for state in self.dictionary[sent[idx][0]]:
                    max_value = -float("inf")
                    max_loc = []
                    for pre_state in self.dictionary[sent[idx-1][0]]:
                        if (pre_state, state) in self.transitions:
                            temp = viterbi[str(idx-1)][pre_state] + self.transitions[(pre_state, state)]
                        else:
                            temp = -float('inf')
                        if temp >= max_value:
                            max_value = temp
                            max_loc = pre_state
                        viterbi[str(idx)][state] = max_value + self.emissions[(sent[idx][0], state)]
                        backpointer[str(idx)][state] = max_loc
                        if max_value > max_check:
                            max_check = max_value
                if max_check == -float("inf"):
                    max_value = -float("inf")
                    max_loc = []
                    for pre_state in viterbi[str(idx-1)]:
                        temp = viterbi[str(idx-1)][pre_state]
                        if temp >= max_value:
                            max_value = temp
                            max_loc = pre_state
                    viterbi[str(idx)][self.max_dict[sent[idx][0]]] = max_value + self.emissions[(sent[idx][0], state)]
                    backpointer[str(idx)][self.max_dict[sent[idx][0]]] = max_loc

        if interpolation == "Laplace":
            # Initialization: for the first non-start token
            for state in self.dictionary[sent[1][0]]:
                if (start_token, state) in self.transitions:
                    viterbi[str(1)][state] = self.transitions[(start_token, state)] + self.emissions[
                        (sent[1][0], state)]
                else:
                    viterbi[str(1)][state] = -float('inf')
                backpointer[str(1)][state] = start_token

            # Iteration
            for idx in xrange(2, len(sent)):
                # max_check = -float("inf")
                for state in self.dictionary[sent[idx][0]]:
                    max_value = -float("inf")
                    max_loc = []
                    for pre_state in self.dictionary[sent[idx - 1][0]]:
                        if (pre_state, state) in self.transitions:
                            temp = viterbi[str(idx - 1)][pre_state] + self.transitions[(pre_state, state)]
                        else:
                            temp = viterbi[str(idx - 1)][pre_state] + \
                                   log(1.0/(self.tag_counter[pre_state]+len(self.tag_counter)))
                        if temp >= max_value:
                            max_value = temp
                            max_loc = pre_state
                        viterbi[str(idx)][state] = max_value + self.emissions[(sent[idx][0], state)]
                        backpointer[str(idx)][state] = max_loc

        temp = end_token
        for idx in xrange(1,len(sent)):
            temp = backpointer[str(len(sent)-idx)][temp]
            tag_list.append(temp)

        for tup in sent:
            sent_tagged.append((tup[0], tag_list.pop()))

        # print sent_tagged
        return sent_tagged

    def Test(self, test_set, interpolation):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        test_retagged = []
        for sent in test_set:
            test_retagged.append(self.Viterbi(sent, interpolation))

        return test_retagged

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    common_dict = defaultdict(dict)
    max_dict = dict()
    test_tagged = []

    # Learing from training set
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

    # Tagging
    for line in test_set:
        temp_line = []
        for i in xrange(len(line)):
            if line[i][0] not in common_dict:
                print "Error!", tup[0], "not in common_dict!"
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

    # print len(token_char)
    # print len(token_pinyin)

    # token_pinyin.pop(45730)

    # print len(token_pinyin)
    # print token_pinyin[45700:45734]
    # characterPrint(token_char[45700:45734])


    dataset = organizeSentence(token_char, token_pinyin)

    print len(dataset)

    """ Supervised Learning
    """

    training_set = copy.deepcopy(dataset[0:1000])
    test_set = copy.deepcopy(dataset[1001:])

    # get vocabulary first!
    vocabulary, labelset = getVoc(training_set)
    # print len(vocabulary)
    # print len(labelset)


    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    # supervised with laplace smoothing
    bigram_hmm_laplace = BigramHMM()
    bigram_hmm_laplace.supervisedLearning(training_set_prep,"Laplace")

    # supervised with no smoothing in learning probabilities but use most common to smooth in Viterbi
    bigram_hmm_mostcommon = BigramHMM()
    bigram_hmm_mostcommon.supervisedLearning(training_set_prep, "no smoothing")
    bigram_hmm_mostcommon.getMostCommon(training_set_prep)


    """ Implement the most common class baseline. Report accuracy of the predicted tags.
        """
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    # print test_set_prep[0]
    # print test_set_predicted_baseline[0]

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm_laplace = bigram_hmm_laplace.Test(test_set_prep,"Laplace")
    test_set_predicted_bigram_hmm_mostcommon = bigram_hmm_mostcommon.Test(test_set_prep, "Most Common")
    print "--- Bigram HMM with most common accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm_mostcommon)
    print "--- Bigram HMM with Laplace accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm_laplace)
    # print test_set_predicted_bigram_hmm[0]
    # print " ".join(untag(test_set_predicted_bigram_hmm[20]))
    print "\n"    
    print "A sequence of Pinyin words:"
    print " ".join(untag(test_set_prep[25]))
    print "Common baseline results:"
    displayTag(test_set_predicted_baseline[25])
    print "HMM with most common results:"
    displayTag(test_set_predicted_bigram_hmm_mostcommon[25])
    print "HMM with laplace result:"
    displayTag(test_set_predicted_bigram_hmm_laplace[25])



    # """ Semi_supervised Learning starts here
    # """
    #
    # training_set_tagged = copy.deepcopy(dataset[0:800])
    # training_set_untagged = copy.deepcopy(dataset[801:1000])
    # test_set = copy.deepcopy(dataset[1001:])
    #
    # # get vocabulary first!
    # vocabulary, labelset = getVoc(training_set)
    #
    #
    # """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    # """
    # training_set_tagged_prep = PreprocessText(training_set_tagged, vocabulary)
    # training_set_untagged_prep = PreprocessText(training_set_untagged, vocabulary)
    # test_set_prep = PreprocessText(test_set, vocabulary)
    #
    # training_with_tag = training_set_tagged_prep
    # training_without_tag = []
    # for sent in training_set_untagged_prep:
    #     temp = []
    #     for tup in sent:
    #         temp.append(tup[0])
    #     training_without_tag.append(temp)
    #
    # bigram_hmm_semi = BigramHMM()
    # bigram_hmm_semi.semisupervisedLearning(training_with_tag, training_without_tag, "Laplace")
    # bigram_hmm_semi.getMostCommon(training_with_tag)
    #
    # """ Implement the most common class baseline. Report accuracy of the predicted tags.
    #     """
    # test_set_predicted_baseline = MostCommonClassBaseline(training_set_tagged_prep, test_set_prep)
    # print "--- Most common class baseline accuracy ---"
    # ComputeAccuracy(test_set_prep, test_set_predicted_baseline)
    #
    #
    # """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    # """
    # test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep, "Laplace")
    # print "--- Bigram HMM accuracy ---"
    # ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)
    # print test_set_predicted_bigram_hmm[0]
    # print " ".join(untag(test_set_predicted_bigram_hmm[20]))
    # print "Common baseline results:"
    # displayTag(test_set_predicted_baseline[25])
    # print "HMM result:"
    # displayTag(test_set_predicted_bigram_hmm[25])



    """ Print the first sentence of each data set.
        """
    print '\n'
    print "A sequence of Pinyin words:"
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print "It's corresponding Chinese characters:"
    displayTag(training_set_prep[0])
    print "A sequence of Pinyin words:"
    print " ".join(untag(test_set_prep[1]))
    print "It's corresponding Chinese characters:"
    displayTag(test_set_prep[1])




if __name__ == "__main__": 
    main()