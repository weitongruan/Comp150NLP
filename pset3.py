import sys
from collections import defaultdict
from math import log, exp
import nltk
from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

# Find vocabulary from training set
def getVoc(dataset):
    dic = defaultdict(int)
    voc = set()
    for line in dataset:
        for tup in line:
            if tup[0] in dic:
                voc.add(tup[0])
            dic[tup[0]] += 1
    voc.add(start_token)
    voc.add(end_token)
    voc.add(unknown_token)
    return voc

def PreprocessText(dataset, voc):
    prepList = []
    for line in dataset:
        for i in xrange(len(line)):
            if line[i][0] not in voc:
                tup = line[i]
                line[i] = unknown_token, tup[1]
        line.append((end_token, end_token))
        line.insert(0,(start_token, start_token))
        prepList.append(line)
    return prepList
        
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
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
        temp_tag_counter = defaultdict(int)
        temp_bigram_counter = defaultdict(int)
        temp_pair_counter = defaultdict(int)
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
        for pair in temp_pair_counter:
            self.emissions[pair] = log(float(temp_pair_counter[pair])/temp_tag_counter[pair[1]])

        # Estimate transition probabilities
        for tup in temp_bigram_counter:
            if temp_tag_counter[tup[0]] == 0:
                print tup, tup[0], 'count is zero'
                return
            self.transitions[tup] = log(float(temp_bigram_counter[tup])/temp_tag_counter[tup[0]])

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

    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        sent_tagged = []
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        tag_list = [end_token]

        viterbi['0'] = 1.0

        # Initialization: for the first non-start token
        for state in self.dictionary[sent[1][0]]:
            if (start_token, state) in self.transitions:
                viterbi[str(1)][state] = self.transitions[(start_token, state)] + self.emissions[(sent[1][0], state)]
            else:
                viterbi[str(1)][state] = -float('inf')
            backpointer[str(1)][state] = start_token

        # Iteration
        for idx in xrange(2,len(sent)):
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

def MostCommonClassBaseline(training_set, test_set):
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

def buildConfusionMatrix(test_set, test_set_predicted):
    confusion = defaultdict(lambda: defaultdict(int))  # confusion matrix
    try:
        len(test_set) == len(test_set_predicted)
    except ValueError:
        print "The size of test set and tagged test set are not the same!"

    for idx in xrange(len(test_set)):
        for jdx in xrange(1, len(test_set[idx])-1):
            if test_set[idx][jdx][1] != test_set_predicted[idx][jdx][1]:
                confusion[test_set[idx][jdx][1]][test_set_predicted[idx][jdx][1]] += 1

    return confusion

def matrixmax(dict_matrix):
    key_max = []
    key_shouldbe = []
    ercount_max = 0
    for key in dict_matrix:
        temp = max(dict_matrix[key], key=dict_matrix[key].get)
        temp_count = max(dict_matrix[key].values())
        try:
            dict_matrix[key][temp] == temp_count
        except ValueError:
            print "find max error in matrixmax!!", "key: ", temp, "count: ", temp_count
        if temp_count >= ercount_max:
            ercount_max = temp_count
            key_max = temp
            key_shouldbe = key

    return key_max, key_shouldbe, ercount_max

    
def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens. 
    training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[3000:]

    # get vocabulary first!
    vocabulary = getVoc(training_set)

    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    # print training_set_prep[0]
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)
    print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])

    """ Implement the most common class baseline. Report accuracy of the predicted tags.
    """
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)

    # Build confusion matrix
    confusionM = buildConfusionMatrix(test_set_prep, test_set_predicted_bigram_hmm)
    maxer_tag, maxer_tag1, max_error = matrixmax(confusionM)

    print "The most confused tag pair: ", maxer_tag1, "(should be)", maxer_tag, "(estimated)"

    # print nltk.help.upenn_tagset("JJ")
    # print nltk.help.upenn_tagset("NN")

if __name__ == "__main__": 
    main()