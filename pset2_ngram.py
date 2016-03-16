import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Implement any helper functions here, e.g., for text preprocessing.
"""
def getVoc(dataset):
    dic = defaultdict(int)
    voc = set()
    for line in dataset:
        for word in line:
            dic[word] += 1
    for element in dic:
        if dic[element] >= 2:
            voc.add(element)
    voc.add(unknown_token)
    voc.add(start_token)
    voc.add(end_token)
    return voc

# preprocessing by adding start and end tokens and converting unknown words to unknown token
def PreprocessText(wordSet, voc):
    prepList = []
    for line in wordSet:
        for i in xrange(len(line)):           
            if line[i] not in voc:
                line[i] = unknown_token
        line.append(end_token)
        line.insert(0,start_token)
        prepList.append(line)
    return prepList

# a helper function for distribution check
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        

        
class LM:
    def __init__(self, vocabulary = set(), ngram = int()):
        self.vocabulary = vocabulary
        self.ngram = ngram
        self.counts = defaultdict(dict)
        self.prob = defaultdict(float)
        self.alpha_dic = defaultdict(set)
        self.trainset_size = int()
        self.lambdas = []

    """ Implement the functions EstimateBigrams, CheckDistribution, Perplexity and any 
    other function you might need here.
    """

    def getCounts(self, data):
        
        counter = 0
        for j in xrange(1,self.ngram+1):
            self.counts[str(j)] = defaultdict(int)
            if j == 1:
                for idx in xrange(len(data)):
                    for jdx in xrange(len(data[idx])-1):
                        self.counts[str(j)][data[idx][jdx]] += 1
                        self.alpha_dic[data[idx][jdx]] = set()
                        counter += 1
            elif j == 2:
                for idx in xrange(len(data)):
                    for jdx in xrange(len(data[idx])-j+1):
                        temp = tuple(data[idx][jdx:jdx+j])
                        self.counts[str(j)][temp] += 1
                        temp1 = temp[0]
                        self.alpha_dic[temp1].add(temp)
                        self.alpha_dic[temp] = set()
            else:
                for idx in xrange(len(data)):
                    for jdx in xrange(len(data[idx])-j+1):
                        temp = tuple(data[idx][jdx:jdx+j])
                        self.counts[str(j)][temp] += 1
                        temp1 = temp[0:j-1]
                        self.alpha_dic[temp1].add(temp)
                        self.alpha_dic[temp] = set()
        self.trainset_size = counter
    def EstimateProb(self, smoothing, *argv):
        self.prob.clear()
        if smoothing == 'no smoothing':
            for j in xrange(1,self.ngram+1):
                if j == 1:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup])/self.trainset_size
                elif j == 2:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup])/self.counts[str(j-1)][tup[0]]
                else:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup])/self.counts[str(j-1)][tup[0:j-1]]
            self.prob['other'] = 0.0

        elif smoothing == 'Laplace interpolation':
            for j in xrange(1,self.ngram+1):
                if j == 1:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup])/self.trainset_size
                elif j == 2:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup]+1) \
                                                        / (self.counts[str(j-1)][tup[0]]+len(self.vocabulary))
                else:
                    for tup in self.counts[str(j)]:
                        self.prob[tup] = float(self.counts[str(j)][tup]+1) \
                                                        / (self.counts[str(j-1)][tup[0:j-1]]+len(self.vocabulary))

        elif smoothing == 'linear interpolation':
            
            j = self.ngram
            weights = 1.0/j
            for tup in self.counts[str(j)]:
                self.prob[tup] = 0.0
                for i in xrange(1,j+1):
                    if i == 1:
                        temp = tup[-1]
                        self.prob[tup] += weights*float(self.counts[str(i)][temp])/(self.trainset_size)
                        self.prob[temp] = float(self.counts[str(i)][temp])/(self.trainset_size)
                    elif i == 2:
                        temp = tup[self.ngram-i:self.ngram]
                        self.prob[tup] +=  weights*float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0]]
                        if i != j:  
                            self.prob[temp] = float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0]]
                    else:
                        temp = tup[self.ngram-i:self.ngram]
                        self.prob[tup] += weights*float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0:i-1]]
                        if i !=j:
                            self.prob[temp] = float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0:i-1]]

            del self.prob[end_token]            
        
            
            
        elif smoothing == 'deleted interpolation':
            if len(argv) == 0:
                print "held out corpus needed!"
            else:
                held_out = argv[0]
                held_out_counts = defaultdict(dict)

                # Get counts for the held_out set
                held_out_size = self.trainset_size #!!!!!!
                for j in xrange(1,self.ngram+1):
                    held_out_counts[str(j)] = defaultdict(int)
                    if j == 1:
                        for idx in xrange(len(held_out)):
                            for jdx in xrange(len(held_out[idx])-1):
                                held_out_counts[str(j)][held_out[idx][jdx]] += 1
                    else:
                        for idx in xrange(len(held_out)):
                            for jdx in xrange(len(held_out[idx])-j+1):
                                temp = tuple(held_out[idx][jdx:jdx+j])
                                held_out_counts[str(j)][temp] += 1

                # Calculate lambdas
                for i in xrange(self.ngram):        # initialize
                    self.lambdas.append(0)

                

                for tup in held_out_counts[str(self.ngram)]:
                    maxvalue = 0.0
                    maxtag = 0
                    for i in range(1,self.ngram+1):
                        if i == 1:
                            temp = tup[-i]
                            tempvalue = float(held_out_counts[str(i)][temp] - 1)/(held_out_size -1)
                        elif i == 2:
                            temp = tup[self.ngram-i:self.ngram]
                            if held_out_counts[str(i-1)][temp[0]] == 1:
                                tempvalue = 0.0
                            else:
                                tempvalue = float(held_out_counts[str(i)][temp] - 1) \
                                                                   / (held_out_counts[str(i-1)][temp[0]] -1)
                        else:
                            temp = tup[self.ngram-i:self.ngram]
                            if held_out_counts[str(i-1)][temp[0:i-1]] == 1:
                                tempvalue = 0.0
                            else:
                                tempvalue = float(held_out_counts[str(i)][temp] - 1) \
                                                / (held_out_counts[str(i-1)][temp[0:i-1]] -1)
                        if tempvalue >=  maxvalue:
                            maxtag = i
                            maxvalue = tempvalue
                    self.lambdas[maxtag-1] += held_out_counts[str(self.ngram)][tup]
                # Normalize lambdas
                lambda_sum = sum(self.lambdas)
                for i in range(self.ngram):
                    self.lambdas[i] = float(self.lambdas[i])/lambda_sum
                

                # Calculate weighted probability

                j = self.ngram
                for tup in self.counts[str(j)]:
                    self.prob[tup] = 0.0
                    for i in xrange(1,j+1):
                        if i == 1:
                            temp = tup[-1]
                            self.prob[tup] += self.lambdas[i-1]*float(self.counts[str(i)][temp])/(self.trainset_size)
                            self.prob[temp] = float(self.counts[str(i)][temp])/(self.trainset_size)
                        elif i == 2:
                            temp = tup[self.ngram-i:self.ngram]
                            self.prob[tup] +=  self.lambdas[i-1]*float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0]]
                            if i != j:  
                                self.prob[temp] = float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0]]
                        else:
                            temp = tup[self.ngram-i:self.ngram]
                            self.prob[tup] += self.lambdas[i-1]*float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0:i-1]]
                            if i !=j:
                                self.prob[temp] = float(self.counts[str(i)][temp])/self.counts[str(i-1)][temp[0:i-1]]

                del self.prob[end_token]

    def CheckDistribution(self):
        isvalid = 1
        for a in self.vocabulary:
            sum_prob = 0.0
            if len(self.alpha_dic[a]) != 0:
                for bg in self.alpha_dic[a]:
                    if bg in self.prob:
                        sum_prob += self.prob[bg]
                    else:
                        print bg, 'not in keys!\n'
                        return
#            if isclose(sum_prob,1.0) == 0:
#                print a
#                print self.alpha_dic[a]
#                print len(self.alpha_dic[a])
#                return
                isvalid = isvalid & isclose(sum_prob,1.0)
        for j in xrange(2,self.ngram):
            for tup in self.counts[str(j)]:
                sum_prob = 0.0
                if len(self.alpha_dic[tup]) != 0:
                    for tup1 in self.alpha_dic[tup]:
                        if tup1 in self.prob:
                            sum_prob += self.prob[tup1]
                        else:
                            print tup1, 'not in keys!\n'
                            return
#                    if isclose(sum_prob,1.0) == 0:
#                       print tup
#                       print self.alpha_dic[tup]
#                       print len(self.alpha_dic[tup])
#                       return
                    isvalid = isvalid & isclose(sum_prob,1.0)
        if isvalid == 1:
            print 'Distribution is valid!'
        else:
            print 'Distribution is not valid! Error!'

    def Perplexity(self, data, smoothing):
        perp = 0.0
        counter = 0
        if smoothing == 'no smoothing':
            for idx in xrange(len(data)):
                counter += len(data[idx]) -1
                for jdx in xrange(len(data[idx])-self.ngram+1):
                    temp = tuple(data[idx][jdx:jdx+self.ngram])
                    if temp in self.counts[str(self.ngram)]:
                        perp += log(self.prob[temp])
                    else:
                        perp = float('inf')
                        return perp

        elif smoothing == 'Laplace interpolation':
            for idx in xrange(len(data)):
                counter += len(data[idx]) -1
                for jdx in xrange(len(data[idx])-self.ngram+1):
                    temp = tuple(data[idx][jdx:jdx+self.ngram])
                    if temp in self.counts[str(self.ngram)]:
                        perp += log(self.prob[temp])
                    else:
                        if self.ngram == 2:
                            perp += log(1.0/(self.counts[str(self.ngram-1)][temp[0]]+len(self.vocabulary)))
                        else:
                            perp += log(1.0/(self.counts[str(self.ngram-1)][temp[0:self.ngram-1]]+len(self.vocabulary)))


        elif (smoothing == 'linear interpolation') | (smoothing == 'deleted interpolation'):
            # Initialize weights list
            weights = []
            if smoothing == 'linear interpolation':
                for i in xrange(self.ngram):
                    weights.append(1.0/self.ngram)
            else:
                weights = self.lambdas 
            # Loop through test set
            for idx in xrange(len(data)):
                counter += len(data[idx]) -1
                for jdx in xrange(len(data[idx])-self.ngram+1):
                    temp = tuple(data[idx][jdx:jdx+self.ngram])
                    if temp in self.counts[str(self.ngram)]:
                        perp += log(self.prob[temp])
                    else:
                        for j in range(1,self.ngram):
                            if j == 1:
                                tup = temp[-1]
                                if tup in self.prob:
                                    if self.prob[tup] == 0:
                                        print tup
                                        return
                                    perp += log(weights[j-1]*self.prob[tup])
                            else:
                                tup = temp[self.ngram-j:self.ngram]
                                if tup in self.prob:
                                    perp += log(weights[j-1]*self.prob[tup])
#                                else:
#                                    print tup, "not in self.prob!"
#                                    return
                    
        perp = exp(-(perp/counter)) 
        return perp

    


        

def main():
    #training_set = brown.sents()[:50000]
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]
    
    vocabulary = getVoc(training_set)    

    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    held_out_set_prep = PreprocessText(held_out_set, vocabulary)
    test_set_prep= PreprocessText(test_set, vocabulary)

    """ Print the first sentence of each data set.
    """
    print training_set_prep[0]
    print held_out_set_prep[0]
    print test_set_prep[0]

    """ Extra Credit: Extend the model to Ngrams
    """
    lm1 = LM(vocabulary, 3)
    lm1.getCounts(training_set_prep)
    lm1.EstimateProb('no smoothing')
    print " Distribution check result:  ", lm1.CheckDistribution()
    print ' Perplexity without smoothing:  ', lm1.Perplexity(test_set_prep,'no smoothing')
    lm1.EstimateProb('Laplace interpolation')
    print ' Perplexity with Laplace smoothing:  ', lm1.Perplexity(test_set_prep,'Laplace interpolation')
    lm1.EstimateProb('linear interpolation')
    print ' Perplexity with linear interpolation:  ', lm1.Perplexity(test_set_prep,'linear interpolation')

    lm1.EstimateProb('deleted interpolation',held_out_set_prep)
    print ' Lambdas are: ', lm1.lambdas
    print ' Perplexity with deleted interpolation:  ', lm1.Perplexity(test_set_prep,'deleted interpolation')

if __name__ == "__main__": 
    main()







    