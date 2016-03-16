import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Implement any helper functions here, e.g., for text preprocessing.
"""

# Find vocabulary from training set
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
    longlist = []
    for line in wordSet:
        longlist.append(start_token)
        for i in xrange(len(line)):
            if line[i] not in voc:
                line[i] = unknown_token
            longlist.append(line[i])
        longlist.append(end_token)
        line.append(end_token)
        line.insert(0,start_token)
        prepList.append(line)
    return prepList, longlist


# a helper function for distribution check
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        


class BigramLM:
    def __init__(self, vocabulary = set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.alpha_dic = defaultdict(set) # for implementing CheckDistribution
        self.bigram_prob = defaultdict(float)
        self.trainset_size = int()
#        self.trainset_size = trainset_size + 50000
        self.lambda1 = float()
        self.lambda2 = float()
    """ Implement the functions EstimateBigrams, CheckDistribution, Perplexity and any 
    other function you might need here.
    """
    
    # Get both unigram_counts and bigram_counts. Also, create alpla_dic that contains all known 
    # bigrams from known unigrams
    def getCounts(self, data):
        self.trainset_size = len(data) - 50000
        for i in xrange(len(data)-1):
            self.unigram_counts[data[i]] += 1
            self.bigram_counts[(data[i],data[i+1])] += 1
            self.alpha_dic[data[i]].add((data[i],data[i+1]))
            
        self.unigram_counts[data[-1]] += 1
        self.unigram_counts[end_token] = 1
        del self.bigram_counts[end_token,start_token]
        del self.alpha_dic[end_token]
#        self.bigram_counts[(end_token,start_token)] = 1
#        self.unigram_counts[start_token] = 1
#        del self.unigram_counts[end_token]
    
    # Calculate probability for bigrams (only for known bigrams)  
    # *argv is used for held_out_set in deleted interpolation         
    def EstimateBigrams(self, smoothing, *argv):
        self.bigram_prob.clear()
        if smoothing == 'no smoothing':
            for (a,b) in self.bigram_counts:
                self.bigram_prob[(a,b)] = float(self.bigram_counts[(a,b)])/self.unigram_counts[a]
            self.bigram_prob['other'] = 0

        elif smoothing == 'Laplace interpolation':
            for (a,b) in self.bigram_counts:
                self.bigram_prob[(a,b)] = (float(self.bigram_counts[(a,b)])+1) \
                                                            / (self.unigram_counts[a]+len(self.vocabulary))                                            

        # for both linear and deleted interpolation, calculate probability for every known bigram, when an unkown
        # bigram occurs in Perplexity, calculate it on the fly.
        elif smoothing == 'linear interpolation':
            for (a,b) in self.bigram_counts:
                self.bigram_prob[(a,b)] = 0.5*(float(self.bigram_counts[(a,b)])/self.unigram_counts[a] \
                                                            + float(self.unigram_counts[b])/(self.trainset_size))
                
        elif smoothing == 'deleted interpolation':
            if len(argv) == 0:
                print "held out corpus needed!"
            else:
                # Get counts for held_out_set
                held_out = argv[0]
                held_out_unigram_counts = defaultdict(int)
                held_out_bigram_counts = defaultdict(int)
                for i in range(len(held_out)-1):
                    held_out_unigram_counts[held_out[i]] += 1
                    held_out_bigram_counts[(held_out[i],held_out[i+1])] += 1
                held_out_unigram_counts[held_out[-1]] += 1
                # Calculate lambdas
                lambda1 = 0.0
                lambda2 = 0.0
                for (a,b) in held_out_bigram_counts:
                    if held_out_unigram_counts[a] == 1:
                        temp2 = 0
                    else:  
                        temp2 = float((held_out_bigram_counts[(a,b)] -1))/(held_out_unigram_counts[a]-1) 
                    temp1 = float((held_out_unigram_counts[b] -1))/(len(held_out) -1)
                    if temp1 > temp2:
                        lambda1 += held_out_bigram_counts[(a,b)]
                    else:
                        lambda2 += held_out_bigram_counts[(a,b)]
                temp = lambda1 + lambda2
                self.lambda1 = float(lambda1)/temp
                self.lambda2 = float(lambda2)/temp
                # Calculate probability for known bigrams
                for (a,b) in self.bigram_counts:
                    self.bigram_prob[(a,b)] = self.lambda2*float(self.bigram_counts[(a,b)])/self.unigram_counts[a] \
                                                + self.lambda1*float(self.unigram_counts[b])/ self.trainset_size

    # Check if the bigram distribution is valid
    def CheckDistribution(self):
        isvalid = 1
        for a in self.vocabulary:
            if a != end_token: 
                sum_prob = 0.0
                for bg in self.alpha_dic[a]:
                    if bg in self.bigram_prob:
                        sum_prob += self.bigram_prob[bg]
                    else:
                        print bg, 'not in bigram\n'
                        return
            isvalid = isvalid & isclose(sum_prob,1.0)
        if isvalid == 1:
            print 'Distribution is valid !' 
        else:
            print 'Distribution is not valid! Error!'
        
    # Perplexity calculation    
    def Perplexity(self, data, smoothing):
        perp = 0.0
        for i in range(len(data)-1):
            if smoothing == 'no smoothing':
                if (data[i],data[i+1])!=(end_token,start_token):
                    if ((data[i],data[i+1]) in self.bigram_prob):
                        perp += log(self.bigram_prob[(data[i],data[i+1])])
                    else:
                        perp = float('inf')
                        return perp
                        
            elif smoothing == 'Laplace interpolation':
                if (data[i],data[i+1])!=(end_token,start_token):
                    if (data[i],data[i+1]) in self.bigram_prob:
                        perp += log(self.bigram_prob[(data[i],data[i+1])])
                    else:
                        perp += log(1.0/(self.unigram_counts[data[i]]+len(self.vocabulary)))
                
            elif smoothing == 'linear interpolation':
                if (data[i],data[i+1])!=(end_token,start_token):
                    if (data[i],data[i+1]) in self.bigram_prob:
                        perp += log(self.bigram_prob[(data[i],data[i+1])])
                    else:
                        if data[i+1] in self.unigram_counts:
                            perp += log(0.5*float(self.unigram_counts[data[i+1]])/(self.trainset_size))
                        else:
                            print "perplexity: word not in dictionary! linear interpolation!"
                            return
            elif smoothing == 'deleted interpolation':
                if (data[i],data[i+1])!=(end_token,start_token):
                    if (data[i],data[i+1]) in self.bigram_prob:
                        perp += log(self.bigram_prob[(data[i],data[i+1])])
                    else:
                        if data[i+1] in self.unigram_counts:
                            perp += log(self.lambda1*float(self.unigram_counts[data[i+1]])/self.trainset_size)
                        else:
                            print "perplexity: word not in dictionary! deleted interpolation!"
                            return    
        perp = exp(-(perp/(len(data)-3000)))
        return perp

def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]
    
    vocabulary= getVoc(training_set)
    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    training_set_prep, train_list = PreprocessText(training_set, vocabulary)
    held_out_set_prep,held_out_list = PreprocessText(held_out_set, vocabulary)
    test_set_prep, test_list = PreprocessText(test_set, vocabulary)

    """ Print the first sentence of each data set.
    """
    print training_set_prep[0]
    print held_out_set_prep[0]
    print test_set_prep[0]

    """ Estimate a bigram_lm object, check its distribution, compute its perplexity.
    """
    
    lm1 = BigramLM(vocabulary)
    lm1.getCounts(train_list)
    
    lm1.EstimateBigrams('no smoothing')
    
    print " Distribution check result:  ", lm1.CheckDistribution()

    
    print ' Perplexity without smoothing:  ', lm1.Perplexity(test_list,'no smoothing')

    

    """ Print out perplexity after Laplace smoothing.
    """ 
    
    lm1.EstimateBigrams('Laplace interpolation')
    print ' Perplexity with Laplace smoothing:  ', lm1.Perplexity(test_list,'Laplace interpolation')

    """ Print out perplexity after simple linear interpolation (SLI) with lambda = 0.5.
    """ 
    
    lm1.EstimateBigrams('linear interpolation')
    print ' Perplexity with linear interpolation:  ', lm1.Perplexity(test_list,'linear interpolation')


    """ Estimate interpolation weights using the deleted interpolation algorithm on the 
    held out set and print out.
    """ 
    
    lm1.EstimateBigrams('deleted interpolation',held_out_list)
    print 'lambda 1 = ', lm1.lambda1, '\n'
    print 'lambda 2 = ', lm1.lambda2, '\n'


    """ Print out perplexity after simple linear interpolation (SLI) with the estimated
    interpolation weights.
    """ 

    print ' Perplexity with deleted interpolation:  ', lm1.Perplexity(test_list,'deleted interpolation')




if __name__ == "__main__": 
    main()







    