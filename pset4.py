import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow, log

unknown_token = "<UNK>"  # unknown word token.

""" Removes all function tags e.g., turns NP-SBJ into NP.
"""         
def RemoveFunctionTags(tree):
    for subtree in tree.subtrees():  # for all nodes of the tree
        # if it's a preterminal node with the label "-NONE-", then skip for now
        if subtree.height() == 2 and subtree.label() == "-NONE-": continue
        nt = subtree.label()  # get the nonterminal that labels the node
        labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
        if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
            subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"

""" Return true if node is a trace node.
"""         
def IsTraceNode(node):
    # return true if the node is a preterminal node and has the label "-NONE-"
    return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"

""" Deletes any trace node children and returns true if all children were deleted.
"""
def RemoveTraces(node):
    if node.height() == 2:  # if the node is a preterminal node
        return False  # already a preterminal, cannot have a trace node child.
    i = 0
    while i < len(node):  # iterate over the children, node[i]
        # if the child is a trace node or it is a node whose children were deleted
        if IsTraceNode(node[i]) or RemoveTraces(node[i]): 
            del node[i]  # then delete the child
        else: i += 1
    return len(node) == 0  # return true if all children were deleted
    
""" Preprocessing of the Penn treebank.
"""
def TreebankNoTraces():
    tb = []
    for t in treebank.parsed_sents():
        if t.label() != "S": continue
        RemoveFunctionTags(t)
        RemoveTraces(t)
        t.collapse_unary(collapsePOS = True, collapseRoot = True)
        t.chomsky_normal_form()
        tb.append(t)
    return tb
        
""" Enumerate all preterminal nodes of the tree.
""" 
def PreterminalNodes(tree):
    for subtree in tree.subtrees():
        if subtree.height() == 2:
            yield subtree
    
""" Print the tree in one line no matter how big it is
    e.g., (VP (VB Book) (NP (DT that) (NN flight)))
"""         
def PrintTree(tree):
    if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
    return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))

""" Build a vocabulary from a data set
"""
def BuildVoc(dataset):
    dic = defaultdict(int)
    voc = set()
    for sent in dataset:
        for word in sent.leaves():
            if word in dic:
                voc.add(word)
            dic[word] += 1
    return voc

""" Preprocessing text treating every word that occurs not more than
    once as an unknown token
"""
def PreprocessText(textset, voc):
    newtext = []
    for sent in textset:
        for NPsubtree in PreterminalNodes(sent):
            if NPsubtree[0] not in voc:
                NPsubtree[0] = unknown_token
        newtext.append(sent)
    return newtext

""" Learning a PCFG from dataset
"""
def PCFGlearning(dataset, start):
    production_list = []
    S = Nonterminal(start)
    for sent in dataset:
        production_list += sent.productions()
    return induce_pcfg(S, production_list)


class InvertedGrammar:
    def __init__(self, pcfg):
        self._pcfg = pcfg
        self._r2l = defaultdict(list)  # maps RHSs to list of LHSs
        self._r2l_lex = defaultdict(list)  # maps lexical items to list of LHSs
        self.BuildIndex()  # populates self._r2l and self._r2l_lex according to pcfg

    def PrintIndex(self, filename):
        f = open(filename, "w")
		for rhs, prods in self._r2l.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		for rhs, prods in self._r2l_lex.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		f.close()
        
    def BuildIndex(self):
        """ Build an inverted index of your grammar that maps right hand sides of all 
        productions to their left hands sides.
        """
        for production in self._pcfg.productions():
            if production.is_lexical():
                self._r2l_lex[production.rhs()].append(production)
            else:
                self._r2l[production.rhs()].append(production)
        self.PrintIndex('index')

    def Parse(self, sent):
        """ Implement the CKY algorithm for PCFGs, populating the dynamic programming 
        table with log probabilities of every constituent spanning a sub-span of a given 
        test sentence (i, j) and storing the appropriate back-pointers. 
        """
        Table = {}
        Back = {}
        for jdx in range(len(sent)):
            for A in self._r2l_lex[sent[jdx]]:
                Table[(jdx, jdx + 1, A.lhs())] = log(A.prob)
            if jdx > 1:
                for idx in reversed(range(jdx-1)):
                    for kdx in range(idx+1, jdx-1):




        
    @staticmethod
    def BuildTree(cky_table, sent):
        """ Build a tree by following the back-pointers starting from the largest span 
        (0, len(sent)) and recursing from larger spans (i, j) to smaller sub-spans 
        (i, k), (k, j) and eventually bottoming out at the preterminal level (i, i+1).
        """

def main():
    treebank_parsed_sents = TreebankNoTraces()
    training_set = treebank_parsed_sents[:3000]
    test_set = treebank_parsed_sents[3000:]
    
    """ Transform the data sets by eliminating unknown words.
    """
    vocabulary = BuildVoc(training_set)
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    """ Print the first trees of both data sets
    """
    print PrintTree(training_set_prep[0])
    print PrintTree(test_set_prep[0])
    
    """ Implement your solutions to problems 2-4.
    """

    """ Training: Learn a PCFG
    """
    pset4_pcfg = PCFGlearning(training_set_prep, "S")
    NP_dic = {}
    for production in pset4_pcfg.productions():
        if str(production.lhs()) == 'NP':
            # NP_dic[(production.lhs(), production.rhs())] = production.prob()
            NP_dic[production] = production.prob()
    print "For NP nonterminal, the total number of productions is: ", len(NP_dic), " \n"
    print "The most probable 10 productions for the NP nonterminal are: \n", sorted(NP_dic, key=NP_dic.get,
                                                                                                reverse=True)[:9]
    """ Testing
    """
    pset4_ig = InvertedGrammar(pset4_pcfg)
    
if __name__ == "__main__": 
    main()  
    





