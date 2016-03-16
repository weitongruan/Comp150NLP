class FSA:
    def __init__(self, num_states = 0):
        self.num_states = num_states
        self.transitions = {}
        self.final_states = set()
    """ TODO: Add methods for adding transitions, setting final states, looking up next
    state in the state transitions table, checking whether or not a state is a final 
    (accepting) state. 
    """


    def AddTrans(self, input_symbol, state, next_state):
        self.transitions[(input_symbol, state)] = next_state 
    
    def AddTransList(self, input_list, state, next_state):
        for i in input_list:
            self.AddTrans(i, state, next_state)

    def SetFinalState(self, fs):
        self.final_states.add(fs)

    def FindNextState(self, input_symbol, state):
        return self.transitions[(input_symbol, state)]

    def CheckFinalState(self, state):
        return state in self.final_states



def NDRecognize(input, fsa):
    """ TODO: Implement ND-RECOGNIZE from SLP Figure 2.19, return true or false based on 
    whether or not the fsa object accepts or rejects the input string.
    """
    if len(input) == 0:
        return False
    agenda = [(0,0)]
    current_state, index = agenda.pop()
    while True:
        if AcceptState(input, fsa, current_state, index):
            return True
        else:   
            agenda.extend(GenerateNewStates(input[index], current_state, index, fsa))
        if len(agenda) == 0:
            return False
        else:
            current_state, index = agenda.pop()
            

def GenerateNewStates(input_symbol, current_state, index, fsa):
    if (input_symbol, current_state) in fsa.transitions:
        pot_next = fsa.FindNextState(input_symbol,current_state)
        pot_idx = index + 1
        result = [(pot_next,pot_idx)]
    else:
        result = []
    if ('eps', current_state) in fsa.transitions:
        result.append((fsa.FindNextState('eps', current_state),index))
    return result
    
def AcceptState(input, fsa, current_state, index):
    if index == len(input) and fsa.CheckFinalState(current_state):
        return True
        

def NDRecognizeMulti(input, fsa_list):
    """ TODO: Extend D-RECOGNIZE such that it inputs a list of FSA instead of a single 
    one. This algorithm should accept/reject input strings such as 12/31/2000 based on 
    whether or not the string is in the language defined by the FSA that is the 
    concatenation of the input list of FSA.
    """
    
    """ Notice the case "//" in seps and perhaps "8" in months/days where the first is
    suppose to have only one character and the second two characters.
    """
    index = 0
    current_state = 0
    temp = True
    while len(fsa_list) - current_state:
        if fsa_list[current_state] == months:
            if len(input) < 2:
                temp &= NDRecognize(input, months)
                index += 1
            else:
                mm_input = input[index: index+2]
                if mm_input[-1] not in list2:
                    temp &= NDRecognize(mm_input[0], months)
                    index += 1
                else:
                    temp &= NDRecognize(mm_input, months)
                    index += 2
        elif fsa_list[current_state] == days:
            if len(input) < 2:
                temp &= NDRecognize(input, days)
                index += 1
            else:
                dd_input = input[index: index+2]
                if dd_input[-1] not in list2:
                    temp &= NDRecognize(dd_input[0], days)
                    index += 1
                else:
                    temp &= NDRecognize(dd_input, days)
                    index += 2
        elif fsa_list[current_state] == years:
            temp &= NDRecognize(input[index: index+4], years)
            index += 4
        elif fsa_list[current_state] == seps:
            temp &= NDRecognize(input[index: index+1], seps)
            index += 1
        else:
            return False
        if temp:
            current_state += 1
        else:
            return False
    if index == len(input):
        return True
    else:
        return False

    
""" Below are some test cases. Include the output of this in your write-up and provide 
explanations. 
"""

def Test(months, days, years, seps):
    print "\nTest Months FSA"
    for input in ["", "00", "01", "09", "10", "11", "12", "13", "3", "5"]:
        print "'%s'\t%s" %(input, NDRecognizeMulti(input, [months]))
    print "\nTest Days FSA"
    for input in ["", "00", "01", "09", "10", "11", "21", "31", "32", "4", "6"]:
        print "'%s'\t%s" %(input, NDRecognizeMulti(input, [days]))
    print "\nTest Years FSA"
    for input in ["", "1899", "1900", "1901", "1999", "2000", "2001", "2099", "2100"]:
        print "'%s'\t%s" %(input, NDRecognizeMulti(input, [years]))
    print "\nTest Separators FSA"
    for input in ["", ",", " ", "-", "/", "//", ":"]:
        print "'%s'\t%s" %(input, NDRecognizeMulti(input, [seps]))
    print "\nTest Date Expressions FSA"
    for input in ["", "1/20/1990", "02/5/2016", "5/6/2000", "12 6 1988", 
                  "10-6-2015", "00 6 2015", "10 50 2000", 
                  "10 12 2200"]:
        print "'%s'\t%s" %(input, 
                           NDRecognizeMulti(input, [months, seps, days, seps, years]))



list1 = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
list2 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

""" Build ND-FSA for Months """
months = FSA(3)
months.SetFinalState(3)
months.AddTrans("0", 0, 1)
months.AddTrans("1", 0, 2)
months.AddTransList(list1, 1, 3)
months.AddTransList(["0", "1", "2"], 2, 3)
months.AddTrans('eps', 0, 1) # new !!


""" Build ND-FSA for Days """
days = FSA(4)
days.SetFinalState(4)
days.AddTrans("0", 0, 1)
days.AddTransList(["1", "2"], 0, 2)
days.AddTrans("3", 0, 3)
days.AddTransList(list1, 1, 4)
days.AddTransList(list2, 2, 4)
days.AddTransList(["0", "1"], 3, 4)
days.AddTrans('eps', 0, 1) # new !!

""" Build FSA for Years """
years = FSA(5)
years.SetFinalState(5)
years.AddTrans("1", 0, 1)
years.AddTrans("2", 0, 2)
years.AddTrans("9", 1, 3)
years.AddTrans("0", 2, 3)
years.AddTransList(list2, 3, 4)
years.AddTransList(list2, 4, 5)

""" Build FSA for Separators """
seps = FSA(1)
seps.SetFinalState(1)
seps.AddTransList(["/", " ", "-"], 0, 1)


""" Testing DRecognize"""
"""
result1 = NDRecognize("1",months)
print result1
result2 = NDRecognize("35",months)
print result2
result3 = NDRecognize("25",days)
print result3
result4 = NDRecognize("6",days)
print result4
"""



Test(months, days, years, seps)







