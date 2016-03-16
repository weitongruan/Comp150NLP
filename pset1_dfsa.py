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



def DRecognize(input, fsa):
    """ TODO: Implement D-RECOGNIZE from SLP Figure 2.12, return true or false based on 
    whether or not the fsa object accepts or rejects the input string.
    """

    index = 0
    current_state = 0
    while True:
        if index == len(input):
            if fsa.CheckFinalState(current_state):
                return True
            else:
                return False
        elif (input[index], current_state) not in fsa.transitions:
            return False
        else:
            current_state = fsa.FindNextState(input[index], current_state)
            index += 1


def DRecognizeMulti(input, fsa_list):
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
            temp &= DRecognize(input[index: index+2], months)
            index += 2
        elif fsa_list[current_state] == days:
            temp &= DRecognize(input[index: index+2], days)
            index += 2
        elif fsa_list[current_state] == years:
            temp &= DRecognize(input[index: index+4], years)
            index += 4
        elif fsa_list[current_state] == seps:
            temp &= DRecognize(input[index: index+1], seps)
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
    for input in ["", "00", "01", "09", "10", "11", "12", "13"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [months]))
    print "\nTest Days FSA"
    for input in ["", "00", "01", "09", "10", "11", "21", "31", "32"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [days]))
    print "\nTest Years FSA"
    for input in ["", "1899", "1900", "1901", "1999", "2000", "2001", "2099", "2100"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [years]))
    print "\nTest Separators FSA"
    for input in ["", ",", " ", "-", "/", "//", ":"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [seps]))
    print "\nTest Date Expressions FSA"
    for input in ["", "12 31 2000", "12/31/2000", "12-31-2000", "12:31:2000", 
                  "00-31-2000", "12-00-2000", "12-31-0000", 
                  "12-32-1987", "13-31-1987", "12-31-2150"]:
        print "'%s'\t%s" %(input, 
                           DRecognizeMulti(input, [months, seps, days, seps, years]))



list1 = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
list2 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

""" Build FSA for Months """
months = FSA(3)
months.SetFinalState(3)
months.AddTrans("0", 0, 1)
months.AddTrans("1", 0, 2)
months.AddTransList(list1, 1, 3)
months.AddTransList(["0", "1", "2"], 2, 3)


""" Build FSA for Days """
days = FSA(4)
days.SetFinalState(4)
days.AddTrans("0", 0, 1)
days.AddTransList(["1", "2"], 0, 2)
days.AddTrans("3", 0, 3)
days.AddTransList(list1, 1, 4)
days.AddTransList(list2, 2, 4)
days.AddTransList(["0", "1"], 3, 4)

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
result1 = DRecognize("05",months)
print result1
result2 = DRecognize("35",months)
print result2
result3 = DRecognize("25",days)
print result3
result4 = DRecognize("56",days)
print result4
"""

Test(months, days, years, seps)







