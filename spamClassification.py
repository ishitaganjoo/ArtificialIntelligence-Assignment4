import os
from _collections import defaultdict
from collections import Counter
import math
import string
import sys
import pickle
import time

# Naive Baye's Algorithm:
# We start by creating a method 'createDictionary' , where we read both spam and non spam mails.
# We start by reading non spam mails, create a dictionary notSpamDict to store the frequency of all the non spam words. Then we create another dictionary 
# documentNSpamFreq which stores the occurence of unique words in non spam mails. Similarly, we create another dictionary allWordsDocFreq to sore the document 
# frequency of both spam and non spam words.
# Similarly, we read the spam mails, and create spamDict, documentSpamFreq to store the total word frequency and document frequency respectively.
# Next, we calculate the priors and create a totalWordsList which contains all the words in allWordsDocFreq which have a document freq greater than 10.
# We use the method 'calculateNaiveBayes' to calculate the accuracy of the bayes classifier for both spam and non spam mails.
# In this method, we calculate the probability of a document being spam or non spam using the training data.
# For binary model, we calculate the probability of a document being spam or non spam dividing the frequency of the word in all the training documents(spam, non spam)
# by the total number of training documents(spam or non spam).
# For continuous model, we calculate the probability of the document being spam or non spam by dividing the total frequency of the word in all training 
# docs(spam, non spam) by the total number of words in the training data(spam, non spam).
# For both spam and non spam mails, we calculate the accuracy by comparing the probability of spam and non spam for each mail, and accordingly we classify the mail.

# Decision Tree Algorithm:
# In the decision tree algorithm we find the best splits among the words. Our approach is that first we create a list of about 5000 most important words and then try
# to find those words in every document, if the word is found in the document, then we set the flag to 1, else it is 0. This is for the binary case and for the
# continuous we keep track of the frequency of the the words in our list and see how many times they occur in the particular document or the mail. And based on this
# matrix, we decide the best split of our data by calculating entropy for each split and selecting the split which gives the min entropy and based on thay split,
# we recursively call our tree on that split and try to find the next best split. We continue this procedure until we can say that no more split is possible and 
# we can classify our document as spam or non spam.
  
class spamClassification:
    def __init__(self, col=-1, value = None, results= None, tb= None, fb = None):
        self.priorSpam = 0.0
        self.priorNSpam = 0.0
        self.documentSpamFreq = defaultdict(int)
        self.documentNSpamFreq = defaultdict(int)
        self.spamDict = defaultdict(int)
        self.notSpamDict = defaultdict(int)
        self.totalWordsList = []
        #to mark 1 and 0
        self.allWordsMatrix = []
        #to mark all frequencies
        self.totalWordsMatrix = []
        self.allWordsDocFreq = defaultdict(int)
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
	self.countSpam = 0.0
	self.countNSpam = 0.0
        self.stopWordList = ['from', 'to', 'a', 'an', 'subject', 'are', 'were', 'is', 'the']

    def createDictionary(self):
        self.countSpam,self.countNSpam = 0.0,0.0
        self.priorSpam,self.priorNSpam= 0.0,0.0
        dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
        dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
        
        mailList= []
        
        for fileName in dirNSpam:
            self.countNSpam += 1
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            uniqueWords = []
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord not in self.stopWordList:
                    self.notSpamDict[lowerEachWord] += 1
                    if lowerEachWord not in uniqueWords:
                        uniqueWords.append(lowerEachWord)

            while uniqueWords:
                newWord = uniqueWords.pop()
                self.documentNSpamFreq[newWord] += 1
                self.allWordsDocFreq[newWord] += 1    

        mailList = []
        for fileName in dirSpam:
            self.countSpam += 1
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            uniqueWords = []
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord not in self.stopWordList:
                    self.spamDict[lowerEachWord] += 1
                    if lowerEachWord not in uniqueWords:
                        uniqueWords.append(lowerEachWord)
                
            while uniqueWords:
                newWord = uniqueWords.pop()
                self.documentSpamFreq[newWord] += 1
                self.allWordsDocFreq[newWord] += 1    

        self.priorSpam = self.countSpam / float(self.countSpam + self.countNSpam)
        self.priorNSpam = self.countNSpam / float(self.countSpam + self.countNSpam)    
        
        #get the keyset of allWords dict to access each word in the dataset
        self.totalWordsList  = [i for i in self.allWordsDocFreq if self.allWordsDocFreq[i] > 10]
        self.decisionMatrix(dirSpam, dirNSpam)

    def decisionMatrix(self, dirSpam, dirNSpam):
        mailList = []
        for fileName in dirSpam: 
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndexContinuous = []
            listIndex = [0]*(len(self.totalWordsList) + 1)
            listIndexContinuous = [0]*(len(self.totalWordsList) + 1)
            flag  = 0
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1
                    listIndexContinuous[self.totalWordsList.index(lowerEachWord)] += 1

            listIndex.append('spam')
            listIndexContinuous.append('spam')
            self.allWordsMatrix.append(listIndex)
            self.totalWordsMatrix.append(listIndexContinuous)

        mailList = []
        for fileName in dirNSpam:
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndex = [0]*(len(self.totalWordsList) + 1)
            listIndexContinuous = []
            listIndexContinuous = [0]*(len(self.totalWordsList) + 1)
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1
                    listIndexContinuous[self.totalWordsList.index(lowerEachWord)] += 1

            listIndex.append('notspam')
            listIndexContinuous.append('notspam')
            self.allWordsMatrix.append(listIndex)
            self.totalWordsMatrix.append(listIndexContinuous)

    def entropyCalculate(self, wordsMatrix):
        results = {}
        for mail in wordsMatrix:
            r = mail[len(mail) - 1]
            if r not in results: results[r] = 0
            results[r] += 1
        return results

    def entropyForWords(self, wordsMatrix):
        from math import log
        log2=lambda x:log(x)/log(2)
        results = self.entropyCalculate(wordsMatrix)
        entropy = 0.0
        for r in results.keys():
            p = float(results[r]) / len(wordsMatrix)
            entropy = entropy-p*log2(p)
        return entropy

    def splitOnWords(self, rows, column, value):
        split_function = None
        if isinstance(value,int): # check if the value is a number i.e int or float
                  split_function=lambda row:row[column]>=value
        else:
                  split_function=lambda row:row[column]==value
   
        # Divide the rows into two sets and return them
        set1=[row for row in rows if split_function(row)]
        set2=[row for row in rows if not split_function(row)]
        return (set1, set2)

    def constructDecTree(self, bestSets, scoref = entropyForWords ):
        if len(bestSets) == 0: 
            return spamClassification()
        current_score = scoref(self,bestSets)
        
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        numberOfWords = len(bestSets[0]) - 1
        
        for col in range(0, numberOfWords):
            words = {}
            for mail in bestSets:
                words[mail[col]] = 1
            for value in words.keys():
                (set1, set2) = self.splitOnWords(bestSets, col, value)
                p = float(len(set1)) / float(len(bestSets))
                gain = current_score - p * scoref(self,set1) - (1 - p) * scoref(self,set2)
                if gain>best_gain and len(set1)>0 and len(set2)>0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        if best_gain > 0:
            trueBranch = self.constructDecTree(best_sets[0])
            falseBranch = self.constructDecTree(best_sets[1])
            return spamClassification(col = best_criteria[0], value = best_criteria[1],tb = trueBranch, fb = falseBranch)
        else:
            return spamClassification(results = self.entropyCalculate(bestSets))


    def assignLabels(self, observation, tree):
        if tree.results != None:
            return tree.results
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.assignLabels(observation, branch)

    def testDecisionTree(self):
        tempTree= self.constructDecTree(self.allWordsMatrix)
        tempTreeCont = self.constructDecTree(self.totalWordsMatrix)
        dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam")
        dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
        mailList = []
        totalCountSpam = 0
        countSpam = 0
        countSpamCont = 0
        for fileName in dirSpam:
            totalCountSpam += 1
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndexCont = []
            listIndex = [0]*len(self.totalWordsList) 
            listIndexCont = [0]*len(self.totalWordsList) 
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1
                    listIndexCont[self.totalWordsList.index(lowerEachWord)] += 1
            res = self.assignLabels(listIndex, tempTree)
            resCont = self.assignLabels(listIndexCont, tempTreeCont)
            if res.keys()[0] == "spam":
                countSpam += 1
            if resCont.keys()[0] == "spam":
                countSpamCont += 1

        mailList = []
        totalCountNSpam = 0
        countNSpam = 0
        countNSpamCont = 0
        for fileName in dirNSpam:
            totalCountNSpam += 1
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndexCont = []
            listIndex = [0]*len(self.totalWordsList) 
            listIndexCont = [0]*len(self.totalWordsList) 
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1
                    listIndexCont[self.totalWordsList.index(lowerEachWord)] += 1

            res = self.assignLabels(listIndex, tempTree)
            resCont = self.assignLabels(listIndexCont, tempTreeCont)
            if res.keys()[0] == "notspam":
                countNSpam += 1
            if resCont.keys()[0] == "notspam":
                countNSpamCont += 1
                 
        print "Spam Accuracy:", countSpam/float(totalCountSpam)
        print "Not Spam Accuracy:", countNSpam/float(totalCountNSpam)
        print "Spam Accuracy Cont:", countSpamCont/float(totalCountSpam)
        print "Not Spam Accuracy Cont:", countNSpamCont/float(totalCountNSpam)
                
    def calculateNaiveBayes(self):
        mailList = []
        
        dirTestSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
        testDocs = 0.0
        accuracyCount = 0.0
        accuracyContinuous = 0.0
        for fileName in dirTestSpam:
            testDocs += 1
            mailList= []
            probSpam = 0.0
            probNSpam = 0.0
            probSpamCont = 0.0
            probNSpamCont = 0.0
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
                spamDictSum = sum(self.spamDict.values())
                notSpamDictSum = sum(self.notSpamDict.values())
                for eachWord in mailList[0]:
                    lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                    #for binary
                    if(self.documentSpamFreq.get(lowerEachWord) != None):
                        probSpam += math.log(float(self.documentSpamFreq[lowerEachWord])/float(self.countSpam))
                    else:
                        probSpam += math.log(1e-9)
                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord]) / float(self.countNSpam))
                    else:
                        probNSpam += math.log(1e-9)
                    if(self.spamDict.get(lowerEachWord) != None):
                        probSpamCont += math.log(float(self.spamDict[lowerEachWord] / float(spamDictSum)))
                    else:
                        probSpamCont += math.log(1e-9)
                    if(self.notSpamDict.get(lowerEachWord) != None):
                        probNSpamCont += math.log(float(self.notSpamDict[lowerEachWord] / float(notSpamDictSum)))
                    else:
                        probNSpamCont += math.log(1e-9)
                probSpam += math.log(float(self.priorSpam))
                probNSpam += math.log(float(self.priorNSpam))
                probSpamCont += math.log(float(self.priorSpam))
                probNSpamCont += math.log(float(self.priorNSpam))
            if(probSpam > probNSpam):
                accuracyCount += 1
            if(probSpamCont > probNSpamCont):
                accuracyContinuous += 1
        print("Accuracy for spam is", accuracyCount/testDocs)
        print "Accuracy for continuous spam is", accuracyContinuous/testDocs
                
        dirTestNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam")
        testDocs = 0.0
        accuracyCount = 0.0
        accuracyContinuous = 0.0
        for fileName in dirTestNSpam:
            mailList= []
            testDocs += 1
            probNSpam = 0.0
            probSpam = 0.0
            probSpamCont = 0.0
            probNSpamCont = 0.0
            with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
                spamDictSum = sum(self.spamDict.values())
                notSpamDictSum = sum(self.notSpamDict.values())
                for eachWord in mailList[0]:
                    lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord])/float(self.countSpam))
                    else:
                        probNSpam += math.log(1e-9)
                    if(self.documentSpamFreq.get(lowerEachWord) != None):
                        probSpam += math.log(float(self.documentSpamFreq[lowerEachWord]) / float(self.countNSpam))
                    else:
                        probSpam += math.log(1e-9)
                    if(self.spamDict.get(lowerEachWord) != None):
                        probSpamCont += math.log(float(self.spamDict[lowerEachWord] / float(spamDictSum)))
                    else:
                        probSpamCont += math.log(1e-9)
                    if(self.notSpamDict.get(lowerEachWord) != None):
                        probNSpamCont += math.log(float(self.notSpamDict[lowerEachWord] / float(notSpamDictSum)))
                    else:
                        probNSpamCont += math.log(1e-9)

                probNSpam += math.log(float(self.priorNSpam))
                probSpam += math.log(float(self.priorSpam))
                probSpamCont += math.log(float(self.priorSpam))
                probNSpamCont += math.log(float(self.priorNSpam))
            if(probNSpam > probSpam):
                accuracyCount += 1
            if(probNSpamCont > probSpamCont):
                accuracyContinuous += 1
        print("Accuracy non spam is", accuracyCount/testDocs)
        print("Accuracy non spam continuous is", accuracyContinuous/testDocs)


stime = time.time()
(mode, technique, dataDirectory, modelFile) = sys.argv[1:]
if mode == 'train':
    spamObj = spamClassification()
    spamObj.createDictionary()
    #spamObj.createDictionary()
    pickle.dump(spamObj, open(modelFile, "wb"))
elif mode == 'test':
    spamObj = pickle.load(open(modelFile, "rb"))
    #spamObj.createDictionary()
    if technique == 'bayes':
        spamObj.calculateNaiveBayes()
    elif technique == 'dt':
        spamObj.testDecisionTree()
    else:
        print "Invalid Technique Name!"


etime = time.time()
print "time:",etime - stime
#spamObj.calculateNaiveBayes()
