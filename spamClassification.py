import os
from _collections import defaultdict
from collections import Counter
import math
import string
import sys
import pickle
import time

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
        self.allWordsDocFreq = defaultdict(int)
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

    def createDictionary(self):
        print "In createDictionary"
        countSpam,countNSpam = 0.0,0.0
        priorSpam,priorNSpam= 0.0,0.0
        dirNSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
        dirSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
        
        mailList= []
        stopWordList = ['from', 'to', 'a', 'an', 'subject', 'are', 'were', 'is', 'the']
        
        for fileName in dirNSpam:
            countNSpam += 1
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            uniqueWords = []
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord not in stopWordList:
                    self.notSpamDict[lowerEachWord] += 1
                    if lowerEachWord not in uniqueWords:
                        uniqueWords.append(lowerEachWord)

            while uniqueWords:
                newWord = uniqueWords.pop()
                self.documentNSpamFreq[newWord] += 1
                self.allWordsDocFreq[newWord] += 1    

        mailList = []
        for fileName in dirSpam:
            countSpam += 1
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            uniqueWords = []
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord not in stopWordList:
                    self.spamDict[lowerEachWord] += 1
                    if lowerEachWord not in uniqueWords:
                        uniqueWords.append(lowerEachWord)
                
            while uniqueWords:
                newWord = uniqueWords.pop()
                self.documentSpamFreq[newWord] += 1
                self.allWordsDocFreq[newWord] += 1    

        self.priorSpam = countSpam / float(countSpam + countNSpam)
        self.priorNSpam = countNSpam / float(countSpam + countNSpam)    
        
        #get the keyset of allWords dict to access each word in the dataset
        self.totalWordsList  = [i for i in self.allWordsDocFreq if self.allWordsDocFreq[i] > 10]
        print "Anand total words are", len(self.totalWordsList)
        self.decisionMatrix(dirSpam, dirNSpam)
        print "Out createDictionary"

    def decisionMatrix(self, dirSpam, dirNSpam):
        print "In decisionMatrix"
        mailList = []
        for fileName in dirSpam: 
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndex = [0]*(len(self.totalWordsList) + 1)
            flag  = 0
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1

            listIndex.append('spam')
            self.allWordsMatrix.append(listIndex)

        mailList = []
        for fileName in dirNSpam:
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            listIndex = [0]*(len(self.totalWordsList) + 1)
            flag = 0
            for eachWord in mail:
                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                if lowerEachWord in self.totalWordsList:
                    listIndex[self.totalWordsList.index(lowerEachWord)] = 1

            listIndex.append('notspam')
            self.allWordsMatrix.append(listIndex)

        for idx in self.allWordsMatrix:
            print idx
            print ""

        print "Out decisionMatrix"
    
    def entropyCalculate(self, wordsMatrix):
        print "In entropyCalculate"
        results = {}
        for mail in wordsMatrix:
            r = mail[len(mail) - 1]
            if r not in results:
                results[r] = 0
            else:
                results[r] += 1
        print "Out entropyCalculate"
        return results

    def entropyForWords(self, wordsMatrix):
        print "In entropyForWords"
        from math import log
        log2=lambda x:log(x)/log(2)
        results = self.entropyCalculate(wordsMatrix)
        entropy = 0.0
        for r in results.keys():
            p = results[r] / float(len(wordsMatrix))
            entropy = entropy - p * log2(p)
        print "Out entropyForWords"
        return entropy

    def divideSet(self, rows, column, value):
        print "In divideSet"
        split_function = None
        if isinstance(value,int): # check if the value is a number i.e int or float
                  split_function=lambda row:row[column]>=value
        else:
                  split_function=lambda row:row[column]==value
   
        # Divide the rows into two sets and return them
        set1=[row for row in rows if split_function(row)]
        set2=[row for row in rows if not split_function(row)]
        print "NAHAR set1 set2 len", len(set1),len(set2)
        print "Out divideSet"
        return (set1, set2)

    def buildTree(self, bestSets, scoref = entropyForWords ):
        print "In buildTree"
        print "ANAND",len(bestSets)
        if len(bestSets) == 0: 
            return spamClassification()
        current_score = scoref(self,bestSets)
        
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        numberOfWords = len(bestSets[0]) - 1
        print "NAHAR", numberOfWords
        
        for col in range(0, numberOfWords):
            words = {}
            for mail in bestSets:
                words[mail[col]] = 1
            for value in words.keys():
                print "ISHITA" , value
                (set1, set2) = self.divideSet(bestSets, col, value)
                p = float(len(set1)) / float(len(bestSets))
                gain = current_score - p * scoref(self,set1) - (1 - p) * scoref(self,set2)
                if gain>best_gain and len(set1)>0 and len(set2)>0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        if best_gain > 0:
            print "before true branch", best_sets[0]
            trueBranch = self.buildTree(best_sets[0])
            print "before false branch", best_sets[1]
            falseBranch = self.buildTree(best_sets[1])
            return spamClassification(col = best_criteria[0], value = best_criteria[1],tb = trueBranch, fb = falseBranch)
        else:
            return spamClassification(results = self.entropyCalculate(bestSets))


    def classify(self, observation, tree):
        print "In classify"
        if tree.results != None:
            print "Out classify from if"
            return tree.results
        else:
            v = observation[tree.col]
            branch = None
            print v, tree.value, tree.results, tree.tb, tree.fb, tree.col, observation[-1]
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
            print "Out classify"
            return self.classify(observation, branch)

    def testDecisionTree(self):
        print "in testDecisionTree"
        dirNSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam")
        dirSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
        mailList = []
        for fileName in dirSpam:
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
        for mail in mailList:
            listIndex = []
            flag  = 0
            for word in self.totalWordsList:
                for eachWord in mail:
                    wordMail = eachWord.lower().translate(None, string.punctuation)
                    if word == wordMail:
                        flag = 1
                        listIndex.append(1)
                        break
                if flag == 0:
                    listIndex.append(0)

            tempTree = self.buildTree(self.allWordsMatrix)
            self.classify(listIndex, tempTree)
            self.allWordsMatrix.append(listIndex)
        print "out testDecisionTree"
                
    def calculateNaiveBayes(self, spamDict, notSpamDict, countSpam, countNSpam):
        mailList = []
        
        dirTestSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
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
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
                spamDictSum = sum(self.spamDict.values())
                notSpamDictSum = sum(self.notSpamDict.values())
                for eachWord in mailList[0]:
                    lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                    #for binary
                    if(self.documentSpamFreq.get(lowerEachWord) != None):
                        probSpam += math.log(float(self.documentSpamFreq[lowerEachWord])/float(countSpam))
                    else:
                        probSpam += math.log(1e-9)
                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord]) / float(countNSpam))
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
                #print "prob Spam, not Spam",probSpam, probNSpam
                #rint "prob spam cont, not spam cont", probSpamCont, probNSpamCont
            if(probSpam > probNSpam):
                accuracyCount += 1
            if(probSpamCont > probNSpamCont):
                accuracyContinuous += 1
        print("Accuracy for spam is", accuracyCount/testDocs)
        print "Accuracy for continuous spam is", accuracyContinuous/testDocs
                
        dirTestNSpam = os.listdir("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam")
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
            with open("/u/anahar/fall2016/artificialIntelligence/assign4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam/"+fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
                spamDictSum = sum(self.spamDict.values())
                notSpamDictSum = sum(self.notSpamDict.values())
                for lowerEachWord in mailList[0]:
                    lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord])/float(countSpam))
                    else:
                        probNSpam += math.log(1e-9)
                    if(self.documentSpamFreq.get(lowerEachWord) != None):
                        probSpam += math.log(float(self.documentSpamFreq[lowerEachWord]) / float(countNSpam))
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


#spamObj = spamClassification()
#spamObj.createDictionary()
stime = time.time()
(mode, technique, dataDirectory, modelFile) = sys.argv[1:]
if mode == 'train':
    spamObj = spamClassification()
    #spamObj.createDictionary()
    pickle.dump(spamObj, open(modelFile, "wb"))
elif mode == 'test':
    spamObj = pickle.load(open(modelFile, "rb"))
    spamObj.createDictionary()
    spamObj.testDecisionTree()

etime = time.time()
print "time:",etime - stime
#spamObj.calculateNaiveBayes()
