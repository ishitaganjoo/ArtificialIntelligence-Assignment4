import os
from _collections import defaultdict
from collections import Counter
import math
import string
import sys
import pickle
import time
import heapq

######################################################################################################
# Naive Baye's Algorithm:
#
# We start by creating a method 'createDictionary' , where we read both spam and non spam mails.
#
# We start by reading non spam mails, create a dictionary notSpamDict to store the frequency of 
# all the non spam words. Then we create another dictionary documentNSpamFreq which stores 
# the occurence of unique words in non spam mails. Similarly, we create another dictionary 
# allWordsDocFreq to store the document  frequency of both spam and non spam words.
#
# Similarly, we read the spam mails, and create spamDict, documentSpamFreq to store the total 
# word frequency and document frequency respectively.
#
# Next, we calculate the priors and create a totalWordsList which contains all the words in 
# allWordsDocFreq which have a document freq greater than 10.
#
# We use the method 'calculateNaiveBayes' to calculate the accuracy of the bayes classifier 
# for both spam and non spam mails.
#
# In this method, we calculate the probability of a document being spam or non spam using 
# the training data.
#
# For binary model, we calculate the probability of a document being spam or non spam dividing the 
# frequency of the word in all the training documents(spam, non spam) by the total number of training 
# documents(spam or non spam).
#
# For continuous model, we calculate the probability of the document being spam or non spam by 
# dividing the total frequency of the word in all training docs(spam, non spam) by the total number 
# of words in the training data(spam, non spam).
#
# For both spam and non spam mails, we calculate the accuracy by comparing the probability of spam 
# and non spam for each mail, and accordingly we classify the mail.
#
# Decision Tree Algorithm:
#
# In the decision tree algorithm we find the best splits among the words. Our approach is that first
#  we create a list of about 5000 most important words and then try to find those words in every 
# document, if the word is found in the document, then we set the flag to 1, else it is 0. 
# This is for the binary case and for the continuous we keep track of the frequency of the the words 
# in our list and see how many times they occur in the particular document or the mail. And based on 
# this matrix, we decide the best split of our data by calculating entropy for each split and 
# selecting the split which gives the min entropy and based on thay split,
#
# We recursively call our tree on that split and try to find the next best split. We continue this
# procedure until we can say that no more split is possible and we can classify our document as 
# spam or non spam.
######################################################################################################

def dd():
    return defaultdict(int)  

class spamClassification:
    def __init__(self, column =-1, value = None, solution= None, leftBranch= None, rightBranch = None):
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
        self.column = column
        self.value = value
        self.solution = solution
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.countSpam = 0.0
        self.countNSpam = 0.0
        self.topTenSpamBinary = defaultdict(int)
        self.topTenSpamCont  = defaultdict(int)
        self.leastTenSpamBinary = defaultdict(int)
        self.leastTenSpamCont = defaultdict(int)
        self.stopWordList = ["from","to","a","an","the","is","subject","are","were","i","and", "", "of", "that", "in", "it", "not", "this", "be", "for", "as", "on", "if", "on", "my" ,"was", "we", "but", "he", "you", "have", "with", "by", "all", "or", "at", "me", "so", "can", "do", "2002"]
        self.tree1 = None
        self.tree2 = None

    def createDictionary(self,dataDirectory):
        self.countSpam,self.countNSpam = 0.0,0.0
        self.priorSpam,self.priorNSpam= 0.0,0.0
        filePathTrNs = dataDirectory + "/train/notspam"
        filePathTrS = dataDirectory + "/train/spam"
        dirNSpam = os.listdir(filePathTrNs)
        dirSpam = os.listdir(filePathTrS)
        
        mailList= []
        
        for fileName in dirNSpam:
            self.countNSpam += 1
            with open(filePathTrNs + "/" + fileName, "r") as f:
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
            with open(filePathTrS + "/" + fileName, "r") as f:
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
        self.decisionMatrix(dirSpam, dirNSpam, filePathTrNs, filePathTrS)

    def decisionMatrix(self, dirSpam, dirNSpam, filePathTrNs, filePathTrS):
        mailList = []
        for fileName in dirSpam: 
            with open(filePathTrS + "/" + fileName, "r") as f:
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
            with open(filePathTrNs + "/" + fileName, "r") as f:
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
        solution = {}
        for mail in wordsMatrix:
            idx = mail[len(mail) - 1]
            if idx not in solution: solution[idx] = 0
            solution[idx] += 1
        return solution

    def entropyForWords(self, wordsMatrix):
        #from math import log
        #log2=lambda x:log(x)/log(2)
        solution = self.entropyCalculate(wordsMatrix)
        entropy = 0.0
        for idx in solution.keys():
            ans = float(solution[idx]) / len(wordsMatrix)
            #entropy = entropy - ans * log2(ans)
            entropy = entropy - ans * math.log(ans)
        return entropy

    def splitOnWords(self, wordsMatrix, column, value):
        divideNode = None
        divideNode = lambda word:word[column]>=value
   
        # Divide the rows into two sets and return them
        splitLeft = [word for word in wordsMatrix if divideNode(word)]
        splitRight = [word for word in wordsMatrix if not divideNode(word)]
        return (splitLeft, splitRight)

    def constructDecTree(self, optSplit ):
        if len(optSplit) == 0: 
            return spamClassification()
        splitEntropy = self.entropyForWords(optSplit)
        
        optInfoGain = 0.0
        optInfoCri = None
        optSets = None

        numberOfWords = len(optSplit[0]) - 1
        
        for col in range(0, numberOfWords):
            words = {}
            for mail in optSplit:
                words[mail[col]] = 1
            for value in words.keys():
                (splitLeft, splitRight) = self.splitOnWords(optSplit, col, value)
                ans = float(len(splitLeft)) / float(len(optSplit))
                currVal = splitEntropy - ans * self.entropyForWords(splitLeft) - (1 - ans) * self.entropyForWords(splitRight)
                if currVal > optInfoGain and len(splitLeft) > 0 and len(splitRight) > 0:
                    optInfoGain = currVal
                    optInfoCri = (col, value)
                    optSets = (splitLeft, splitRight)

        if optInfoGain > 0:
            lb = self.constructDecTree(optSets[0])
            rb = self.constructDecTree(optSets[1])
            return spamClassification(column = optInfoCri[0], value = optInfoCri[1],leftBranch = lb, rightBranch = rb)
        else:
            return spamClassification(solution = self.entropyCalculate(optSplit))

    def assignLabels(self, label, tree):
        if tree.solution != None:
            return tree.solution
        else:
            val = label[tree.column]
            branch = None
            if val >= tree.value:
                branch = tree.leftBranch
            else:
                branch = tree.rightBranch
            return self.assignLabels(label, branch)

    def testDecisionTree(self, dataDirectory):
        filePathTeNs = dataDirectory + "/test/notspam"
        filePathTeS = dataDirectory + "/test/spam"
        tempTree= self.tree1
        tempTreeCont = self.tree2
        dirNSpam = os.listdir(filePathTeNs)
        dirSpam = os.listdir(filePathTeS)
        mailList = []
        totalCountSpam = 0
        countSpam = 0
        countSpamCont = 0
        for fileName in dirSpam:
            totalCountSpam += 1
            with open(filePathTeS + "/" + fileName, "r") as f:
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
            with open(filePathTeNs + "/" + fileName, "r") as f:
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
                 
        print "Binary Spam Accuracy:", (countSpam/float(totalCountSpam) ) *100
        print "Binary Not Spam Accuracy:",( countNSpam/float(totalCountNSpam) ) * 100
        print "Continuous Spam Accuracy:", (countSpamCont/float(totalCountSpam)) * 100
        print "Continuous Not Spam Accuracy:", (countNSpamCont/float(totalCountNSpam) ) *100
   
                     
    def calculateNaiveBayes(self, dataDirectory):
        confusionMatrix = defaultdict(dd)
        confusionMatrixCont = defaultdict(dd)

        mailList = []

        filePathTeNs = dataDirectory + "/test/notspam"
        filePathTeS = dataDirectory + "/test/spam"
        
        dirTestSpam = os.listdir(filePathTeS)
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
            with open(filePathTeS + "/" + fileName, "r") as f:
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
                        self.topTenSpamBinary[lowerEachWord] =  math.log(float(self.documentSpamFreq[lowerEachWord])/float(self.countSpam))
                    else:
                        probSpam += math.log(1e-9)
                        self.topTenSpamBinary[lowerEachWord] =  math.log(1e-9)

                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord]) / float(self.countNSpam))
                    else:
                        probNSpam += math.log(1e-9)

                    if(self.spamDict.get(lowerEachWord) != None):
                        probSpamCont += math.log(float(self.spamDict[lowerEachWord] / float(spamDictSum)))
                        self.topTenSpamCont[lowerEachWord] =  math.log(float(self.spamDict[lowerEachWord])/float(spamDictSum))
                    else:
                        probSpamCont += math.log(1e-9)
                        self.topTenSpamCont[lowerEachWord] =  math.log(1e-9)

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

        confusionMatrix["spam"]["spam"] = accuracyCount
        confusionMatrix["spam"]["nonspam"] = testDocs - accuracyCount
        
        print "In Naive Bayes:"
        print "Accuracy for binary spam is", (accuracyCount/float (testDocs)) * 100
        print "Accuracy for continuous spam is", accuracyContinuous/float(testDocs) * 100
        print " "
                
        dirTestNSpam = os.listdir(filePathTeNs)
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
            with open(filePathTeNs + "/" + fileName, "r") as f:
                content = f.readlines()
                newStr = "".join(content)
                mailList.append(newStr.split(' '))
                spamDictSum = sum(self.spamDict.values())
                notSpamDictSum = sum(self.notSpamDict.values())
                for eachWord in mailList[0]:
                    lowerEachWord = eachWord.lower().translate(None, string.punctuation)

                    if(self.documentNSpamFreq.get(lowerEachWord) != None):
                        probNSpam += math.log(float(self.documentNSpamFreq[lowerEachWord])/float(self.countSpam))
                        self.leastTenSpamBinary[lowerEachWord] =  math.log(float(self.documentNSpamFreq[lowerEachWord])/float(self.countSpam))
                    else:
                        probNSpam += math.log(1e-9)
                        self.leastTenSpamBinary[lowerEachWord] =  math.log(1e-9)

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
                        self.leastTenSpamCont[lowerEachWord] =  math.log(float(self.notSpamDict[lowerEachWord])/float(notSpamDictSum))
                    else:
                        probNSpamCont += math.log(1e-9)
                        self.leastTenSpamCont[lowerEachWord] =  math.log(1e-9)

                probNSpam += math.log(float(self.priorNSpam))
                probSpam += math.log(float(self.priorSpam))
                probSpamCont += math.log(float(self.priorSpam))
                probNSpamCont += math.log(float(self.priorNSpam))

            if(probNSpam > probSpam):
                accuracyCount += 1

            if(probNSpamCont > probSpamCont):
                accuracyContinuous += 1

        confusionMatrix["nonspam"]["nonspam"] = accuracyCount
        confusionMatrix["nonspam"]["spam"] = testDocs - accuracyCount

        print "Accuracy for binary nonspam is", accuracyCount/float(testDocs) * 100
        print "Accuracy non continuous nonspam is", accuracyContinuous/float(testDocs)
        print " "

        print "" 
        print "Confusion Matrix for naive Bayes binary:"
        print "" 
        print " ".ljust(15) + "spam".ljust(15) + "nonspam".ljust(15)
        print "spam".ljust(15) + (str(confusionMatrix["spam"]["spam"])).ljust(15)\
                 + (str(confusionMatrix["nonspam"]["spam"])).ljust(15)
        print "nonspam".ljust(15) + (str(confusionMatrix["nonspam"]["spam"])).ljust(15)\
                 + (str(confusionMatrix["nonspam"]["nonspam"])).ljust(15)
        print "" 

        print "" 
        print "Confusion Matrix for naive Bayes continuous:"
        print "" 

        print " ".ljust(15) + "spam".ljust(15) + "nonspam".ljust(15)
        print "spam".ljust(15) + (str(confusionMatrixCont["spam"]["spam"])).ljust(15)\
                 + (str(confusionMatrixCont["nonspam"]["spam"])).ljust(15)
        print "nonspam".ljust(15) + (str(confusionMatrixCont["nonspam"]["spam"])).ljust(15)\
                 + (str(confusionMatrixCont["nonspam"]["nonspam"])).ljust(15)

        print ""

    def printTopTenandLeastTen(self):
        print "Top ten spam words Binary",heapq.nlargest(10, self.topTenSpamBinary, key = self.topTenSpamBinary.get)
        print "Top ten spam words Continuous", heapq.nlargest(10, self.topTenSpamCont, key = self.topTenSpamCont.get)
        print "Top ten not spam words Binary", heapq.nlargest(10, self.leastTenSpamBinary, key = self.leastTenSpamBinary.get)
        print "Top ten not spam words Continuous", heapq.nlargest(10, self.leastTenSpamCont, key = self.leastTenSpamCont.get)
        print ""

    def createTree(self, tree, level, indent=''):
    # Is this a leaf node?
        if level > 4:
           return
        if tree.solution!=None:
           print str(tree.solution)
        else:
            # Print the criteria

            print 'Column ' + str(self.totalWordsList[tree.column])+'? '

            # Print the branches
            print indent+'True->',
            self.createTree(tree.leftBranch,level+1,indent+'  ')
            print indent+'False->',
            self.createTree(tree.rightBranch,level+1,indent+'  ')

    def trainTrees(self):
        tempTree= self.constructDecTree(self.allWordsMatrix)
        self.createTree(tempTree, 1)
        tempTreeCont = self.constructDecTree(self.totalWordsMatrix)
        self.createTree(tempTreeCont, 1)
        self.tree1 = tempTree
        self.tree2 = tempTreeCont
        
stime = time.time()
(mode, technique, dataDirectory, modelFile) = sys.argv[1:]
if mode == 'train':
    spamObj = spamClassification()
    spamObj.createDictionary(dataDirectory)
    if technique == 'dt':
        spamObj.trainTrees()
    #spamObj.createDictionary()
    pickle.dump(spamObj, open(modelFile, "wb"))
elif mode == 'test':
    spamObj = pickle.load(open(modelFile, "rb"))
    #spamObj.createDictionary()
    if technique == 'bayes':
        spamObj.calculateNaiveBayes(dataDirectory)
        spamObj.printTopTenandLeastTen()
    elif technique == 'dt':
        spamObj.testDecisionTree(dataDirectory)
    else:
        print "Invalid Technique Name!"


etime = time.time()
print "time:",etime - stime
#spamObj.calculateNaiveBayes()
