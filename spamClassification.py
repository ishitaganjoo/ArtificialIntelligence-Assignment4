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
                self.mostFreq = []
                self.documentSpamFreq = defaultdict(int)
                self.documentNSpamFreq = defaultdict(int)
                self.entropyDict = defaultdict(int)
                self.spamWordDocList = []
                self.nSpamWordDocList = []
                self.allWords = defaultdict(int)
                self.spamDict = {}
                self.notSpamDict = {}
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
                countSpam,countNSpam = 0.0,0.0
                priorSpam,priorNSpam= 0.0,0.0
                dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
                dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
                
                mailList= []
                stopWordList = ['from', 'to', 'a', 'an', 'subject', 'are', 'were', 'is', 'the']
                
                #allWords = defaultdict(int)
                testDict = {}
                                
                for fileName in dirNSpam:
                        countNSpam += 1
                        with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                                content = f.readlines()
                                newStr = "".join(content)
                                mailList.append(newStr.split(' '))
                for mail in mailList:
                        uniqueWords = []
                        wordsListPerMail = []
                        for eachWord in mail:
                                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                                if lowerEachWord not in stopWordList:
                                        if (self.notSpamDict.get(lowerEachWord) != None):
                                                self.notSpamDict[lowerEachWord] += 1
                                        else:
                                                self.notSpamDict[lowerEachWord] = 1
                                        
                                        self.allWords[lowerEachWord] += 1
                                        wordsListPerMail.append(lowerEachWord)
                                        
                                        if lowerEachWord not in uniqueWords:
                                                uniqueWords.append(lowerEachWord)
                                        #if lowerEachWord not in self.totalWords:
                                                #self.totalWords.append(lowerEachWord)        
                                while uniqueWords:
                                        newWord = uniqueWords.pop()
                                        self.documentNSpamFreq[newWord] += 1
                                        if newWord.isalpha():
                                                self.allWordsDocFreq[newWord] += 1        
                                self.nSpamWordDocList.append(wordsListPerMail)

                #print(self.nSpamWordDocList)
                        
                        

                mailList = []
                for fileName in dirSpam:
                        countSpam += 1
                        with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                                content = f.readlines()
                                newStr = "".join(content)
                                mailList.append(newStr.split(' '))
                for mail in mailList:
                        uniqueWords = []
                        wordsListPerMail = []
                        for eachWord in mail:
                                lowerEachWord = eachWord.lower().translate(None, string.punctuation)
                                if lowerEachWord not in stopWordList:
                                        if(self.spamDict.get(lowerEachWord) != None):
                                                self.spamDict[lowerEachWord] += 1
                                        else:
                                                self.spamDict[lowerEachWord] = 1
                                        self.allWords[lowerEachWord] += 1
                                        wordsListPerMail.append(lowerEachWord)
                                        if lowerEachWord not in uniqueWords:
                                                uniqueWords.append(lowerEachWord)
                                        #if lowerEachWord not in self.totalWords:
                                                #self.totalWords.append(lowerEachWord)        
                                while uniqueWords:
                                        newWord = uniqueWords.pop()
                                        self.documentSpamFreq[newWord] += 1
                                        if newWord.isalpha():
                                                self.allWordsDocFreq[newWord] += 1        
                                self.spamWordDocList.append(wordsListPerMail)
                self.priorSpam = countSpam/(countSpam+countNSpam)
                self.priorNSpam = countNSpam/(countSpam+countNSpam)        
                
                #combine and select top 50 in both, make a table with rows as docs and columns as words.
                
                #get the keyset of allWords dict to access each word in the dataset
                words = self.allWords.keys()
                abc = []        
                #self.calculateEntropy(words,countNSpam+countSpam, countSpam, countNSpam)
                #abc = self.allWordsDocFreq{k:v for (k, v) in self.allWordsDocFreq.items() if v>10}
                self.totalWordsList  = [self.allWordsDocFreq[i] for i in self.allWordsDocFreq if self.allWordsDocFreq[i] > 10]
                #self.totalWordList = ['a','an','password','the']
                print "total words are", len(self.totalWordsList)
                self.decisionMatrix(dirSpam, dirNSpam)
                #print(self.entropyForWords())
                #self.calculateNaiveBayes(self.spamDict, self.notSpamDict,  countSpam, countNSpam)

        def decisionMatrix(self, dirSpam, dirNSpam):
                mailList = []
                for fileName in dirSpam: 
                        with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
                                content = f.readlines()
                                newStr = "".join(content)
                                mailList.append(newStr.split(' '))
                for mail in mailList:
                        #print("Ishita pagal hai, sadi hui")
                        listIndex = []
                        flag  = 0
                        listIndex = [0]*len(self.totalWordsList)
                        for wordMail in mail:
                                if wordMail in self.totalWordsList:
                                        listIndex[self.totalWordsList.index(wordMail)] = 1
                                '''
                                for i in range(0, len(self.totalWordsList)):
                                        if self.totalWordsList[i] == wordMail:
                                                listIndex[i] = 1
                                                break'''
                        listIndex.append('spam')
                        self.allWordsMatrix.append(listIndex)

                        #print("Rohil", len(self.allWordsMatrix))

                mailList = []
                for fileName in dirNSpam:
                        with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
                                content = f.readlines()
                                newStr = "".join(content)
                                mailList.append(newStr.split(' '))
                for mail in mailList:
                        listIndex = []
                        flag = 0
                        listIndex = [0]*len(self.totalWordsList)
                        for wordMail in mail:
                                if wordMail in self.totalWordsList:
                                        listIndex[self.totalWordsList.index(wordMail)] = 1
                                '''
                                for i in range(0, len(self.totalWordsList)):
                                        if self.totalWordsList[i] ==  wordMail:
                                                listIndex[i] = 1
                                                break'''
                        listIndex.append('notspam')
                        self.allWordsMatrix.append(listIndex)
        
        def entropyCalculate(self, wordsMatrix):
                results = {}
                for mail in wordsMatrix:
                        r = mail[len(mail) - 1]
                        if r not in results:
                                results[r] = 1
                        else:
                                results[r] += 1
                return results

        def entropyForWords(self, wordsMatrix):
                results = self.entropyCalculate(wordsMatrix)
                entropy = 0.0
                for r in results.keys():
                        p = results[r] / float(len(wordsMatrix))
                        entropy = entropy-p*(math.log(p))
                return entropy

        def divideSet(self, rows, column, value):
                split_function = None
                if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
                              split_function=lambda row:row[column]>=value
                   else:
                              split_function=lambda row:row[column]==value
   
                   # Divide the rows into two sets and return them
                set1=[row for row in rows if split_function(row)]
                set2=[row for row in rows if not split_function(row)]
                return (set1, set2)

        def buildTree(self, bestSets):
                print "ANAND",len(bestSets)
                if len(bestSets) == 0: 
                        return spamClassification()
                current_score = self.entropyForWords(bestSets)
                
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
                                (set1, set2) = self.divideSet(bestSets, col, value)
                                p = float(len(set1)) / float(len(bestSets))
                                gain = current_score - p*self.entropyForWords(set1)-(1-p)*self.entropyForWords(set2)
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
                else :
                        return spamClassification(results = self.entropyCalculate(bestSets))


        def classify(self, observation, tree):
                print "NAHAR"
                if tree.results != None:
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
                        return self.classify(observation, branch)

        def testDecisionTree(self):
                dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam")
                dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
                mailList = []
                for fileName in dirSpam:
                        with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
                                content = f.readlines()
                                newStr = "".join(content)
                                mailList.append(newStr.split(' '))
                for mail in mailList:
                        #print("Ishita pagal hai, sadi hui")
                        listIndex = []
                        flag  = 0
                        for word in self.totalWordsList:
                                for wordMail in mail:
                                        if word == wordMail:
                                                flag = 1
                                                listIndex.append(1)
                                                break
                                if flag == 0:
                                        listIndex.append(0)
                        #print("Rohil", self.allWordsMatrix)
                        tempTree = self.buildTree(self.allWordsMatrix)
                        self.classify(listIndex, tempTree)
                        self.allWordsMatrix.append(listIndex)
                                
        def calculateNaiveBayes(self, spamDict, notSpamDict, countSpam, countNSpam):
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

                                
        def calculateEntropy(self,words,totalDocs, countSpam, countNSpam):
                entropyDict = defaultdict(int)
                for word in words:
                        numberSpam,numberNSpam,spamEntropy,nonSpamEntropy,entropyLeftSpam,entropyLeftNSpam, entropyRightSpam, entropyRightNSpam = 0,0,0,0,0,0,0,0
                        if self.documentSpamFreq.get(word) != None:
                                numberSpam = self.documentSpamFreq[word]
                        if self.documentNSpamFreq.get(word) != None:        
                                numberNSpam = self.documentNSpamFreq[word]
                        totalOccOfWord = numberSpam + numberNSpam
                        totalNonOccOfWord = totalDocs - (totalOccOfWord)
                        spamNotContainingWord = countSpam - numberSpam
                        notSpamNotContainingWord = countNSpam - numberNSpam


                        entropyBefore = -(float(countSpam/ totalDocs) * math.log(float(countSpam/ totalDocs))) - (float(countNSpam / totalDocs) * 
                        math.log(float(countNSpam/ totalDocs)))        
                        if numberSpam!=0.0:
                                entropyLeftSpam = -(float(float(numberSpam) / float(totalOccOfWord)) * math.log(float(float(numberSpam) / float(totalOccOfWord))))
                        if numberNSpam!=0.0:
                                entropyLeftNSpam = - (float(float(numberNSpam) / float(totalOccOfWord)) * math.log(float(float(numberNSpam) / float(totalOccOfWord))))
                        entropyLeft = entropyLeftSpam+entropyLeftNSpam
                        
                        if spamNotContainingWord != 0.0:
                                entropyRightSpam = -(float(float(spamNotContainingWord) / float(totalNonOccOfWord)) 
                                * math.log(float(float(spamNotContainingWord) / float(totalNonOccOfWord))))
                        if notSpamNotContainingWord!=0.0:
                                entropyRightNSpam = - (float(float(notSpamNotContainingWord) / float(totalNonOccOfWord)) * 
                                math.log(float(float(notSpamNotContainingWord) / float(totalNonOccOfWord))))        
                        entropyRight = entropyRightSpam+entropyRightNSpam
                        
                        entropyAfter = (float(totalOccOfWord / totalDocs)*entropyLeft) + (float(totalNonOccOfWord / totalDocs) * entropyRight)
                        
                        infoGain = entropyBefore - entropyAfter

                        entropyDict[word] = infoGain
                        
                maxInfoGain = max(entropyDict, key=(lambda key: entropyDict[key]))
                print("Rohil",maxInfoGain)
                infoGainMail = []
                nInfoGainMail = []
                for i in range(0, len(self.spamWordDocList)):
                        if(maxInfoGain in self.spamWordDocList[i]):
                                for j in range(0, len(self.spamWordDocList[i])):
                                        if self.spamWordDocList[i][j] not in infoGainMail and self.spamWordDocList[i][j] != maxInfoGain:
                                                infoGainMail.append(self.spamWordDocList[i][j])
                        else:
                                for j in range(0, len(self.spamWordDocList[i])):
                                        if self.spamWordDocList[i][j] not in nInfoGainMail:
                                                nInfoGainMail.append(self.spamWordDocList[i][j])
                for i in range(0, len(self.nSpamWordDocList)):
                        if(maxInfoGain in self.nSpamWordDocList[i]):
                                for j in range(0, len(self.nSpamWordDocList[i])):
                                        if self.nSpamWordDocList[i][j] not in infoGainMail and self.nSpamWordDocList[i][j] != maxInfoGain:
                                                infoGainMail.append(self.nSpamWordDocList[i][j])
                        else:
                                for j in range(0, len(self.nSpamWordDocList[i])):
                                        if self.nSpamWordDocList[i][j] not in nInfoGainMail:
                                                nInfoGainMail.append(self.nSpamWordDocList[i][j])
                
                #print("abc",infoGainMail)                
                #print("cde",nInfoGainMail)
                #self.calculateEntropy(infoGainMail, totalDocs, countSpam, countNSpam)
                print(len(self.documentSpamFreq))
                print(len(self.documentNSpamFreq))

                #for i in entropyDict:
                        #print(i, entropyDict[i])
                #print(max(entropyDict, key=(lambda key: entropyDict[key])))        
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
