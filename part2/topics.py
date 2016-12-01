from collections import defaultdict
import sys
import os
import math
import pickle
import string
import time
import random
import heapq

def dd():
    return defaultdict(int)

class topics:

    def __init__(self):
        #Number of Documents
        self.priorD = defaultdict(int)
        #words for each topic
        self.likelihoodDD = defaultdict(dd)
        #total count of words for each topic
        self.topicWordCount = defaultdict(int)
        #Store words against each document
        self.docsWordsDL = defaultdict(list)
        #store the names of all topics
        self.topicL = []
        #used to store the list of documents that has topic
        self.labelledFileD = defaultdict(list)
        #used to store the list of documents whose label is not known
        self.unlabelledFileD = defaultdict(list)
        #used to store the list of documents whose label has been found out
        self.afterLabellingD = defaultdict(list)
        #used to store topics in each cluster
        self.clusterTopicD = defaultdict(list)
        self.tempDict = defaultdict()
        #self.skipWords = ["a","an","the","from","to","of","as","hi","hello","on","is","i","in","subject","has","have","my","and","or"]
        #self.skipWords = ["a","also","am","an","and","any","are","as","at","be","been","but","by","can","come","could","did","do","for","from","get","go","had","has","have","he","her","here","him","his","how","i","if","in","is","it","its","may","me","more","most","my","new","no","not","now","of","on","one","or","our","over","pm","see","she","so","than","that","the","their","them","then","there","these","they","this","to","us","was","we","web","were","what","when","which","who","will","with","would","you","your","subject"]
        self.skipWords = ["from","to","a","an","the","is","subject","are","were","i","and", "", "of", "that", "in", "it", "not", "this", "be", "for", "as", "on", "if", "on", "my" ,"was", "we", "but", "he", "you", "have", "with", "by", "all", "or", "at", "me", "so", "can", "do"]

    def flipCoin(self):
        return random.uniform(0.0,1.0)

    def printWords(self):
        file = open("distinctive_words.txt", "wb")

        for dirTopic in self.likelihoodDD:
            topWordsL = heapq.nlargest(10, self.likelihoodDD[dirTopic], self.likelihoodDD[dirTopic].get)
            topicLine = str(dirTopic) + ":" + " ".join(map(str, topWordsL)) + "\n"
            file.write(topicLine)

        file.close()

    def createFileListing(self,directory,fraction,mode):
        #Create a list of all topics in the directory
        self.topicL = os.listdir(directory)
        #iterate through each topic
        for dirTopic in self.topicL:
            self.tempDict[dirTopic] = dirTopic
            #Create a path for the actual files
            filePath = directory + "/" + dirTopic
            #Create a list of files present in a topic directory
            allFiles = os.listdir(filePath)
            labelledFileNPath = []
            unlabelledFileNPath = []
            #Create a List of files with entire path in the current topic
            if mode =='train':
                for fileName in allFiles:
                    #Tossing a baised coin
                    if(self.flipCoin() <= fraction):
                        #append path of files whose label is known
                        labelledFileNPath.append([filePath + "/" + fileName])
                    else:
                        #append path of files whose label is unknown
                        unlabelledFileNPath.append([filePath + "/" + fileName])
                #Save the list against the topic
                if fraction != 0 and len(labelledFileNPath) == 0:
                    labelledFileNPath.append([filePath + "/" + allFiles[0]])

                self.labelledFileD[dirTopic] = labelledFileNPath
                self.unlabelledFileD[dirTopic] = unlabelledFileNPath
            else:
                for fileName in allFiles:
                    #append path of files whose label is unknown
                    unlabelledFileNPath.append([filePath + "/" + fileName])
                #Save the list against the topic
                self.unlabelledFileD[dirTopic] = unlabelledFileNPath

    def assignRandomTopics(self, directory):
        for dirTopic in self.topicL:
            #Create a path for the actual files
            #filePath = directory + "/" + dirTopic
            #Create a list of files present in a topic directory
            #allFiles = os.listdir(filePath)
            allFiles = self.unlabelledFileD[dirTopic]
            #print self.labelledFileD,"ANAND",self.afterLabellingD
            #print allFiles
            #Iterate over each file
            for fileName in allFiles:
                #print fileName[0]
                #increment the number of documents for current topic
                currTopic = self.topicL[int(round(random.uniform(0.0,1.94) * 10))]
                self.priorD[currTopic] += 1
                #Open each file
                #with open(filePath + "/" + fileName) as fp:
                with open(fileName[0]) as fp:
                    #Read entire file and return as list of words
                    allWords = fp.read().split()
                    #Iterate over all words in the list
                    for eachWord in allWords:
                        eachWordLower = eachWord.lower().translate(None,string.punctuation)
                        if eachWordLower not in self.skipWords:
                            #increment tne count for current word in current topic
                            self.likelihoodDD[currTopic][eachWordLower] += 1
                            #increment count of words for curr topic
                            self.topicWordCount[currTopic] += 1

    def trainModel(self,directory,fraction):
        if fraction == 1:
            self.calcPriorLikelihood(directory)
        elif fraction != 0:
            for idx in range(5):
                self.calcPriorLikelihood(directory)
                self.classify(directory)
                #print "Nahar idx",idx
            self.calcPriorLikelihood(directory)
        elif fraction == 0:
            self.assignRandomTopics(directory)
            for idx in range(1):
                self.classify(directory)
                self.calcPriorLikelihood(directory)
                #print "Nahar idx",idx

            for dirTopic in self.topicL:
                cluTopicL = []
                for filename in self.afterLabellingD[dirTopic]:
                    cluTopicL += [filename[0].split("/")[-2]]
                self.clusterTopicD[dirTopic] = cluTopicL

            #for dirTopic in self.topicL:
            usedTopic = []
            for idx in range (20):
                length = len(self.clusterTopicD[dirTopic])
                newTopic = self.clusterTopicD[dirTopic][random.randint(0,length-1)]
                #print idx
                if newTopic in usedTopic:
                    idx -= idx
                    continue
                usedTopic.append(newTopic)
                self.tempDict[dirTopic] = newTopic
                #print self.tempDict[dirTopic]

    def calcPriorLikelihood(self,directory):
        self.likelihoodDD.clear()
        self.topicWordCount.clear()
        self.priorD.clear()
        #Create a list of all topics in the directory
        #self.topicL = os.listdir(directory)
        #Iterate over each topic 
        #ANAND NAHAR
        sumCount = 0
        for dirTopic in self.topicL:
            #Create a path for the actual files
            #filePath = directory + "/" + dirTopic
            #Create a list of files present in a topic directory
            #allFiles = os.listdir(filePath)
            allFiles = self.labelledFileD[dirTopic] + self.afterLabellingD[dirTopic]
            #print self.labelledFileD,"ANAND",self.afterLabellingD
            #print allFiles
            #Iterate over each file
            for fileName in allFiles:
                #print fileName[0]
                #increment the number of documents for current topic
                self.priorD[dirTopic] += 1
                #Open each file
                #with open(filePath + "/" + fileName) as fp:
                with open(fileName[0]) as fp:
                    #Read entire file and return as list of words
                    allWords = fp.read().split()
                    #Iterate over all words in the list
                    for eachWord in allWords:
                        eachWordLower = eachWord.lower().translate(None,string.punctuation)
                        if eachWordLower not in self.skipWords:
                            #increment tne count for current word in current topic
                            self.likelihoodDD[dirTopic][eachWordLower] += 1
                            #increment count of words for curr topic
                            self.topicWordCount[dirTopic] += 1
            sumCount += self.topicWordCount[dirTopic]

        print sumCount

    def classify(self,directory):
        self.afterLabellingD.clear()
        totalFiles = 0
        accurateCount = 0
        totalDocs = sum(self.priorD.values())
        #Iterate over each topic
        for dirTopic in self.topicL:
            finalTopic = dirTopic
            #Create a path for the actual files
            #filePath = directory + "/" + dirTopic
            #Create a list of files present in a topic directory
            #allFiles = os.listdir(filePath)
            allFiles = self.unlabelledFileD[dirTopic]
            #Iterate over each file
            for fileName in allFiles:
                totalFiles += 1
                maxProb = float('-inf')
                #Open each file
                with open(fileName[0]) as fp:
                    #Read entire file and return as list of words
                    allWords = fp.read().split()
                    #Iterate over all topics
                    for currTopic in self.topicL:
                        postProb = 0.0
                        #Take log of prior - number of documents for current topic
                        #print self.priorD[currTopic]
                        #print totalDocs
                        priorProb = math.log(self.priorD[currTopic] / float(totalDocs))
                        #Total number of words for current topic
                        totalWords = float(self.topicWordCount[currTopic])
                        #Iterate over all words in the list
                        for eachWord in allWords:
                            eachWordLower = eachWord.lower().translate(None,string.punctuation)
                            #eachWordLower = eachWord.lower()
                            #if(self.likelihoodDD[currTopic][eachWordLower] != 0 and eachWordLower.isalpha()):
                                #postProb +=  math.log(self.likelihoodDD[currTopic][eachWordLower] / totalWords)
                            #else:
                                #ELSECNT += 1
                            #TOTALCNT += 1
                            #if eachWordLower.isalpha():
                                #pass
                            if eachWordLower not in self.skipWords: 
                                postProb +=  math.log(max (1E-9,self.likelihoodDD[currTopic][eachWordLower] / totalWords))
                        postProb += priorProb

                        if postProb > maxProb:
                            maxProb = postProb
                            finalTopic = currTopic

                if mode == 'test' and finalTopic == self.tempDict[dirTopic]:
                    accurateCount += 1
                elif mode == 'train':
                    self.afterLabellingD[finalTopic].append([fileName[0]])

        #print self.afterLabellingD

        #print totalFiles, accurateCount
        if mode == 'test':
            print "Accuracy:", ( accurateCount / float(totalFiles) ) * 100

#Start main program
topicObj = topics()

if len(sys.argv) < 4:
    print "Please enter valid parameters..Exiting!!"
    sys.exit()
mode, directory, modelFile = sys.argv[1:4]

if mode == "train":
    if len(sys.argv) < 5:
        print "Enter fraction value..Exiting!!"
        sys.exit(0)
    else:
        fraction = float(sys.argv[4])

if mode == 'train':
    stime = time.time()
    topicObj.createFileListing(directory,fraction,mode)
    topicObj.trainModel(directory,fraction)
    topicObj.printWords()
    pickle.dump(topicObj, open(modelFile, "wb"))
    etime = time.time()
    print "Time",etime - stime
    print "Training complete.", modelFile ,"created!!"
else:
    stime = time.time()
    topicObj = pickle.load(open(modelFile, "rb"))
    topicObj.createFileListing(directory,1.0,mode)
    topicObj.classify(directory)
    etime = time.time()
    print "Time",etime - stime

sys.exit(0)
