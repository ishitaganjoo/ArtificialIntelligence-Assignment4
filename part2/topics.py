##########################################################################################
#Assignment 4 - Topic Classification
#
#Fully Supervised Learning (fraction 1.0)
#   Train Mode:
#       In case of fully supervised, we know the label of the entire training data, so 
#       we can classify the entire data in respective topics. Read each file one by one
#       and calculate the likelihood and priors for all files present in all classes.
#       Once this is complete, write the model to the given model file.
#
#Semi Supervised Learning (0.0 < fraction < 1.0) 
#   Train Mode:
#       In this case, a fraction is provided depending on which we have to read the 
#       percentage of files. From each topic select the given fraction of files and 
#       train the model. Once the training is complete, use the rest of file (unlabelled)
#       as test data and try to classify the files. Once these files have been
#       classified, retrain the model, this time considering 100% data set. Once
#       training is complete, re classify the unlabelled files. reapeat the above
#       steps till the data converges. This is EM algorithm. Once this is complete, 
#       write the model to the given model file.
#
#Unsupervised Learning
#   Train Mode:
#       In this case we are not given label of any documents. I have used a random
#       clustering algorithm. Initially all the documents are divided randomly 
#       in 20 clusters. A small looping of EM algorithm is applied to the documents
#       in the 20 cluster so that the somewhat similar documents are clustered 
#       together. Once the final clusters are formed, a file is randomly choosen
#       from each of the clusters and the topic is peeked in the actual training set.
#       The cluster is assigned that topic. Once this is complete, write the model 
#       to the given model file.
#
#Test Mode: Test mode is independent of fraction
#       In case of test mode, we need to load the model from the previously saved
#       model file. For each file, posterior prob is calulated considering each 
#       topic. We assign topic to the file which has the maximum posterior.
#       If the assigned topic is same as the topic, from which we have selected 
#       the file initially, we have successfully classified the file. If the 
#       topic is not same, the file was wrongly classified. Accuracy is classified
#       as a percent of (files_correctly_classified / total_number_of_files)
#
#Output
#   Train Mode:
#       1. Model_File: which will store the model from training phase and will be
#                      used during testing phase
#       2. distinctive words.txt: contains a list of 10 words with highest 
#                      P(T = tijWj) for each of the 20 topics
#
#Results
#   Please find result.txt which has been uploaded on github. It consist a confusion 
#   matrix along with accuracy for fraction = 0, 0.1, 0.5, 0.9, 1.0
#
#Challenges
#   In case of unsupervised, it was difficult to cluster the documents and then 
#   a single document is chosen randomly to determine the topic of the cluster
#   The randomly chosen document may not give us the accurate topic for that
#   cluster
#########################################################################################

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
        self.mappingDict = defaultdict()
        #Words which will be skipped since they are common occuring words
        self.skipWords = ["from","to","a","an","the","is","subject","are","were","i","and", "", "of", "that", "in", "it", "not", "this", "be", "for", "as", "on", "if", "on", "my" ,"was", "we", "but", "he", "you", "have", "with", "by", "all", "or", "at", "me", "so", "can", "do"]

    def flipCoin(self):
        return random.uniform(0.0,1.0)

    def printWords(self):
        file = open("distinctive_words.txt", "wb")

        file.write("  Topic".ljust(30) + "Words".rjust(20) + "\n\n")
        for dirTopic in self.likelihoodDD:
            topWordsL = heapq.nlargest(10, self.likelihoodDD[dirTopic], self.likelihoodDD[dirTopic].get)
            topicLine = str(dirTopic).ljust(15) + ":".ljust(10) + " ".join(map(str, topWordsL)) + "\n"
            file.write(topicLine)

        file.close()

    def createFileListing(self,directory,fraction,mode):
        #Create a list of all topics in the directory
        self.topicL = os.listdir(directory)
        #iterate through each topic
        for dirTopic in self.topicL:
            self.mappingDict[dirTopic] = dirTopic
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
                if fraction != 0.0 and len(labelledFileNPath) == 0:
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
            #Create a list of files present in a topic directory
            allFiles = self.unlabelledFileD[dirTopic]
            #Iterate over each file
            for fileName in allFiles:
                #increment the number of documents for current topic
                currTopic = self.topicL[int(round(random.uniform(0.0,1.94) * 10))]
                self.priorD[currTopic] += 1
                #Open each file
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
        if fraction == 1.0:
            #in case of supervised calculate priority for 100% data
            self.calcPriorLikelihood(directory)
        elif fraction != 0.0:
            #in case of fraction, consider fractional files as labelled and rest as unlabelled and use EM algo till data converges
            for idx in range(5):
                self.calcPriorLikelihood(directory)
                self.classify(directory)
            self.calcPriorLikelihood(directory)
        elif fraction == 0.0:
            #In case of unsupervised, use random clusters
            #Once the data is segregated in different clusters
            #randomly select a file and peek at the topic and use that topic for the cluster
            self.assignRandomTopics(directory)
            for idx in range(1):
                self.classify(directory)
                self.calcPriorLikelihood(directory)

            for dirTopic in self.topicL:
                cluTopicL = []
                for filename in self.afterLabellingD[dirTopic]:
                    cluTopicL += [filename[0].split("/")[-2]]
                self.clusterTopicD[dirTopic] = cluTopicL

            usedTopic = []
            for idx in range (20):
                length = len(self.clusterTopicD[dirTopic])
                newTopic = self.clusterTopicD[dirTopic][random.randint(0,length-1)]
                if newTopic in usedTopic:
                    idx -= idx
                    continue
                usedTopic.append(newTopic)
                self.mappingDict[dirTopic] = newTopic

    def calcPriorLikelihood(self,directory):
        self.likelihoodDD.clear()
        self.topicWordCount.clear()
        self.priorD.clear()
        #Iterate over each topic 
        sumCount = 0
        for dirTopic in self.topicL:
            #Create a list of files present in a topic directory
            allFiles = self.labelledFileD[dirTopic] + self.afterLabellingD[dirTopic]
            #Iterate over each file
            for fileName in allFiles:
                #increment the number of documents for current topic
                self.priorD[dirTopic] += 1
                #Open each file
                with open(fileName[0]) as fp:
                    #Read entire file and return as list of words
                    allWords = fp.read().split()
                    #Iterate over all words in the list
                    for eachWord in allWords:
                        #Clean the data
                        eachWordLower = eachWord.lower().translate(None,string.punctuation)
                        if eachWordLower not in self.skipWords:
                            #increment tne count for current word in current topic
                            self.likelihoodDD[dirTopic][eachWordLower] += 1
                            #increment count of words for curr topic
                            self.topicWordCount[dirTopic] += 1

    def classify(self,directory):
        confusionMatrix = defaultdict(dd)
        self.afterLabellingD.clear()
        totalFiles = 0
        accurateCount = 0
        totalDocs = sum(self.priorD.values())
        #Iterate over each topic
        for dirTopic in self.topicL:
            finalTopic = dirTopic
            #Create a list of files present in a topic directory
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
                        priorProb = math.log(self.priorD[currTopic] / float(totalDocs))
                        #Total number of words for current topic
                        totalWords = float(self.topicWordCount[currTopic])
                        #Iterate over all words in the list
                        for eachWord in allWords:
                            eachWordLower = eachWord.lower().translate(None,string.punctuation)
                            #Calculate posterior probability
                            if eachWordLower not in self.skipWords: 
                                postProb +=  math.log(max (1E-9,self.likelihoodDD[currTopic][eachWordLower] / totalWords))
                        postProb += priorProb

                        #Find the most probable topic
                        if postProb > maxProb:
                            maxProb = postProb
                            finalTopic = currTopic

                if mode == 'test':
                    if finalTopic == self.mappingDict[dirTopic]:
                        accurateCount += 1
                    confusionMatrix[self.mappingDict[dirTopic]][finalTopic] += 1

                elif mode == 'train':
                    self.afterLabellingD[finalTopic].append([fileName[0]])

        if mode == 'test':
            print " ".center(12, ' ') + "     " + "   ".join(str(currDirRow).center(12, ' ') for currDirRow in self.topicL)
            print "".center(315, " ")
            for dirTopic in self.topicL:
                print str(dirTopic).center(12, ' ') + "     " + "   "\
                .join( str(confusionMatrix[self.mappingDict[dirTopic]][currDirCol]).center(12, ' ')\
                for currDirCol in self.topicL)
                print "".center(315, " ")

            print "\n"
            print "Accurate classified files:",accurateCount 
            print "Total files in corpus:",totalFiles
            print "Accuracy is:", ( accurateCount / float(totalFiles) ) * 100

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
    print "Training complete.", modelFile ,"created!!"
    print "Time taken for training",(etime - stime) / 60, "minutes."
else:
    stime = time.time()
    topicObj = pickle.load(open(modelFile, "rb"))
    topicObj.createFileListing(directory,1.0,mode)
    topicObj.classify(directory)
    etime = time.time()
    print "Time taken for testing",(etime - stime) / 60, "minutes."

sys.exit(0)
