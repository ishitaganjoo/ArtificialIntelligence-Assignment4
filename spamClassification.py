import os
from _collections import defaultdict

class spamClassification:
	def __init__(self):
		self.priorSpam = 0.0
        	self.priorNSpam = 0.0
        	self.sortedSpamList =[]
        	self.sortedNSpamList =[]
        	self.spamMostFreq = []
        	self.nonSpamMostFreq = []
	
	def createDictionary(self):
		countSpam,countNSpam = 0.0,0.0
		priorSpam,priorNSpam= 0.0,0.0
		dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
		dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
		notSpamDict = {}
		spamDict = {}
		
		mailList= []
		
		allWords = defaultdict(int)
				
		for fileName in dirNSpam:
			countNSpam += 1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			for eachWord in mail:
				if (notSpamDict.get(eachWord) != None):
					notSpamDict[eachWord] +=1
				else:
					notSpamDict[eachWord] = 1
				allWords[eachWord] += 1
			

		mailList = []
		for fileName in dirSpam:
			countSpam += 1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			for eachWord in mail:
				if(spamDict.get(eachWord) != None):
					spamDict[eachWord] +=1
				else:
					spamDict[eachWord] = 1
				allWords[eachWord] += 1

		self.priorSpam = countSpam/(countSpam+countNSpam)
		self.priorNSpam = countNSpam/(countSpam+countNSpam)	
		#print("prior Spam",self.priorSpam)
		#print("prior Non Spam",self.priorNSpam)
		
		#pick top 50 words from spamDict and non spam Dict
		self.sortedSpamList = sorted(spamDict, key=spamDict.get)
		#print "sorted list is",self.sortedSpamList
		
		self.sortedNSpamList = sorted(notSpamDict, key=notSpamDict.get)
		#print "sorted list is",self.sortedNSpamList

		print "ANAND",spamDict["the"],notSpamDict["the"],allWords["the"]
		#combine and select top 50 in both, make a table with rows as docs and columns as words.
		for i in range(len(self.sortedSpamList)-1, len(self.sortedSpamList)-49, -1):
			self.spamMostFreq.append(self.sortedSpamList[i])
			
		for i in range(len(self.sortedNSpamList)-1, len(self.sortedNSpamList)-49, -1):
			self.nonSpamMostFreq.append(self.sortedNSpamList[i])
		#print(notSpamDict)
		#self.calculateNaiveBayes(spamDict, notSpamDict, priorSpam, priorNSpam, countSpam, countNSpam)

	def calculateNaiveBayes(self, spamDict, notSpamDict, priorSpam, priorNSpam, countSpam, countNSpam):
		mailList = []
		wordList = []
		probSpam = 1.0
		dirTestSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
		for fileName in dirTestSpam:
			wordList= []
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
				for eachWord in mailList[0]:
					wordList.append(eachWord)
				for eachWord in wordList:
					if(spamDict.get(eachWord) != None):
						value = spamDict[eachWord]
						print(float(value[1]))
						print(float(countSpam))
						print(float(value[1])/float(countSpam))
						probSpam = float(probSpam) * (float(value[1])/float(countSpam))
						print("inside", probSpam)
					else:
						probSpam *= 1e-2
				probSpam *= priorNSpam
	
	
	def decisionTree(self):
		#
		
		print "in decision tree"	
spamObj = spamClassification()
spamObj.createDictionary()
#spamObj.calculateNaiveBayes()
