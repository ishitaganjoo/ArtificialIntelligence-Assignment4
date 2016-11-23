import os
from _collections import defaultdict
from collections import Counter

class spamClassification:
	def __init__(self):
		self.priorSpam = 0.0
        self.priorNSpam = 0.0
        self.mostFreq = []
        self.sortedList = []

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
		
		#pick top 50 words from spamDict and non spam Dict
		self.sortedList = sorted(allWords, key=spamDict.get)
		print "sorted list is", self.sortedList
		#combine and select top 50 in both, make a table with rows as docs and columns as words.
		#for i in range(len(self.sortedList)-1, len(self.sortedList)-49, -1):
			#self.mostFreq.append(self.sortedList[i])
		print dict(Counter(allWords).most_common(5))
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
