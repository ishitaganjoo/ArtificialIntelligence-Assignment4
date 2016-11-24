import os
from _collections import defaultdict
from collections import Counter
import math

class spamClassification:
	def __init__(self):
		self.priorSpam = 0.0
        	self.priorNSpam = 0.0
        	self.mostFreq = []
        	self.sortedList = []
        	self.documentSpamFreq = defaultdict(int)
		self.documentNSpamFreq = defaultdict(int)
		self.entropyDict = defaultdict(int)

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
			uniqueWords = []
			for eachWord in mail:
				lowerEachWord = eachWord.lower()
				if (notSpamDict.get(lowerEachWord) != None):
					notSpamDict[lowerEachWord] +=1
				else:
					notSpamDict[lowerEachWord] = 1
					
				if eachWord.isalpha():
					allWords[lowerEachWord] += 1
					
				if lowerEachWord not in uniqueWords and lowerEachWord.isalpha():
					uniqueWords.append(lowerEachWord)	
			while uniqueWords:
				self.documentSpamFreq[uniqueWords.pop()] += 1	
			

		mailList = []
		for fileName in dirSpam:
			countSpam += 1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			uniqueWords = []
			for eachWord in mail:
				lowerEachWord = eachWord.lower()
				if(spamDict.get(lowerEachWord) != None):
					spamDict[lowerEachWord] +=1
				else:
					spamDict[lowerEachWord] = 1
				if eachWord.isalpha():
					allWords[lowerEachWord] += 1
				if lowerEachWord not in uniqueWords and lowerEachWord.isalpha():
					uniqueWords.append(lowerEachWord)	
			while uniqueWords:
				self.documentNSpamFreq[uniqueWords.pop()] += 1	
				
		self.priorSpam = countSpam/(countSpam+countNSpam)
		self.priorNSpam = countNSpam/(countSpam+countNSpam)	
		
		#pick top 50 words from spamDict and non spam Dict
		self.sortedList = sorted(allWords, key=spamDict.get)
		#print "sorted list is", self.sortedList
		#combine and select top 50 in both, make a table with rows as docs and columns as words.
	
		#print dict(Counter(allWords).most_common(50))
		#print dict(Counter(allWords).most_common()[:-50:-1])
		#print "document freq is", self.documentFreq
		
		#get the keyset of allWords dict to access each word in the dataset
		words = allWords.keys()
		self.calculateEntropy(words,countNSpam+countSpam)
			
		#self.calculateNaiveBayes(spamDict, notSpamDict, priorSpam, priorNSpam, countSpam, countNSpam)

	def calculateNaiveBayes(self, spamDict, notSpamDict, priorSpam, priorNSpam, countSpam, countNSpam):
		mailList = []
		
		dirTestSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
		for fileName in dirTestSpam:
			mailList= []
			probSpam = 1.0
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
				for eachWord in mailList[0]:
					if(self.documentFreq.get(eachWord) != None):
						probSpam = float(probSpam) * (float(self.documentFreq[eachWord])/float(countSpam))
					else:
						probSpam *= 1e-2
				probSpam *= priorSpam
				
		dirTestNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam")
		for fileName in dirTestNSpam:
			mailList= []
			probNSpam = 1.0
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
				for eachWord in mailList[0]:
					if(self.documentFreq.get(eachWord) != None):
						probNSpam = float(probNSpam) * (float(self.documentFreq[eachWord])/float(countSpam))
					else:
						probNSpam *= 1e-2
				probNSpam *= priorNSpam
				
	def calculateEntropy(self,words,totalDocs):
		entropyDict = defaultdict(int)
		for word in words:
			numberSpam,numberNSpam,spamEntropy,nonSpamEntropy,entropyLeftSpam,entropyLeftNSpam = 0,0,0,0,0,0
			if self.documentSpamFreq.get(word) != None:
				numberSpam = self.documentSpamFreq[word]
			if self.documentNSpamFreq.get(word) != None:	
				numberNSpam = self.documentNSpamFreq[word]
			totalOccOfWord = numberSpam + numberNSpam
			totalNonOccOfWord = totalDocs - (totalOccOfWord)
			spamNotContainingWord = countSpam - numberSpam
			notSpamNotContainingWord = countNSpam - numberNSpam


			entropyBefore = -(float(countSpam/ totalDocs) * math.log(float(countSpam/ totalDocs))) - (float(countNSpam / totalDocs) * math.log(float(countNSpam/ totalDocs)))
			
			if numberSpam!=0:
				entropyLeftSpam = -(float(numberSpam / totalOccOfWord) * math.log(float(numberSpam / totalOccOfWord)))
			if numberNSpam!=0:
				entropyLeftNSpam = 	- (float(numberNSpam / totalOccOfWord) * math.log(float(numberNSpam / totalOccOfWord)))
			entropyLeft = entropyLeftSpam+entropyLeftNSpam
			
			if spamNotContainingWord!=0:
				entropyRightSpam = -(float(spamNotContainingWord / totalNonOccOfWord) * math.log(float(spamNotContainingWord / totalNonOccOfWord)))
			if notSpamNotContainingWord!=0:
				entropyRightNSpam = - (float(notSpamNotContainingWord / totalNonOccOfWord) * math.log(float(notSpamNotContainingWord / totalNonOccOfWord)))	
			entropyRight = entropyRightSpam+entropyRightNSpam
			
			entropyAfter = (float(totalOccOfWord / totalDocs)*entropyLeft) + (float(totalNonOccOfWord / totalDocs) * entropyRight)
			
			infoGain = entropyBefore - entropyAfter

			entropyDict[word] = infoGain
		
		print "entropy Dict is", self.entropyDict	
spamObj = spamClassification()
spamObj.createDictionary()
#spamObj.calculateNaiveBayes()
