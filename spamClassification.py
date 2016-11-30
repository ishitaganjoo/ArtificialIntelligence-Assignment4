import os
from _collections import defaultdict
from collections import Counter
import math
import string

class spamClassification:
	def __init__(self):
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

	def createDictionary(self):
		countSpam,countNSpam = 0.0,0.0
		priorSpam,priorNSpam= 0.0,0.0
		dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
		dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
		
		mailList= []
		stopWordList = ['from', 'to', 'a', 'an', 'subject', 'are', 'were', 'is', 'the']
		
		#allWords = defaultdict(int)
				
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
					
					if eachWord.isalpha():
						self.allWords[lowerEachWord] += 1
						wordsListPerMail.append(lowerEachWord)
					
					if lowerEachWord not in uniqueWords:
						uniqueWords.append(lowerEachWord)	
				while uniqueWords:
					newWord = uniqueWords.pop()
					self.documentNSpamFreq[newWord] += 1	
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
					if eachWord.isalpha():
						self.allWords[lowerEachWord] += 1
						wordsListPerMail.append(lowerEachWord)
					if lowerEachWord not in uniqueWords:
						uniqueWords.append(lowerEachWord)	
				while uniqueWords:
					newWord = uniqueWords.pop()
					self.documentSpamFreq[newWord] += 1	
				self.spamWordDocList.append(wordsListPerMail)
		self.priorSpam = countSpam/(countSpam+countNSpam)
		self.priorNSpam = countNSpam/(countSpam+countNSpam)	
		
		#combine and select top 50 in both, make a table with rows as docs and columns as words.
		
		#get the keyset of allWords dict to access each word in the dataset
		words = self.allWords.keys()
		#self.calculateEntropy(words,countNSpam+countSpam, countSpam, countNSpam)
			
		self.calculateNaiveBayes(self.spamDict, self.notSpamDict,  countSpam, countNSpam)

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
					#for binary
					if(self.documentSpamFreq.get(eachWord) != None):
						probSpam += math.log(float(self.documentSpamFreq[eachWord])/float(countSpam))
					else:
						probSpam += math.log(1e-9)
					if(self.documentNSpamFreq.get(eachWord) != None):
						probNSpam += math.log(float(self.documentNSpamFreq[eachWord]) / float(countNSpam))
					else:
						probNSpam += math.log(1e-9)
					if(self.spamDict.get(eachWord) != None):
					#print "sum is", sum(self.spamDict.values())
						probSpamCont += math.log(float(self.spamDict[eachWord] / float(spamDictSum)))
					else:
						probSpamCont += math.log(1e-9)
					if(self.notSpamDict.get(eachWord) != None):
						probNSpamCont += math.log(float(self.notSpamDict[eachWord] / float(notSpamDictSum)))
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
		for fileName in dirTestNSpam:
			mailList= []
			testDocs += 1
			probNSpam = 0.0
			probSpam = 0.0
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/test/notspam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
				for eachWord in mailList[0]:
					if(self.documentNSpamFreq.get(eachWord) != None):
						probNSpam += math.log(float(self.documentNSpamFreq[eachWord])/float(countSpam))
					else:
						probNSpam += math.log(1e-9)
					if(self.documentSpamFreq.get(eachWord) != None):
						probSpam += math.log(float(self.documentSpamFreq[eachWord]) / float(countNSpam))
					else:
						probSpam += math.log(1e-9)
				probNSpam += math.log(float(self.priorNSpam))
				probSpam += math.log(float(self.priorSpam))
			if(probNSpam > probSpam):
				accuracyCount += 1
		print("Accuracy non spam is", accuracyCount/testDocs)
				
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
spamObj = spamClassification()
spamObj.createDictionary()
#spamObj.calculateNaiveBayes()
