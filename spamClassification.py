import os

class spamClassification:
	def __init__(self):
		self.priorSpam = 0
        self.priorNSpam = 0
	
	def createDictionary(self):
		countSpam,countNSpam = 0,0
		dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
		dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
		notSpamDict = {}
		spamDict = {}
		mailList= []
		
		for fileName in dirNSpam:
			countNSpam += 1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			for eachWord in mail:
				if (notSpamDict.get(eachWord) != None):
					notSpamDict[eachWord] += 1
				else:
					notSpamDict[eachWord] = 1

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
					spamDict[eachWord] += 1
				else:
					spamDict[eachWord] = 1
		self.priorSpam = countSpam/(countSpam+countNSpam)
		self.priorNSpam = countNSpam/(countSpam+countNSpam)			
		print("prior Spam",self.priorSpam)
		print("prior Non Spam",self.priorNSpam)
		
spamObj = spamClassification()
spamObj.createDictionary()