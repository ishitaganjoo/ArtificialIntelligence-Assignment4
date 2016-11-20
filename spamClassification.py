import os

class spamClassification:
	countSpam,countNSpam = 0,0
	priorSpam,priorNSpam= 0,0
	def createDictionary(self):
		
		dirNSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam")
		dirSpam = os.listdir("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam")
		notSpamDict = {}
		spamDict = {}
		mailList= []
		
		for fileName in dirNSpam:
			countNSpam+=1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			for eachWord in mail:
				if (notSpamDict.get(eachWord) != None):
					notSpamDict[eachWord] = notSpamDict[eachWord] + 1
				else:
					notSpamDict[eachWord] = 1

		mailList = []
		for fileName in dirSpam:
			countSpam+=1
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				mailList.append(newStr.split(' '))
		for mail in mailList:
			for eachWord in mail:
				if(spamDict.get(eachWord) != None):
					spamDict[eachWord] = spamDict[eachWord] + 1
				else:
					spamDict[eachWord] = 1
		priorSpam = countSpam/(countSpam+countNSpam)
		priorNSpam = countNSpam/(countSpam+countNSpam)			
		#print(notSpamDict)
		#print(spamDict)
		print("prior Spam",priorSpam)
		print("prior Non Spam",priorNSpam)

spamObj = spamClassification()
spamObj.createDictionary()
