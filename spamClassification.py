import os

class spamClassification:
	def createDictionary(self):
		path = "/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam"
		path2 = "/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam"
		dirs = os.listdir(path)
		dirs2 = os.listdir(path2)
		notSpamDict = {}
		spamDict = {}
		wordList= []

		for fileName in dirs:
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/notspam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				wordList.append(newStr.split(' '))
		for word in wordList:
			for eachWord in word:
				if (notSpamDict.get(eachWord) != None):
					value = notSpamDict[eachWord]
					value += 1
					notSpamDict[eachWord] = value
				else:
					notSpamDict[eachWord] = 1

		wordList = []
		for fileName in dirs2:
			with open("/u/bansalro/csci_b551_assignment_4/anahar-bansalro-iganjoo-a4/part1/part1/train/spam/"+fileName, "r") as f:
				content = f.readlines()
				newStr = "".join(content)
				wordList.append(newStr.split(' '))
		for word in wordList:
			for eachWord in word:
				if(spamDict.get(eachWord) != None):
					value = spamDict[eachWord]
					value += 1
					spamDict[eachWord] = value
				else:
					spamDict[eachWord] = 1
		print(notSpamDict)
		print(spamDict)

spamObj = spamClassification()
spamObj.createDictionary()
