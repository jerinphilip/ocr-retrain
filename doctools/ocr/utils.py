from Levenshtein import distance

class Metrics:
	def __init__(self):
		print("Calculating Errors....")

	def cer(self, s1, s2):
		edit_dist = distance(s1, s2)
		return edit_dist

	def wer(self,s1,s2):
		var = 0
		if s1 != s2:
			var = 1
		return(var)
