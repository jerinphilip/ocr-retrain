from matplotlib import pyplot as plt
import pandas as pd
import os
hi_fmap =  dict(map(lambda x: (x.split('_')),open('hi.fmap', 'r').readlines()))
ml_fmap =  dict(map(lambda x: (x.split('_')),open('ml.fmap', 'r').readlines()))

class PlotStats:
	def __init__(self, path, lang):
		self.path = path
		self.lang = lang
		print("Hello")

	def extract_name(self, filepath):
		book_id = filepath.split('_')[-1].split('.')[0]
		#print (book_id)
		if self.lang == "Hindi":
			my_dict = hi_fmap
		else:
			my_dict = ml_fmap
		
		#print(my_dict)
		return(my_dict[book_id])

	def x_y_axis(self, filepath):
		df = pd.read_csv(filepath)
		Pages = df["Pages"]
		ler = df["LabeErrorRate"]
		return ler, Pages

	def each_book(self):
		files = os.listdir(self.path)
		plt.figure(figsize=(20,10))
		for each_file in files:
			if each_file.endswith('.csv'):
				#print(each_file)
				filepath =os.path.join(self.path, each_file)
				book_name = self.extract_name(filepath)
				ler, Pages = self.x_y_axis(filepath)
				#print (ler)
				plt.plot(Pages, ler, label=book_name)
				#plt.text(ler.iloc[-1], Pages.iloc[-1], book_name)
				#plt.show()
		plt.xlabel("no of pages includes")
		plt.ylabel("label error rates")
		plt.xlim(xmax = 150, xmin = 10)
		plt.legend()
		plt.savefig('output/LabelErorRates-%s.png'%(self.lang), dpi=300)
		plt.clf()




if __name__ == '__main__':
	plot = PlotStats('new_ocr_stats/Malayalam','Malayalam')
	plot.each_book()



