import os
from ocr import GravesOCR
import numpy as np
import parser.webtotrain as webtotrain
from error_module.dictionary import Dictionary
from Levenshtein import distance
import json
import sys
import re

def split_index(data,pages_to_include):
	si=0
	for i in range(pages_to_include):
		si+=len(data[i][0])
	return si

def split_train(data,fr):
	total=len(data)
	train_pages = int(np.ceil(fr*total))
	val_pages = total-train_pages
	return(train_pages,val_pages)

def evaluate_validataion(ocr,em,validation_data):
	print("evaluating.....")
	edit_dist,ler=0,0.0
	images,truths=[],[]
	for im,tr in validation_data:
		images.extend(im)
		truths.extend(tr)
	total_chars = ocr.no_of_characters(truths) #gets the total no. of characters from all the ground truths.
	predictions = [ocr.recognize(image) for image in images]
	errors = [em.error(prediction) for prediction in predictions]
	threshold = 0
	for i in range(len(truths)):
	  	truth, prediction, error = truths[i], predictions[i], errors[i]
	  	if error > threshold:
	  		edit_dist+=distance(truth, prediction) 
	ler = (edit_dist/total_chars)*100
	return ler
           
def retrain(ocr,book_path,lookup,book_index):
	fraction =0.8
	epochs=2 
	pagewise= webtotrain.read_book(book_path)
	pagewise=pagewise[:20]
	train,val=split_train(pagewise,fraction) #splits the data set into training and validation, with respect to a given fraction.
	train_data=pagewise[:train]
	validation_data=pagewise[train:]
	images,truths=[],[]
	for im,tr in pagewise:
		images.extend(im)																											
		truths.extend(tr)
	batch_size=2
	print("Training.....")
	c=0
	for pages_to_include in range(2,len(train_data),batch_size):
		c+=1
		si = split_index(train_data,pages_to_include)
		print("Number of sequences: "+str(si))
		ctcML_errors = ocr.train2(images[0:si],truths[0:si],epochs)
		print("exporting OCR model for pages: %d"%pages_to_include)
		new_weights = ocr.export()
		with open("xmls/nn.xml",'w') as in_file:
			in_file.write(new_weights)
		'''new_ocr = GravesOCR('xmls/nn.xml',lookup)
		error = Dictionary(**config["error"])
		val_ler = evaluate_validataion(new_ocr,error,validation_data) #we evaluate the label error rate on validation dataset using the new OCR model
		print(val_ler)
		with open("log_file_%s.out"%book_index,'a') as in_file:
			in_file.write('Pages: '+str(pages_to_include)+'\n'+"Validation Label Error Rate: "+str(val_ler)+'\n')'''
		
if __name__ == '__main__':
	config = json.load(open(sys.argv[1]))
	book_index = int(sys.argv[2])
	ocr = GravesOCR(config["model"], config["lookup"])
	book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
	#new_model,validation_data,training_error=retrain(ocr,book_locs[book_index])
	retrain(ocr,book_locs[book_index],config["lookup"],book_index)
	
	
	