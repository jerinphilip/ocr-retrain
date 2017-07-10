import os
from webtotrain import webtotrain


def page_to_unit(pagewise):
	images, truths = [], []
	for im, tr in pagewise:
		images.extends(im)
		truths.extends(tr)
	#units = tuple(zip(images, truths)) 
	return (images,truths) #(list of images, list of truths)
