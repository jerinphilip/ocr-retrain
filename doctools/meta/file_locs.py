import pickle
import json
import os

path = {
		"pickle": "/data5/deepayan/new_ocr/doctools/outdir/pickled/",
		"json":{
				"words": "/data5/deepayan/new_ocr/doctools/outdir/jsons_word/",
				"images": "/data5/deepayan/new_ocr/doctools/outdir/jsons_feat/",
				"combined": "/data5/deepayan/new_ocr/doctools/outdir/jsons/"
		}
}

def get_pickeled(book_name, **kwargs):
	flag = kwargs["type"]
	edges_feat = None
	if flag == "predictions":
		if os.path.exists(os.path.join(path["pickle"],'%s.pkl'%book_name)):
			print('Loading predictions...')
			with open(os.path.join(path["pickle"],'%s.pkl'%book_name), 'rb') as f:
				predictions = pickle.load(f)
			return(predictions)
		else:
			print("predictions do not exist...")
	if flag == "edges":
		if os.path.exists(os.path.join(path["pickle"],'%s.features_cluster.pkl'%book_name)):
			print('Loading Edges...')
			with open(os.path.join(path["pickle"],'%s.features_cluster.pkl'%book_name), 'rb') as f:
				edges_feat = pickle.load(f)
		return(edges_feat)
		

def get_clusters(book_name, **kwargs):
	features = kwargs["features"]
	try:
		with open(os.path.join(path["json"]["%s"%features], '%s.json'%book_name)) as json_data:
			data = json.load(json_data)
	except TypeError:
		print('%s/%s.json'%(path,book))
	return(data)

