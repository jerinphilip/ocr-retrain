import pickle
import json
import os
import pdb
path = {
		"pickle": {
					"predictions":"/data5/deepayan/new_ocr/doctools/outdir/pickled/",
					"features": "/data5/deepayan/new_ocr/doctools/outdir/backup/"
					}
		"json":{
				"words": "/data5/deepayan/new_ocr/doctools/outdir/jsons_word/",
				"images": "/data5/deepayan/new_ocr/doctools/outdir/jsons_feat/",
				"combined": "/data5/deepayan/new_ocr/doctools/outdir/jsons/",
				"cost": "/data5/deepayan/new_ocr/doctools/outdir/jsons_cost/"
		}
}

def get_pickeled(book_name, **kwargs):
	flag = kwargs["type"]
	edges_feat, predictions = None, None
	if flag == "predictions":
		if os.path.exists(os.path.join(path["pickle"]["predictions"],'%s.pkl'%book_name)):
			print('Loading predictions...')
			with open(os.path.join(path["pickle"]["predictions"],'%s.pkl'%book_name), 'rb') as f:
				predictions = pickle.load(f)
			return(predictions)
		
	if flag == "edges":
		if os.path.exists(os.path.join(path["pickle"]["features"],'%s.features_cluster.pkl'%book_name)):
			print('Loading Edges...')
			with open(os.path.join(path["pickle"]["features"],'%s.features_cluster.pkl'%book_name), 'rb') as f:
				edges_feat = pickle.load(f)
		return(edges_feat)
	if flag == "word edges":
		if os.path.exists(os.path.join(path["pickle"],'%s.words_cluster.pkl'%book_name)):
			print('Loading Edges...')
			with open(os.path.join(path["pickle"],'%s.words_cluster.pkl'%book_name), 'rb') as f:
				edges_feat = pickle.load(f)
		return(edges_feat)
		

def get_clusters(book_name, **kwargs):
	features = kwargs["features"]
	data=None
	# pdb.set_trace()
	if os.path.exists(os.path.join(path["json"]["%s"%features], '%s.json'%book_name)):
		with open(os.path.join(path["json"]["%s"%features], '%s.json'%book_name)) as json_data:
			data = json.load(json_data)
	return data
	
		

