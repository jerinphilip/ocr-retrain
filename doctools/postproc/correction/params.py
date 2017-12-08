from doctools.cluster.distance import jaccard, lev, euc, cos
from doctools.cluster.k_nearest.distance import normalized_euclid_norm  
params = {
    "type": 15,
    "verify": 5,
    "dropdown": 5,
    "ignore": 0,
    "cluster": 30,
    "selection": 1,
    "deselection": 5,
    
    
    "books":[
        "0365",
        "0002",
        "0003",
        "0005",
        "0006",
        "0008",
        "0009",
        "0018",
        "0028",
        "0072",
        "0075",
        "0078",
        "0109",
        "0128",
        "0135",
        "0188",
        "0230",
        "0231",
        "0235",
        "0236",
        "0237",
        "0238",
        "0239",
        "0240",
        "0241",
        "0242",
        "0243",
        "0244",
        "0245",
        "0246",
        "0365", 
        "0020", 
        "0033", 
        "0057", 
        "0142", 
        "0059", 
        "0040", 
        "0058", 
        "0039", 
        "0048", 
        "0037",  
        "0053", 
        "0024", 
        "0038", 
        "0014", 
        "0143",  
        "0025", 
        "0062", 
        "0366",  
        "0141", 
        "0049",  
        "0032", 
        "0036", 
        "0064", 
        "0367",
        "0022",  
        "0029", 
        "0040", 
        "0060", 
        "0061", 
        "0069", 
        "0191",   
        "0211",]

}

cluster_params =  {
        "words": {
            "distance" : lev,
            "threshold" : 0.5
        },
        "images": {
            "distance": normalized_euclid_norm,
            "threshold" : 0.36
        }
    }