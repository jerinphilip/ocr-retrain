import os
import json
from pprint import pprint
from parser import webtotrain 

# Cost of correction, in seconds, per word
cost = {
    "not_gt_uncorrectable": 16,
    "gt_correctable": 5,
    "gt_uncorrectable": 5,
    "not_gt_corerctable": 5
}

# Cost of human per day
human_cost = 350 # Per  working day
working_hours = 6 # Hours working a day
human_cost_per_hour = human_cost/working_hours

def analyze(save):
    new = {
            "not_gt_uncorrectable": save["uncorrectable"] - \
                    save["ocr_equals_gt"]["uncorrectable"] ,
            "gt_correctable" : save["ocr_equals_gt"]["correctable"],
            "not_gt_correctable": save["correctable"] - save["ocr_equals_gt"]["correctable"],
            "gt_uncorrectable": save["ocr_equals_gt"]["uncorrectable"]
    }

    total_cost = 0
    for key in cost.keys():
        total_cost += cost[key] * new[key]
    return total_cost


def final_error(save):
    return save["real_word_error"]/save["total"]

def tohours(seconds):
    return seconds/3600;

def to_money_units(hours):
    return human_cost_per_hour * hours


fmap = {}
fmap['hi'] = dict(map(lambda x: x.strip().split('_'), open("hi.fmap")))
fmap['ml'] = dict(map(lambda x: x.strip().split('_'), open("ml.fmap")))

def extract_bcode(save):
    bname = save["book_dir"].split('/')[-2]
    return bname

def get_total_words(save):
    td = save["0.0"]["unseen"]
    return td["total"]

print('Lang,Book,Cost,Error,Word')
for lang in ['hi', 'ml']:
    saves = []
    for dr, drs, fls in os.walk('output/%s'%(lang)):
        for fn in fls:
            fn_with_path = dr + '/' + fn
            saves.append(json.load(open(fn_with_path)))

    words = list(map(get_total_words, saves))

    pages = list(map(lambda x: len(webtotrain.read_book(x["book_dir"])), saves))
    bcodes = list(map(extract_bcode, saves))
    bnames = list(map(lambda bc: fmap[lang][bc], bcodes))
    costs_in_seconds = map(lambda x: analyze(x["0.0"]["unseen"]), saves)
    errors = map(lambda x: final_error(x["0.0"]["unseen"]), saves)
    #avg_error = sum(errors)/len(list(errors)
    costs_in_hours = map(tohours, costs_in_seconds)
    costs_in_cash = map(to_money_units, costs_in_hours)
    pretty_print = lambda name, cost, error, word, page: '%s,%s,%.2lf,%.2lf,%d,%d'%(lang, name, cost, error, word, page)
    str_ls = map(pretty_print, bnames, costs_in_cash, errors, words, pages)
    print('\n'.join(str_ls))


