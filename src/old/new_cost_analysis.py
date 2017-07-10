import os
import json
from pprint import pprint
from parser import webtotrain 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

all_unseen = "0"

# Cost of correction, in seconds, per word
cost = {
    "not_gt_uncorrectable": 16,
    "gt_correctable": 5,
    "gt_uncorrectable": 5,
    "not_gt_correctable": 5
}

# Cost of human per day
human_cost = 350 # Per  working day
working_hours = 6 # Hours working a day
human_cost_per_hour = human_cost/working_hours

def cost_seconds(save):
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
    td = save[all_unseen]["unincluded"]
    return td["total"]

def get_xs_ys(save):
    page_count = int(save["pages"])
    batchSize = 10
    results = []
    review_cost = 0
    for pages_included in range(0, page_count, batchSize):
        key = str(pages_included)
        pages_unincluded = page_count - pages_included
        """
        total_cost_seconds = cost_seconds(save[key]["unincluded"]) + review_cost
                #+ cost_seconds(save[key]["included"]) # Part of review

        #total_cost_rupees = to_money_units(tohours(total_cost_seconds))
        cost_per_page = total_seconds/pages_unincluded
        review_cost += cost_per_page * 10
        """
        cost_seconds_per_page= cost_seconds(save[key]["unincluded"])/pages_unincluded
        total_cost_seconds = review_cost + cost_seconds_per_page*pages_unincluded
        total_cost_rupees = to_money_units(tohours(total_cost_seconds))
        results.append((pages_included,total_cost_rupees))
        review_cost += cost_seconds_per_page * 10
    return list(zip(*results))

print('Lang,Book,Cost,Error,Word')
for lang in ['hi', 'ml']:
    saves = []
    for dr, drs, fls in os.walk('output/%s'%(lang)):
        for fn in fls:
            fn_with_path = dr + '/' + fn
            saves.append(json.load(open(fn_with_path)))

    # words = list(map(get_total_words, saves))
    plt.figure(figsize=(20,10))
    for save in saves:
        xs, ys = get_xs_ys(save)
        bname = fmap[lang][extract_bcode(save)]
        plt.plot(xs, ys, label=bname)
        label_str = "%s, %s"%(bname[:5], save["pages"])
        plt.text(xs[-1], ys[-1], label_str)

    plt.xlabel("no of pages included in dictionary")
    plt.ylabel("estimated cost for entire book")
    plt.savefig('output/images/cost-projected-%s.png'%(lang), dpi=300)
    plt.clf()


