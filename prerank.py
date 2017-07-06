#coding=UTF-8
import operator
import json
candidates = []
word_map = json.load('wordmap.txt')
def compute_prob(history,next_see):
    '''
    history = [v1,v2,....]
    next_see = v_k
    '''
    transformed = [word_map[x] for x in history if x in word_map]
    next_see = word_map[next_see]

    pass

def find_candidate(history):
    x_score = {}
    for x in candidates:
        p = compute_prob(history,x)
        x_score[x] = p
    sorted_res = sorted(x_score.items(),key=operator.itemgetter(1),reverse=True)
    top_res = sorted_res[:100]
    return top_res
