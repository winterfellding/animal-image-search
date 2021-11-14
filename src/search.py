import gen_vector as gen
import numpy as np
import json
import collections

with open('vec_data.txt') as f:
    all_vecdata = json.loads(f.read())

def l2_dist(vec1, vec2):
    np1 = np.array(vec1)
    np2 = np.array(vec2)
    return np.sum(np.power(np1 - np2, 2))

def search(img):
    vector = gen.gen_vector(img)
    dist_to_key = {}
    for k, v in all_vecdata.items():
        dist = l2_dist(vector, v)
        dist_to_key[dist] = k
    od = collections.OrderedDict(sorted(dist_to_key.items()))
    return list(od.items())[:10]
    
if __name__ == '__main__':
    print(search('/Users/winterfall/Desktop/dog-test.jpeg'))