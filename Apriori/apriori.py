import sys
from itertools import combinations
import time

start_time = time.time()
# args들을 변수에 저장
min_sup = int(sys.argv[1]) / 100
input_file = sys.argv[2]
output_file = sys.argv[3]

db = []
with open(input_file, "r") as f:
    while True:
        line = f.readline()
        if not line: break
        db.append(tuple(map(int, line.split())))
db_len = len(db)

def count_sup(tup_list) -> dict :
    freq_dict = {}
    for tup in tup_list:
        sorted_tup = make_key(tup)
        for trx in db:
            if set(trx).issuperset(set(sorted_tup)):
                if sorted_tup in freq_dict:
                    freq_dict[sorted_tup] += 1
                else:
                    freq_dict[sorted_tup] = 1
    sup_dict = dict(map(lambda x :(x[0], x[1]/db_len), freq_dict.items()))
    return sup_dict

def check_min_sup(sup_dict: dict) -> dict:
    return dict(filter(lambda x : x[1] >= min_sup, sup_dict.items()))

def make_key(iterable) :
    return tuple(sorted(iterable))

def make_fp_dict_list() -> list :
    fp_dict_list = [{}]         # k-th dict : dict(k길이 tuple : support)

    # 1. k=1, get 1-FP
    # Count sup
    freq1_dict = {}
    for trx in db:
        for item in trx:
            key_tup = tuple([item])
            if key_tup in freq1_dict:
                freq1_dict[key_tup] += 1
            else:
                freq1_dict[key_tup] = 1
    sup1_dict = dict(map(lambda x: (x[0], x[1]/db_len), freq1_dict.items()))
    # CheckFP
    fp_dict_list.append(check_min_sup(sup1_dict))

    # 2. repeat with index k
    # 2.1 k = 1
    k = 1
    candidates_tup_list = []
    for tup1, tup2 in combinations(fp_dict_list[1].keys(), 2):
        candidates_tup_list.append(make_key(tup1 + tup2))
    # Count sup
    sup_dict = count_sup(candidates_tup_list)
    # Check FP
    fp_dict_list.append(check_min_sup(sup_dict))

    # 2.2 k > 1
    k = 2
    while True:
        # 1) Generate (k+1)-candidates (self-joining)
        candidates_tup_list = []
        for tup1, tup2 in combinations(fp_dict_list[k].keys(), 2):
            uni_set = set(tup1 + tup2)
            if len(uni_set) == k + 1 and make_key(uni_set) not in candidates_tup_list:
                inter_set = set(tup1) & set(tup2)           # k - 1
                result = True
                for elem in inter_set:
                    if make_key(uni_set - set([elem])) not in fp_dict_list[k]:
                        result = False
                        break
                if result:
                    candidates_tup_list.append(make_key(uni_set))
                    
        if len(candidates_tup_list) == 0:
            break

        # 2) Test the candidates (pruning)
        # a. Count sup
        sup_dict = count_sup(candidates_tup_list)    
        # b. Check if FP
        fp_dict_list.append(check_min_sup(sup_dict))         # (k+1)-FP

        # 3) Terminate if (k+1)-FP is empty or no candidate
        if len(fp_dict_list[k+1]) == 0:
            break

        k += 1
    return fp_dict_list


# Apriori Algorithm
fp_dict_list = make_fp_dict_list()

# Associative item set
with open(output_file, "w") as f:
    for len_of_uniset in range(2, len(fp_dict_list)):
        for uniset_tup in fp_dict_list[len_of_uniset]:
            # itemset 조합
            for len_of_itemset in range(1, len_of_uniset):
                for itemset_tup in combinations(uniset_tup, len_of_itemset):
                    # assoset: 차집합
                    len_of_assoset = len_of_uniset - len_of_itemset
                    assoset_tup = tuple(sorted(set(uniset_tup) - set(itemset_tup)))
                    # confidence
                    sup_of_uniset = fp_dict_list[len_of_uniset][uniset_tup]
                    sup_of_itemset = fp_dict_list[len_of_itemset][make_key(itemset_tup)]
                    conf = sup_of_uniset / sup_of_itemset
                    if conf >= min_sup:
                        f.write("{" + ",".join(map(str, itemset_tup)) + "}\t")
                        f.write("{" + ",".join(map(str, assoset_tup)) + "}\t")
                        f.write(f"{(sup_of_uniset*100):.2f}\t{((sup_of_uniset/sup_of_itemset)*100):.2f}\n")
                    # print(f"{set(itemset_tup)}\t{set(assoset_tup)-set(itemset_tup)}\t{sup_of_assoset:.2f}\t{(sup_of_assoset/sup_of_itemset):.2f}")
                    
end_time = time.time()
print(end_time - start_time)