import sys
from collections import defaultdict
import math

METADATAS = []
LABEL = -1
LABEL_SET = set()
FEATURE_SET = set()
NUM_OF_FEATURES = 0
FEATURE_VALUE_DICT = defaultdict(set)
TR_DATA = []

class Node:
	def __init__(self, parent=None, data_list=[]):
		self.feature = -1
		self.parent = parent
		self.childs = dict()
		self.entropy = entropy(data_list)
		
		self.is_leaf = None
		self.decision = None  # is_leaf == True이면 접근할 변수
	
	def print(self):
		if self.is_leaf:
			print("<LEAF info>")
			print(f"\tdecision: {self.decision}")
		else:
			print("<BRANCH info>")
			print(f"\tfeature: {self.feature}")
			print(f"\tchilds: {self.childs}")
		print(f"\tentropy: {self.entropy}")
		print()

def entropy(data_list):
	# Label counting
	total = 0
	num_dict = defaultdict(int)
	for data in data_list:
		total += 1
		num_dict[data[LABEL]] += 1
	# Calculate entropy
	prob = list(map(lambda x: x/sum(num_dict.values()), num_dict.values()))
	return sum(map(lambda x: -x * math.log2(x), prob))


def gain_ratio(now: Node, childs: dict):
	
	num_total = sum(map(len, childs.values()))

	if len(childs) == 1:
		return 0
	gain = now.entropy - sum(map(lambda x: (len(x)/num_total)*entropy(x), childs.values()))
	splitinfo = sum(map(lambda x: -(len(x)/num_total)*math.log2(len(x)/num_total), childs.values()))
	return gain / splitinfo


def branch_by_feature(feature, data_list):
	# ret_childs = dict([(val, []) for val in list(FEATURE_VALUE_DICT[feature])])
	ret_childs = defaultdict(list)
	for d in data_list:
		ret_childs[d[feature]].append(d)
	return ret_childs


def select_a_feature(now: Node, feature_list, data_list):
	# TODO: ret_childs 우쨔지
	ret_feature = -1

	if len(feature_list) == 1:
		return feature_list[0]
	
	max_gain = -1
	for f in feature_list:
		childs = branch_by_feature(f, data_list)
		# valid_childs = dict(filter(lambda x: x[1], childs.items()))
		gain = gain_ratio(now, childs)
		if gain > max_gain:
			max_gain = gain
			ret_feature = f
	return ret_feature


def vote_majority(data_list) -> str:
	if not data_list:
		print("in vote_majority, arg is empty.\n")
		exit()

	ret_major_val = 0
	num_dict = defaultdict(int)
	for data in data_list:
		num_dict[data[LABEL]] += 1

	classes = list(num_dict.items())
	classes.sort(key=lambda x: -x[1])
	ret_major_val = classes[0][0]
	return ret_major_val
	

def make_tree_recursively(curr_node : Node, data_list, acc_features: list):
	
	# 0. No Data -> stop
	if not data_list:
		curr_node.is_leaf = True
		curr_node.decision = curr_node.parent.decision
		return
	
	# 1. Classification done -> stop
	if curr_node.entropy == 0:
		curr_node.is_leaf = True
		curr_node.decision = data_list[0][-1]
		return

	# 2. No feature -> stop
	if len(acc_features) == NUM_OF_FEATURES:
		curr_node.is_leaf = True
		curr_node.decision = vote_majority(data_list)
		return
	
	feature = select_a_feature(curr_node, list(FEATURE_SET - set(acc_features)), data_list)
	childs = branch_by_feature(feature, data_list)

	# 3. No branch -> stop
	if len(childs) == 1:
		curr_node.is_leaf = True
		curr_node.decision = vote_majority(data_list)
		return
	
	# 3. Recursion
	curr_node.is_leaf = False
	curr_node.feature = feature
	curr_node.decision = vote_majority(data_list)
	# 비어있는 child도 set으로 만들어주기
	for val in FEATURE_VALUE_DICT[feature]:
		if val not in childs:
			childs[val] = []
	for val, child_data in childs.items():
		child_node = Node(parent=curr_node, data_list=child_data)
		curr_node.childs[val] = child_node
		make_tree_recursively(child_node, child_data, acc_features + [feature])

def print_tree(tree: Node):
	
	lines = [[tree]]
	curr_line = lines[0]

	while curr_line:
		next_val = []
		next_line = []
		for node in curr_line:
			if node.is_leaf:
				next_val += [f"__{node.decision}__"]
			else:
				next_val += node.childs.keys()
				next_line += node.childs.values()
		lines += [[], next_val, next_line]
		curr_line = next_line
	for line in lines:
		# print(line)
		if line:
			if type(line[0]) == Node:
				print("\t".join(list(map(lambda x: METADATAS[x.feature], line))))
			elif type(line[0]) == str:
				print("\t\t".join(line))
			else:
				print("......d")
		else:
			print()

def classify(data):
	global decision_tree

	node = decision_tree
	# leaf까지 탐색
	while not node.is_leaf:
		node = node.childs[data[node.feature]]
	
	return node.decision
	
	
tr_data_file = sys.argv[1]
test_data_file = sys.argv[2]
output_file = sys.argv[3]

# 1st scan
with open(tr_data_file, "r") as f:
	# line 1
	METADATAS = f.readline().split()
	NUM_OF_FEATURES = len(METADATAS) - 1
	FEATURE_SET = set(range(NUM_OF_FEATURES))
	while True:
		tr_data = f.readline().split()
		if not tr_data:
			break
		for i in range(NUM_OF_FEATURES):
			FEATURE_VALUE_DICT[i].add(tr_data[i])
		TR_DATA.append(tr_data)
		LABEL_SET.add(tr_data[-1])


decision_tree = Node(data_list=TR_DATA)
make_tree_recursively(decision_tree, TR_DATA, [])

# print_tree(decision_tree)
with open(test_data_file, "r") as f_test, open(output_file, "w") as f_out:

	# line 1
	f_test.readline()
	f_out.write("\t".join(METADATAS) + "\n")

	# line 2~
	while True:
		test_data = f_test.readline().split()
		if not test_data:
			break
		test_data.append(classify(test_data))
		f_out.write("\t".join(test_data) + "\n")

