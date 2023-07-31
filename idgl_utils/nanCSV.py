import os
import sys
import csv

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
site_fp = os.path.join(proj_dir, '../data/2048_nodes.CSV')

list_csv = []
with open(site_fp, 'r') as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        list_csv.append(line)

node_num_list = []
with open('node_repeat.txt','r') as txt:
    lines = txt.readlines()
    for line in lines:
        node_num = line.replace('\n','')
        node_num_list.append(node_num)

idx = 0
node_new_list = []
for i in range(len(list_csv)):
    node = list_csv[i]
    if node[0] not in node_num_list:
        idx0 = str(idx)
        idx = idx + 1
        node[0] = idx0
        node_new_list.append(node)

node_num = str(len(node_new_list))
document_name = node_num + "_nodes.CSV"
with open(document_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(node_new_list)

