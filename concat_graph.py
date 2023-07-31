import csv

site_fp1 = './data/2160nodes.CSV'
site_fp = './data/2114nodes.CSV'
with open(site_fp, 'r') as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        num = 0
        idx = line[1]+line[2]
        with open(site_fp1, 'r') as f1:
            csv_reader1 = csv.reader(f1)
            for line1 in csv_reader1:
                idx1 = line1[1]+line1[2]
                if idx == idx1 :
                    # alt1 = float(line1[5])
                    # print(alt1)
                    num = 1
            if num == 0:
                # print("None",)
                print(idx)

