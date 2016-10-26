# randomly set a number lower than 3000 to check if matched
randNum = 0

# load file of feature_name
featureName = []
f = open('./dat/selected_feature_2000-01-01_2016-12-31')
for num, l in enumerate(f):
    line = l.strip()
    featureName.append(line)

# load file of feature_label
f = open('./dat/feature_label_2000-01-01_2016-12-31')
for num, l in enumerate(f):
    if num != randNum: # randomly pick one to check if matched
        continue
    line = l.strip().split(',')
    ticker = line[0]
    values = line[1:]
    for i in range(len(featureName)):
        print ticker, featureName[i], values[i]




    

