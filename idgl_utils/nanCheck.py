import numpy as np

feature_load = np.load('../data/feature_2160nodes.npy')
list_nan = []

for i in range(feature_load.shape[0]):
    node_feat = feature_load[i,:,:]
    tmp = 0
    if tmp == 0:
        for j in range(node_feat.shape[0]):
            if tmp == 0:
                for z in range(node_feat.shape[1]):
                    node_value = node_feat[j][z]
                    if np.isnan(node_value):
                        list_nan.append(i)
                        tmp = 1
                        break
            else:
                break

# 保存节点为txt
str = '\n'
f = open("node_repeat.txt","w")
f.write(str.join('%s' %id for id in list_nan))
f.close()

feature_final = feature_load
isLoad = True
for i in range(feature_load.shape[0]):
    if i in list_nan:
        print(i)
    else:
        node_feat_save = feature_load[i:i+1,:,:]
        if isLoad:
            feature_final = node_feat_save
            isLoad = False
        else:
            feature_final = np.concatenate((feature_final,node_feat_save),axis=0)

np.save('feature_2160nodes.npy',feature_final)
print()

