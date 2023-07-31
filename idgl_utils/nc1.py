from netCDF4 import Dataset
import numpy as np

list_nan = [0,2,3,598,77]
str = '\n'
f = open("node_repeat.txt","w")
f.write(str.join('%s' %id for id in list_nan))
f.close()
