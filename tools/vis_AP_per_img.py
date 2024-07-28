import sys
from matplotlib import pyplot as plt
import numpy as np
log_path = sys.argv[1]
with open(log_path) as f:
    log_content = f.readlines()

AP_line = [l for l in log_content if 'AP' in l and 'mean' not in l]

AP = [float(l.split(' ')[12][:-1]) for l in AP_line]
Recall = [float(l.split(' ')[14][:-1]) for l in AP_line]
num_data = len(AP)
plt.figure()
plt.subplot(1,2,1)
plt.hist(AP)
plt.subplot(1,2,2)
mean_data = []
step = 100
for i in range(num_data//step):
    mean_data.append( sum(AP[i*step:(i+1)*step])/step)
plt.bar(range(len(mean_data)),mean_data)
plt.savefig('vis_output.jpg')