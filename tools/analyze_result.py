import sys
from matplotlib import pyplot as plt
import numpy as np
import json
from matplotlib import pyplot as plt
log_path = sys.argv[1]
with open(log_path) as f:
    js = json.load(f)

AP = [item["AP"] for item in js]
num_gt = [item["num_gt"] for item in js]
plt.gca().set_xlabel("#GT")
plt.gca().set_ylabel("AP")
plt.scatter(np.array(num_gt), np.array(AP))
plt.savefig("AP_gt.jpg")