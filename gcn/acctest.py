import numpy as np

gcn_ori = [0.81270, 0.81120, 0.81190, 0.81140, 0.81420, 0.81320, 0.81450, 0.81110, 0.80850, 0.80940]
baye = [0.81190, 0.81000, 0.80970, 0.81060, 0.81160, 0.81110, 0.81320, 0.80700, 0.80910, 0.80840]
# 1000

# 100

gcn_ori = [0.81360, 0.81330, 0.81400, 0.81540, 0.81730, 0.81740, 0.82330, 0.80950, 0.80350, 0.80790]
baye = [0.80990, 0.81000, 0.81080, 0.81270, 0.81310, 0.81300, 0.81900, 0.80640, 0.80430]

# 1

# 50 parameter, 500 bayes

print("Random accuracy=", "{:.5f}".format(np.mean(gcn_ori)), "Random std=", "{:.5f}".format(np.std(gcn_ori)))
print("Random accuracy=", "{:.5f}".format(np.mean(baye)), "Random std=", "{:.5f}".format(np.std(baye)))