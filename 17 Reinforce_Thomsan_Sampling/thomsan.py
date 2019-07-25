import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Ads_CTR_Optimisation.csv')

# N = 10000
# d = 10
# ad_selected = []
# toral_rwd = 0
# for n in range(0,N):
#     ad = random.randrange(d)
#     ad_selected.append(ad)
#     rwd = df.values[n,ad]
#     toral_rwd += rwd
#
# plt.hist(ad_selected)
# plt.show()

import math,random
d = 10
N = 10000
number_of_reward_1 = [0]*d
number_of_reward_0 = [0]*d
ads_selected = []
for n in range(N):
    max_random = 0
    ad = 0
    for i in range(d):
        random_beta = random.betavariate(number_of_reward_1[i] + 1, number_of_reward_0[i] + 1)
        if max_random < random_beta:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    if df.values[n,ad] == 0:
        number_of_reward_0[ad] += 1
    else:
        number_of_reward_1[ad] += 1

total_reward = sum(number_of_reward_1)

plt.hist(ads_selected)
plt.show()