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

import math
d = 10
N = 10000
number_selection = [0] * d
sum_of_rewards = [0] * d
ads_selected = []
for n in range(N):
    max_upr_bound = 0
    ad = 0
    for i in range(d):
        # for first 10 round we want to select the nth ad so,
        # for n = 0 at iter i = 0 , num_selection[0] will be 0 thus upper bound will set to
        # 10^400 and add 0 will be selected . At next step ad0 will be calculated normally
        # but ad1 will get 10^ 400 thus giving it the chance.
        if number_selection[i] > 0:
            average_reward = sum_of_rewards[i] / number_selection[i]
            delta_sqrt = math.sqrt(3/2 * math.log(n+1) / number_selection[i])
            upper_bound = average_reward + delta_sqrt
        else:
            upper_bound = 1e400
        if max_upr_bound < upper_bound:
            max_upr_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_selection[ad] += 1
    sum_of_rewards[ad] += df.values[n,ad]

total_reward = sum(sum_of_rewards)

plt.hist(ads_selected)
plt.show()