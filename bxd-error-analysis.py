import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math

path = r"C:\Users\Mike.DESKTOP-CA70LTI\Google Drive\PhD\BXD\applications\EVB_HF\rxnSystemOnly\jacobian\20_friction\analysis\FPT_arrays"

allBoxes = pd.DataFrame()  # creates a new dataframe that's empty
mfpts_fwd = []
mfpts_bwd = []
std_fwd = []
std_bwd = []
for box in range(0, 11):
    box_forward = "{}to{}".format(box, box + 1)
    box_back = "{}to{}".format(box, box - 1)
    fwd_fpts = pd.read_csv(path + "\\" + box_forward + ".txt", header=None,
                           names=['FPT']).assign(Direction="Forward", Box=box)
    bkd_fpts = pd.read_csv(path + "\\" + box_back + ".txt", header=None,
                           names=['FPT']).assign(Direction="Backward", Box=box)

    mfpts_fwd.append(np.mean(fwd_fpts["FPT"]))
    mfpts_bwd.append(np.mean(bkd_fpts["FPT"]))
    std_fwd.append(np.std(fwd_fpts["FPT"]) / math.sqrt(float(len(fwd_fpts))))
    std_bwd.append(np.std(bkd_fpts["FPT"]) / math.sqrt(float(len(bkd_fpts))))
    allBoxes = pd.concat([allBoxes, fwd_fpts, bkd_fpts])

mdf = pd.melt(allBoxes, id_vars=['Box', 'Direction'], value_vars=['FPT'], value_name='FPT')
mdf.head()

ax = sns.boxplot(x="Box", y="FPT", hue="Direction", data=mdf)
ax.set_yscale('log')
plt.show()

#ax = sns.barplot(x="Box", y="FPT", hue="Direction", data=mdf)
#ax.set_yscale('log')
#plt.show()

# formula i calculated. 
# box_rates_var = [math.pow(m_b / s_b, 2) + (m_b * m_b * s_f * s_f / math.pow(m_f, 4)) for m_f, s_f, m_b, s_b in zip(mfpts_fwd, std_fwd, mfpts_bwd, std_bwd)]
# formula from web.
box_rates_var = [math.pow(m_b / m_f, 2) * (math.pow(s_f / m_f, 2) + math.pow(s_b / m_b, 2)) for m_f, s_f, m_b, s_b in
                 zip(mfpts_fwd, std_fwd, mfpts_bwd, std_bwd)]

box_rates_std = [math.sqrt(x) for x in box_rates_var]

box_rate = [m_b / m_f for m_f, m_b in zip(mfpts_fwd, mfpts_bwd)]

rates = pd.DataFrame({'Box': range(len(box_rate)), 'Rate': box_rate, 'Error': box_rates_std })

ax = sns.barplot(x="Box", y="Rate", data=rates)
plt.show()
