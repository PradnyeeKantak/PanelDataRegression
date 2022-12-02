# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:57:05 2022

@author: PPK
"# -*- coding: utf-8 -*-
"""

import pandas as pd
import numpy as np
from scipy import stats

### For importing Fixed and Random Effects Regression:
from linearmodels.panel import PanelOLS
from linearmodels import RandomEffects
import linearmodels.iv.model as lm


### Extract the data from an Excel file:
panel = pd.read_excel('D:\IIM Lucknow\CIS\Code\Data\GNPA Ratio\Segmentation\PANEL_PSB.xlsx')
time = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
var_list = list(panel.columns)
bank_list = set(panel['BANK_NAME'])

del panel['YEAR']
del panel['BANK_NAME']
del var_list[0]
del var_list[-1]

panel_arr = panel.to_numpy()

### Transform 3D data to 2D
index = pd.MultiIndex.from_product([bank_list, time])
df = pd.DataFrame(panel_arr.reshape(len(bank_list)*len(time), len(var_list)),index=index, columns=var_list)

del var_list[0]

exp_var = df.filter([i for i in var_list])

### Fixed Effects Regression
mod = PanelOLS(df.GNPA, exp_var , entity_effects=True)
res_FE = mod.fit(cov_type='heteroskedastic')
print(res_FE)
print(type(res_FE))

### Random Effects Regression
mod = RandomEffects(df.GNPA, exp_var)
res_RE = mod.fit(cov_type='heteroskedastic')
print(res_RE)
print(type(res_RE))

### Hausman Test
cov_diff = res_FE.cov - res_RE.cov
est_diff = res_FE.params - res_RE.params
W = est_diff.dot(np.linalg.inv(cov_diff)).dot(est_diff)
dof = res_RE.params.size
pvalue = stats.chi2(dof).sf(W)
print("Hausman Test: chisq = {0}, df = {1}, p-value = {2}".format(W, dof, pvalue))
