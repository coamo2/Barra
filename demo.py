# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:49:20 2019

@author: asus
"""

    import os
    os.chdir('C:\\Users\\asus\\Desktop\\MFM')
    
    from mfm.MFM import MFM
    import pandas as pd 
    import numpy as np


    ####导入数据
    data = []
    for i in range(4):
        data.append(pd.read_csv('./data/Barra_data' + str(i+1) + '.csv'))

    data = pd.concat(data, axis = 0).iloc[:,1:]
    naidx = 1*np.sum(pd.isna(data), axis = 1)>0
    data = data[~naidx]
    data.index = range(len(data))


    ####行业数据
    industry_info = pd.read_csv(open('./data/industry_info.csv'))
    industry = np.array([1*(data.industry.values == x) for x in industry_info.code.values]).T
    industry = pd.DataFrame(industry, columns = list(industry_info.industry_names.values))
    data = pd.concat([data.iloc[:,:4], industry, data.iloc[:,5:]], axis = 1)


    model = MFM(data, 11, 10)
    (factor_ret, specific_ret, R2) = model.reg_by_time()
    nw_cov_ls = model.Newey_West_by_time(q = 2, tao = 252)                 #Newey_West调整
    er_cov_ls = model.eigen_risk_adj_by_time(M = 100, scale_coef = 1.4)    #特征风险调整
    vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 42) 