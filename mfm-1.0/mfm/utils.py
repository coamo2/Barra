# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:33:25 2019

@author: asus
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



def Newey_West(ret, q = 2, tao = 252):
    '''
    Newey_West方差调整
    时序上存在相关性时，使用Newey_West调整协方差估计
    factor_ret: DataFrame, 行为时间，列为因子收益
    q: 假设因子收益为q阶MA过程
    tao: 算协方差时的半衰期
    '''
    from functools import reduce
    from statsmodels.stats.weightstats import DescrStatsW 
    
    T = ret.shape[0]           #时序长度
    K = ret.shape[1]           #因子数
    if T <= q or T <= K:
        raise Exception("T <= q or T <= K")
         
    names = ret.columns    
    weights = 0.5**(np.arange(T-1,-1,-1)/tao)   #指数衰减权重
    weights = weights / sum(weights)
    
    w_stats = DescrStatsW(ret, weights)
    ret = ret - w_stats.mean
    
    ret = np.matrix(ret.values)
    Gamma0 = [weights[t] * ret[t].T  @ ret[t] for t in range(T)]
    Gamma0 = reduce(np.add, Gamma0)
    
    
    V = Gamma0             #调整后的协方差矩阵
    for i in range(1,q+1):
        Gammai = [weights[i+t] * ret[t].T  @ ret[i+t] for t in range(T-i)]
        Gammai = reduce(np.add, Gammai)
        V = V + (1 - i/(1+q)) * (Gammai + Gammai.T)
    
    return(pd.DataFrame(V, columns = names, index = names))
    
    
    

def eigen_risk_adj(covmat, T = 1000, M = 100, scale_coef = 1.4):
    '''
    Eigenfactor Risk Adjustment
    T: 序列长度
    M: 模拟次数
    scale_coef: scale coefficient for bias
    '''
    F0 = covmat
    K = covmat.shape[0]
    D0,U0 = np.linalg.eig(F0)      #特征值分解; D0是特征因子组合的方差; U0是特征因子组合中各因子权重; F0是因子协方差方差
    #F0 = U0 @ D0 @ U0.T    D0 = U0.T @ F0 @ U0  
    
    if not all(D0>=0):         #检验正定性
        raise('covariance is not symmetric positive-semidefinite')
   
    v = []  #bias
    for m in range(M):
        ## 模拟因子协方差矩阵
        np.random.seed(m+1)
        bm = np.random.multivariate_normal(mean = K*[0], cov = np.diag(D0), size = T).T  #特征因子组合的收益
        fm = U0 @ bm       #反变换得到各个因子的收益
        Fm = np.cov(fm)    #模拟得到的因子协方差矩阵

        ##对模拟的因子协方差矩阵进行特征分解
        Dm,Um = np.linalg.eig(Fm)   # Um.T @ Fm @ Um 
    
        ##替换Fm为F0
        Dm_hat = Um.T @ F0 @ Um 

        v.append(np.diagonal(Dm_hat) / Dm)

    v = np.sqrt(np.mean(np.array(v), axis = 0))
    v = scale_coef * (v-1) + 1
    
    
    D0_hat = np.diag(v**2) * np.diag(D0)  #调整对角线
    F0_hat = U0 @ D0_hat @ U0.T           #调整后的因子协方差矩阵
    return(pd.DataFrame(F0_hat, columns = covmat.columns, index = covmat.columns))

    


def eigenfactor_bias_stat(cov, ret, predlen = 1):
    '''
    计算特征因子组合的bias统计量
    '''
    #bias stat
    b = []
    for i in range(len(cov)-predlen):
        try:
            D, U = np.linalg.eig(cov[i])                              #特征分解, U的每一列就是特征因子组合的权重
            U = U / np.sum(U, axis = 0)                               #将权重标准化到1
            sigma = np.sqrt(predlen * np.diag(U.T @ cov[i] @ U))      #特征因子组合的波动率
            retlen = (ret.values[(i+1):(i+predlen+1)] + 1).prod(axis=0) - 1
            r = U.T @ retlen                                          #特征因子组合的收益率
            b.append(r / sigma)
        except:
            pass
    
    b = np.array(b)
    bias_stat = np.std(b, axis = 0)
    plt.plot(bias_stat)
    return(bias_stat)
    


 
    
def progressbar(cur, total, txt):
    '''
    显示进度条
    '''
    percent = '{:.2%}'.format(cur / total)
    print("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent) + txt, end = '')
    



def group_mean_std(x):
    '''
    计算一组的加权平均和波动率
    '''
    m =sum(x.volatility*x.capital) / sum(x.capital)
    s = np.sqrt(np.mean((x.volatility - m)**2))
    return([m, s])


def shrink(x, group_weight_mean, q):
    '''
    计算shrink估计量
    '''
    a = q * np.abs(x['volatility'] - group_weight_mean[x['group']][0])
    b =  group_weight_mean[x['group']][1]
    v = a / (a + b)     #收缩强度
    SH_est = v * group_weight_mean[x['group']][0] + (1-v) * np.abs(x['volatility'])    #贝叶斯收缩估计量
    return(SH_est)
    

def bayes_shrink(volatility, capital, ngroup = 10, q = 1):
    '''
    使用市值对特异性收益率波动率进行贝叶斯收缩，以保证波动率估计在样本外的持续性
    volatility: 波动率
    capital: 市值
    ngroup: 划分的组数
    q: shrinkage parameter
    '''
    group = pd.qcut(capital, ngroup).codes    #按照市值分为10组
    data = pd.DataFrame(np.array([volatility, capital, group]).T, columns = ['volatility', 'capital', 'group'])
    #分组计算加权平均
    grouped = data.groupby('group')
    group_weight_mean = grouped.apply(group_mean_std)
    
    SH_est = data.apply(shrink, axis = 1, args = (group_weight_mean, q))   #贝叶斯收缩估计量 
    return(SH_est.values)
