# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:39:01 2019

@author: asus
"""



import pandas as pd
import numpy as np
from mfm.CrossSection import CrossSection
from mfm.utils import Newey_West, progressbar, eigen_risk_adj, eigenfactor_bias_stat



class MFM():
    '''
    data: DataFrame
    column1: date
    colunm2: stocknames
    colunm3: capital
    column4: ret
    style_factors: DataFrame
    industry_factors: DataFrame
    '''
    
    def __init__(self, data, P, Q):
        self.Q = Q                                                           #风格因子数
        self.P = P                                                           #行业因子数
        self.dates = pd.to_datetime(data.date.values)                        #日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   #排序后的日期
        self.T = len(self.sorted_dates)                                      #期数
        self.data = data                                                     #数据
        self.columns = ['country']                                           #因子名
        self.columns.extend((list(data.columns[4:])))
        
        self.last_capital = None                                             #最后一期的市值 
        self.factor_ret = None                                               #因子收益
        self.specific_ret = None                                             #特异性收益
        self.R2 = None                                                       #R2
        
        self.Newey_West_cov = None                        #逐时间点进行Newey West调整后的因子协方差矩阵
        self.eigen_risk_adj_cov = None                    #逐时间点进行Eigenfactor Risk调整后的因子协方差矩阵
        self.vol_regime_adj_cov = None                    #逐时间点进行Volatility Regime调整后的因子协方差矩阵
    
    
    def reg_by_time(self):
        '''
        逐时间点进行横截面多因子回归
        '''
        factor_ret = []
        R2 = []
        specific_ret = []
        
        print('===================================逐时间点进行横截面多因子回归===================================')       
        for t in range(self.T):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            cs = CrossSection(data_by_time.iloc[:,:4], data_by_time.iloc[:,-self.Q:], data_by_time.iloc[:,4:(4+self.P)])
            factor_ret_t, specific_ret_t, _ , R2_t = cs.reg()
            
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
            R2.append(R2_t)
            self.last_capital = cs.capital
         
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        
        self.factor_ret = factor_ret                                               #因子收益
        self.specific_ret = specific_ret                                           #特异性收益
        self.R2 = R2                                                               #R2
        return((factor_ret, specific_ret, R2))



    def Newey_West_by_time(self, q = 2, tao = 252):
        '''
        逐时间点计算协方差并进行Newey West调整
        q: 假设因子收益为q阶MA过程
        tao: 算协方差时的半衰期
        '''
        
        if self.factor_ret is None:
            raise Exception('please run reg_by_time to get factor returns first')
            
        Newey_West_cov = []
        print('\n\n===================================逐时间点进行Newey West调整=================================')    
        for t in range(1,self.T+1):
            try:
                Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao))
            except:
                Newey_West_cov.append(pd.DataFrame())
            
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.Newey_West_cov = Newey_West_cov
        return(Newey_West_cov)
    
    
    
    def eigen_risk_adj_by_time(self, M = 100, scale_coef = 1.4):
        '''
        逐时间点进行Eigenfactor Risk Adjustment
        M: 模拟次数
        scale_coef: scale coefficient for bias
        '''
        
        if self.Newey_West_cov is None:
            raise Exception('please run Newey_West_by_time to get factor return covariances after Newey West adjustment first')        
        
        eigen_risk_adj_cov = []
        print('\n\n===================================逐时间点进行Eigenfactor Risk调整=================================')    
        for t in range(self.T):
            try:
                eigen_risk_adj_cov.append(eigen_risk_adj(self.Newey_West_cov[t], self.T, M, scale_coef))
            except:
                eigen_risk_adj_cov.append(pd.DataFrame())
            
            progressbar(t+1, self.T, '   date: ' + str(self.sorted_dates[t])[:10])
        
        self.eigen_risk_adj_cov = eigen_risk_adj_cov
        return(eigen_risk_adj_cov)
        
        
    
    def vol_regime_adj_by_time(self, tao = 84):
        '''
        Volatility Regime Adjustment
        tao: Volatility Regime Adjustment的半衰期
        '''
        
        if self.eigen_risk_adj_cov is None:
            raise Exception('please run eigen_risk_adj_by_time to get factor return covariances after eigenfactor risk adjustment first')        
        
        
        K = len(self.eigen_risk_adj_cov[-1])
        factor_var = list()
        for t in range(self.T):
            factor_var_i = np.diag(self.eigen_risk_adj_cov[t])
            if len(factor_var_i)==0:
                factor_var_i = np.array(K*[np.nan])
            factor_var.append(factor_var_i)
         
        factor_var = np.array(factor_var)
        B = np.sqrt(np.mean(self.factor_ret**2 / factor_var, axis = 1))      #截面上的bias统计量
        weights = 0.5**(np.arange(self.T-1,-1,-1)/tao)                            #指数衰减权重
        
        
        lamb = []
        vol_regime_adj_cov = []
        print('\n\n==================================逐时间点进行Volatility Regime调整================================') 
        for t in range(1, self.T+1):
            #取除无效的行
            okidx = pd.isna(factor_var[:i]).sum(axis = 1) == 0 
            okweights = weights[:t][okidx] / sum(weights[:t][okidx])
            fvm = np.sqrt(sum(okweights * B.values[:t][okidx]**2))   #factor volatility multiplier
            
            lamb.append(fvm)  
            vol_regime_adj_cov.append(self.eigen_risk_adj_cov[t-1] * fvm**2)
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.vol_regime_adj_cov = vol_regime_adj_cov
        return((vol_regime_adj_cov, lamb))

    



if __name__ == '__main__':

    import os
    os.chdir('C:\\Users\\asus\\Desktop\\MFM')


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
    vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 42)               #vol regime调整


    eigenfactor_bias_stat(nw_cov_ls[1000:], factor_ret[1000:], predlen = 21)    #特征风险调整前特征因子组合的bias统计量
    eigenfactor_bias_stat(er_cov_ls[1000:], factor_ret[1000:], predlen = 21)      #特征风险调整后特征因子组合的bias统计量

