import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
import sys
import math

eps = sys.float_info.epsilon  # 微小数

class Bayes_1d():
    '''
    1自由度バネマス系でベイズ推定を行うクラス
    '''
    def __init__(self, senarios, acc_end, dis_up, m=1e5, delta=1e-2):
        '''
        初期化

        Parameters
        ----------
        senarios : list
            パラメータのシナリオ
        acc_end : pandas.core.series.Series
            最下層の加速度（地震波の加速度）
        dis_up : pandas.core.series.Series
            上層の変位
        m : float
            真の質量
        delta : float
            時間幅
        dist_his : list
            パラメータの分布の推移
        '''
        self.delta2 = delta**2  # 計算ではΔtの2乗しか用いない
        self.m = m
        self.senarios = senarios
        self.acc_end = acc_end
        self.dis_up = dis_up
        self.dist_his = []
    
    def infer(self):
        '''
        タイムステップごとにベイズ推定を行う
        '''

        # パラメータの分布の推移
        self.dist_his.append([1/len(self.senarios)]*len(self.senarios))

        for i, (x, f) in tqdm(enumerate(zip(self.dis_up, self.acc_end))):
            if i > 1:
                x_next = x

                norm_const = 0  # 正規化定数を初期化
                prior = self.dist_his[-1]   # 事前分布
                posterior = np.array([])  # 事後分布の配列を初期化

                for j, k_conditioned in enumerate(self.senarios):
                    p_likelihood = self.likelihood(x_pre, x_cur, x_next, k_conditioned, f_cur)  # 尤度の計算
                    p = p_likelihood * prior[j]  # 尤度と事前確率の積
                    if math.isnan(p) or p < eps:
                        p = eps
                    norm_const += p
                    posterior = np.append(posterior, p)
                
                posterior /= norm_const  # 正規化

                self.dist_his.append(posterior)

                # 変位をずらす
                x_pre, x_cur = x_cur, x_next
                f_cur = f
            # 初めは変数に代入するだけ
            elif i == 0:
                x_pre = x
            elif i == 1:
                x_cur = x
                f_cur = f
            else:
                print('break')
                break

    def likelihood(self, x_pre, x_cur, x_next, k_conditioned, f):
        '''
        尤度関数

        Attributes
        ----------
        x_pre : float
            1ステップ前の変位
        x_cur : float
            現在の変位
        x_next : float
            1ステップ後の変位
        k_conditioned : float
            条件付けたkの平均値
        f : float
            入力された加速度
        '''
        k_std = k_conditioned/10  # 条件づけられたkの標準偏差
        mu = -x_cur/self.m*self.delta2*k_conditioned + 2*x_cur - x_pre - f*self.delta2  # 平均
        std = np.abs(self.delta2/self.m*x_cur*k_std) # 標準偏差
        p = st.norm.pdf(x_next, loc=mu, scale=std)
        return p



class Bayes_2d():
    '''
    2自由度バネマス系でベイズ推定を行うクラス
    '''
    def __init__(self, senarios, acc_end, dis_down, dis_up, m1=1e5, m2=8e4, delta=1e-2):
        '''
        初期化

        Parameters
        ----------
        senarios : list
            パラメータのシナリオ
        acc_end : pandas.core.series.Series
            最下層の加速度（地震波の加速度）
        dis_down : pandas.core.series.Series
            第1層の変位
        dis_up : pandas.core.series.Series
            第2層の変位
        m1, m2 : float
            真の質量
        delta : float
            時間幅
        dist_his : list
            パラメータの分布の推移
        '''
        self.delta2 = delta**2  # 計算ではΔtの2乗しか用いない
        self.m1 = m1
        self.m2 = m2
        self.senarios = senarios
        self.acc_end = acc_end
        self.dis_down = dis_down
        self.dis_up = dis_up
        self.dist_his = []
    
    def infer(self):
        '''
        タイムステップごとにベイズ推定を行う
        '''
        # パラメータの分布の推移
        self.dist_his.append([1/len(self.senarios)]*len(self.senarios))

        for i, (x1, x2, f) in tqdm(enumerate(zip(self.dis_down, self.dis_up, self.acc_end))):
            if i > 1:
                x1_next, x2_next = x1, x2

                norm_const = 0  # 正規化定数を初期化
                prior = self.dist_his[-1]   # 事前分布
                posterior = np.array([])  # 事後分布の配列を初期化

                for j, senario in enumerate(self.senarios):
                    k1_conditioned, k2_conditioned = senario  # パラメータをアンパック
                    p_likelihood = self.likelihood(x1_pre, x2_pre, x1_cur, x2_cur, x1_next, x2_next, k1_conditioned, k2_conditioned, f_cur)  # 尤度の計算
                    p = p_likelihood * prior[j]  # 尤度と事前確率の積
                    if math.isnan(p) or p < eps:
                        p = eps
                    norm_const += p
                    posterior = np.append(posterior, p)
                
                posterior /= norm_const  # 正規化

                self.dist_his.append(posterior)

                # 変位をずらす
                x1_pre, x2_pre, x1_cur, x2_cur = x1_cur, x2_cur, x1_next, x2_next
                f_cur = f
            # 初めは変数に代入するだけ
            elif i == 0:
                x1_pre = x1
                x2_pre = x2
            elif i == 1:
                x1_cur = x1
                x2_cur = x2
                f_cur = f
            else:
                print('break')
                break

    # 尤度関数
    def likelihood(self, x1_pre, x2_pre, x1_cur, x2_cur, x1_next, x2_next, k1_conditioned, k2_conditioned, f):
        '''
        尤度関数

        Attributes
        ----------
        x1_pre, x2_pre : float
            1ステップ前の変位
        x1_cur, x2_cur : float
            現在の変位
        x1_next, x2_next : float
            1ステップ後の変位
        k1_conditioned, k2_conditioned : float
            条件付けたkの平均値
        f : float
            入力された加速度
        '''
        mu1 = -x1_cur/self.m1*self.delta2*k1_conditioned - self.delta2/self.m1*(x1_cur-x2_cur)*k2_conditioned + 2*x1_cur - x1_pre - f*self.delta2
        mu2 = (x1_cur-x2_cur)/self.m2*self.delta2*k2_conditioned + 2*x2_cur - x2_pre
        std1 = 5e-2
        std2 = 5e-2
        p_1 = st.norm.pdf(x1_next, loc=mu1, scale=std1)
        p_2 = st.norm.pdf(x2_next, loc=mu2, scale=std2)
        p = p_1*p_2
        return p