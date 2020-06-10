# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:11:50 2020

@author: s1997003
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def cov_exp(x1,x2):
    '''
    Squared exponential covariance function.
    
    Input two arrays and returns a covariance matrix.
    '''

    
    diff = abs(np.sum(x1,1).reshape(-1,1) - np.ravel(x2))
    
    return  np.exp(-0.5*diff**2)

def cov_matern(x1, x2, l = 1.0):
    '''
    Matern covariance function with v= 3/2
    
    Input two arrays and returns a covariance matrix.
    '''
    
    diff = abs(np.sum(x1,1).reshape(-1,1) - np.ravel(x2))
     
    return (1+np.sqrt(3)*diff/l)*np.exp(-np.sqrt(3)*diff/l)
    

def cov_rq(x1, x2,alpha=0.5, l=1):
    '''
    Rational Quadratic Covariance Function
    '''
    
    diff = abs(np.sum(x1,1).reshape(-1,1) - np.ravel(x2))
    
    return (1+diff**2 / (2*alpha*l))**-alpha




def predictive_process(x, x_s, f, k):
    '''
    Returns the predictive mean and predictive covariance for use
    in gaussian regression
    Input:
        x: set of training points 
        x_s: domain
        f: function to emulate
        k: covariance function
    
    '''
    y = f(x) 
    K = k(x, x) 
    K_s = k(x, x_s)
    K_ss = k(x_s, x_s) 
    K_inv = inv(K)
    

    predictive_mean = np.dot(np.dot(K_s.T,K_inv),(y))

    predictive_cov = K_ss - np.dot(np.dot(K_s.T,K_inv),(K_s))
    
    return predictive_mean, predictive_cov

def predictive_process2(x, x_s, f, k):
    '''
    Work in progress. Algorithm 2.1 (Rasmussen & Williams 2006) page 19
    '''
    
    y= f(x)
    L = np.linalg.cholesky(k(x,x)+ 1e-9 *np.eye(len(x)))
    alpha = L.T @ (L@y)
    f_s = np.dot(k(x,x_s).T,alpha) # predictive mean
    v = L @ k(x,x_s)
    Vf = k(x_s,x_s) - np.dot(v.T,v) #Predictive cov
    
    return f_s, Vf



def plot_gpr(x, x_s, f, mu, covs):
    '''
    Plot gaussian process compared with a true solution
    '''
    sol = np.random.multivariate_normal(mu.ravel(), covs) #GP(muf, kn(x,x))
    
    plt.plot(x, f(x), 'ro')
    plt.plot(x_s, f(x_s))
    plt.plot(x_s, sol, linestyle="--")
    plt.legend(["Training points", "Solution", "Emulation"])
    plt.show()



### Testing 
x = np.array([-1, 0, 0.5]).reshape(-1,1)
x_s = np.arange(-1,1,0.01).reshape(-1,1)

mu, covs = predictive_process(x, x_s, np.sin, cov_exp)
    
plt.figure(1)
plt.title("Sq exponential cov function")
plot_gpr(x, x_s, np.sin, mu, covs)

x = np.array([-1,-0.5, 0, 0.5,1]).reshape(-1,1)

mu, covs = predictive_process(x, x_s, np.sin, cov_exp)

plt.figure(2)
plt.title("Sq exponential cov function")
plot_gpr(x, x_s, np.sin, mu, covs)


x = np.array([-1,-0.5, 0, 0.5,1]).reshape(-1,1)

mu, covs = predictive_process(x, x_s, np.sin, cov_matern)

plt.figure(3)
plt.title("Matern cov function, v=3/2")
plot_gpr(x, x_s, np.sin, mu, covs)


x = np.array([-1,-0.5, 0, 0.5,1]).reshape(-1,1)

mu, covs = predictive_process(x, x_s, np.sin, cov_rq)

plt.figure(4)
plt.title("Rational quadratic cov function")
plot_gpr(x, x_s, np.sin, mu, covs)