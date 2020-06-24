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
    Avoids calculating inverse of a matrix
    Input:
        x: set of training points 
        x_s: domain
        f: function to emulate
        k: covariance function
    '''
    
    y = f(x)
        
    alpha = np.linalg.solve(k(x,x),y)
    
    f_s = np.dot(k(x,x_s).T,alpha) # predictive mean
    
    v = np.linalg.solve(k(x,x),k(x,x_s))
    
    Vf = k(x_s,x_s) - np.dot(k(x,x_s).T,v) #Predictive cov
    
    return f_s, Vf



def plot_gpr(x, x_s, f, mu, covs, samples=3, draw_samples=True):
    '''
    Plot gaussian process compared with a true solution
    Input:
        
    '''
    
    sol = np.random.multivariate_normal(mu.ravel(), covs,samples) #GP(muf, kn(x,x))
    
    
    plt.plot(x, f(x), 'rx',zorder = 10, label="Training points",markeredgewidth  = 2, markersize=8)
    plt.plot(x_s, f(x_s), linewidth = 2,  label="f",zorder=5,)
    plt.plot(x_s, mu.ravel(), linewidth = 2,  label=r"$\bar{f}_*$",zorder=4,color="red")
    
    #Plot functions from the posterior
    if draw_samples:
        for i in sol:
            plt.plot(x_s, i, linestyle="--",color="green")
    

    #plot credible interval
    plt.fill_between(x_s.ravel(),
                     y1=(mu.ravel() - 1.96* np.sqrt(covs.diagonal())),
                     y2=(mu.ravel() + 1.96* np.sqrt(covs.diagonal())),
                     color = "orange",
                     alpha = 0.4,
                     label = "Confidence interval"
    )
    
    plt.legend()
    plt.xlim(x_s[0],x_s[-1])
    
    plt.show()


### Testing 
x = np.array([-0.75, 0, 0.75]).reshape(-1,1)
x_s = np.linspace(-1,1,100, endpoint=True).reshape(-1,1)

mu, covs = predictive_process2(x, x_s, np.sin, cov_exp)
    
plt.figure(1)
plt.title("Sq exponential cov function")
plot_gpr(x, x_s, np.sin, mu, covs)


