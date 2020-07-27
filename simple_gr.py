# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:11:50 2020

@author: s1997003
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d import Axes3D

def cov_expd(x1,x2):
    '''
    Squared exponential covariance function.
    
    Input two arrays and returns a covariance matrix.
    '''

    dist = distance_matrix(x1,x2)
    
    return  np.exp(-0.5*dist**2)

def cov_exp(x1,x2, l = 1, sigma = 1):
    '''
    Squared exponential covariance function.
    
    Input two arrays and returns a covariance matrix.
    '''

    
    diff = distance_matrix(x1,x2)
    
    return  sigma**2*np.exp(-0.5*diff**2/l**2)

def cov_matern(x1, x2, l = 1.0, sigma=1):
    '''
    Matern covariance function with v= 3/2
    
    Input two arrays and returns a covariance matrix.
    '''
    
    diff = distance_matrix(x1,x2)
     
    return sigma**2*(1+np.sqrt(3)*diff/l)*np.exp(-np.sqrt(3)*diff/l)
    

def cov_rq(x1, x2,alpha=0.5, l=1):
    '''
    Rational Quadratic Covariance Function
    '''
    
    diff = distance_matrix(x1,x2)
    
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
    y = f(x).reshape(-1,1) 
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
    

    #plot confidence interval
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

def make_grid(x):
    xx, yy = np.meshgrid(x,x)
    return np.array(list(zip(yy.ravel(), xx.ravel())))


def plot_gpr3d(x, x_s, f, mu, covs, labels = ["x","y","z"] ):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    grid = make_grid(x_s)
    training_points = make_grid(x)
    dots = len(x_s)
    #plot_gpr(x, x_s, np.sin, mu, covs)
    
    xx, yy = np.meshgrid(x,x)
    
    xs, ys = np.meshgrid(x_s,x_s)

    surf = ax.plot_surface(xs,ys, mu.reshape(dots,dots), alpha=0.8, label=r"$\bar{f}_*$")
    
    surf1 = ax.plot_surface(xs, ys, f(grid).reshape(dots,dots), alpha=0.5, label="f")
    
    surf2 = ax.scatter(xx, yy, f(training_points).reshape(len(x),len(x)), marker="x",
               label="Training points",zorder=10, color="r", s=100)
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    
    surf1._facecolors2d=surf._facecolors3d
    surf1._edgecolors2d=surf._edgecolors3d
    
    surf2._facecolors2d=surf._facecolors3d
    surf2._edgecolors2d=surf._edgecolors3d
    plt.tight_layout()
    ax.legend(loc="best") 
    plt.show()


def rejection_sample(mu, cov, a, b, s=100, itermax=10000):
    
    '''
    Input:
        mn: Mean, cov: Covariance matrix, a,b: boundries, s: samplesize
    output:
        sample
    '''
    sol =[]
    i=0
    for i in range(s):
        sample = np.random.multivariate_normal(mu.ravel(), cov)
        if max(sample)<b and min(sample>a):
            sol.append(sample)


    return np.array(sol)

def rejection_sample2(mu, cov, a, b, s=100):
    
    sol = np.random.multivariate_normal(mu.ravel(), cov, s) #GP(muf, kn(x,x))
    for i in range(sol.shape[1]):
        if (max(sol[i])>b or min(sol[i]<a)):
            sol = np.delete()


def error(mu, sol, err=2):
    '''
    Calculate error
    Input:
        mu: simulated sol,
        sol: true solution,
        err: Integer for some norm, inf for infity norm or MSE for mean sq error
    '''
    if err==2:
        return np.linalg.norm(mu-sol,err)
    elif err=='inf':
        return np.linalg.norm(mu-sol, np.inf)
    else:
        return np.square(sol-mu).mean()
def GridSearch(x, x_s, f, k, parL=[0.5,0.5], parU=[1.5,1.5], size=0.1):
    
    
    return 

### Testing 

x = np.array([-0.75, 0, 0.75]).reshape(-1,1)
x_s = np.linspace(-1,1,100,endpoint=True).reshape(-1,1)

mu, covs = predictive_process2(x, x_s, np.sin, cov_exp)

f = lambda x: np.sin(x)
#plot_gpr(x, x_s, np.sin, mu, covs)
sol = rejection_sample(mu, covs, -1.1, 1.1, 100)



    