# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:26:57 2024

@author: lazar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:51:29 2022

@author: lazar
"""
import numpy as np


def wnmf(X,k, Wght, l2_reg_H=0,l2_reg_W=0,l1_reg_H=0,l1_reg_W=0,b=2,
           max_iter=1000,max_tol=1e-3,max_err=0,replicates=1):
    
    '''
    
     Calculates a weighted version of nonnegative matrix factorization 
    X = W*H, where X is a [n, m] matrix, W is a [n, k] matrix and H is 
    a [k, m] matrix. Wght is a [k, m] weight matrix for H, placing emphasis 
    on specific regions of the k features in H to influence the reconstruction 
    of X.
    
    The optimization terminates when the change in H and W is not greater than
    max_tol, after max_iter iterations or when the norm of the error matrix
    is smaller than max_err. The multiplicative update rules, incorporating 
    the weight matrix Wght are used for the optimization (Lee and Seung, 2001).
    Optional L1 and L2 regularization terms  are included to promote sparsity 
    and smoothness respectively to W and H.
    
    # Input arguments:
    X          - Input matrix to be factorized (size n x m)
    k          - Number of modules/features
    Wght       - Weight matrix for H (size k x m)
    
    # Optional input arguments:
    l2_reg_H: L2 regularization for H, default = 0
    l2_reg_W: L2 regularization for W, default = 0
    l1_reg_H: L1 regularization for H, default = 0
    l1_reg_W: L1 regularization for W, default = 0
    b          - Divergence index; Set b=2 for Frobenius norm (default),
                  b=1 for KL-divergence or b=0 for Itakura-Saito divergence
    max_iter   - Maximum number of iterations after which to stop
    max_tol    - Maximum tolerance below which to stop
    max_err    - Maximum error below which to stop
    replicates - Number of optimization repetitions for avoiding local
                  minima
    
    # Output:
    # %  W          - First matrix factor (size n x k)
    # %  H          - Second matrix factor (size k x m)
    # %  err        - Final reconstruction error
    
    
    '''

    # if ~optimize_W && (size(W,1)~=size(X,1) || size(W,2)~=k)
    #     error('nmf: X, W and k must have compatible sizes.');
    # end
    
    eps=1e-15
    n = X.shape[1]
        
    err_best = np.inf
    for j in range(replicates):
        W = np.random.uniform(size=(n,k))
        H = np.dot(np.linalg.pinv(W),X.T)
        H[H<0] = 0
        i = 0
        err = np.inf
        tol = np.inf
    
        while tol>max_tol and i<max_iter and err>max_err:
            H_old = H
            # scale H l2 regularization
            if l2_reg_H!=0:
                l2_H=0.5*l2_reg_H*H.shape[1]
            else:
                l2_H=0
                
            # scale H l1 regularization
            if l1_reg_H!=0:
                l1_H=l1_reg_H*H.shape[1]
            else:
                l1_H=0
            # Apply multiplicative update rules
            WH = np.dot(W, H) + eps
            H = H * (W.T@((Wght.T*WH)**(b-2)*(Wght*X).T)) / (
                W.T@(Wght.T*WH)**(b-1) + eps +H*l2_H+l1_H)
            WH = W @ H + eps
                
            # W l2 regularization
            if l2_reg_W!=0:
                l2_W=0.5*l2_reg_W*W.shape[0]
            else:
                l2_W=0 
                
            # W l1 regularization
            if l1_reg_W!=0:
                l1_W=l1_reg_W*W.shape[0]
            else:
                l1_W=0 
                
            W_old = W
            W = W * (((Wght.T*WH)**(b-2)*(Wght*X).T)@H.T) / (
                (Wght.T*WH)**(b-1)@H.T + eps +W*l2_W+l1_W)
            WH = np.dot(W, H) + eps

            if b==0:
                # Minimize Itakura-Saito divergence
                err = sum(X / (WH+eps) - np.log(X / (WH+eps)+eps) - 1)
            elif b==1:
                # Minimize KL divergence
                err = np.sum(Wght.T*(X.T * np.log(X.T / (WH+eps)+eps) - X.T + WH))+l2_H
            else:
                # Minimize Frobenius norm
                err = np.linalg.norm(Wght.T*(X.T - WH))+ \
                    l2_H*(H**2).sum()+l2_W*(W**2).sum()+\
                      l1_H*(H).sum()+l1_W*(W).sum()  
    
            i = i+1
            tol = np.sum(np.abs(H-H_old))
            tol = tol + np.sum(np.abs(W-W_old))

        if err<err_best:
            err_best = err
            W_best = W
            H_best = H

    err = err_best
    W = W_best
    H = H_best

    return W,H,err
