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


def wnmf(X,k, Wght, l2_reg_H=0,l2_reg_W=0,l1_reg_H=0,l1_reg_W=0,W=None,b=2,
           max_iter=1000,max_tol=1e-3,max_err=0,replicates=1):
    # % Calculates the nonnegative matrix factorization X = W*H, where X is a
    # % [n, m] matrix, W is a [n, k] matrix and H is a [k, m] matrix. If W is
    # % specified then W is fixed and not optimized (for test sets). The
    # % optimization terminates when the change in H and W is not greater than
    # % max_tol, after max_iter iterations or when the norm of the error matrix
    # % is smaller than max_err. The multiplicative update rules are used for
    # % the optimization (Lee and Seung, 2001).
    # % Input arguments:
    # %  X          - Input matrix to be factorized (size n x m)
    # %  k          - Number of modules
    # % Optional input arguments:
    # %  W          - First matrix factor (size n x k); if specified, then W is
    # %               not optimized
    # %  b          - Divergence index; Set b=2 for Frobenius norm (default),
    # %               b=1 for KL-divergence or b=0 for Itakura-Saito divergence
    # %  max_iter   - Maximum number of iterations after which to stop
    # %  max_tol    - Maximum tolerance below which to stop
    # %  max_err    - Maximum error below which to stop
    # %  replicates - Number of optimization repetitions for avoiding local
    # %               minima
    # % Output:
    # %  W          - First matrix factor (size n x k)
    # %  H          - Second matrix factor (size k x m)
    # %  err        - Final reconstruction error
    
    # if ~optimize_W && (size(W,1)~=size(X,1) || size(W,2)~=k)
    #     error('nmf: X, W and k must have compatible sizes.');
    # end

    eps=1e-15
    n = X.shape[1]
    
    if W is None:
        optimize_W=True
    else:
        optimize_W=False
        
    err_best = np.inf
    for j in range(replicates):
        if optimize_W is True:
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
            if optimize_W is True:
                
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
            
            # margin normalization step 
            # for i_fac in range(k):
            #     cop_H=H[i_fac,:].reshape(200,200)
                
            #     for i in range(1000):
            #         marg1=np.sum(cop_H.reshape(200,200),axis=0)
            #         marg2=np.sum(cop_H.reshape(200,200),axis=1)
            #         cop_H=cop_H.reshape(200,200)/(marg1*marg2)
            #         cop_H=cop_H/np.sum(cop_H)
                
            #     H[i_fac,:]=cop_H.flatten()
            # H=H/np.sum(H,axis=1)[:,np.newaxis]

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
            if optimize_W is True:
                tol = tol + np.sum(np.abs(W-W_old))

        if err<err_best:
            err_best = err
            W_best = W
            H_best = H

    err = err_best
    W = W_best
    H = H_best

    return W,H,err
