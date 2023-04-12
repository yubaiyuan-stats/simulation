

import numpy as np
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H  

def DataGene_nonLinear_low_rank(sample_size):

    ##generate the underlying latent confounder U 
        N = sample_size      
        U = np.zeros([N,1])
        U_ind = np.zeros([N,1])
        for i in range(N):
            ind =  np.random.binomial(1,0.5)
            U_ind[i] = ind
            if ind == 1:
               U[i] = np.random.normal(2,1.5,1)
            else: 
               U[i] = np.random.normal(-2,1.5,1)

    #generate T, M, Y   
               
        P_whole = np.exp(0.4*U)/(1+np.exp(0.4*U))
        #plt.hist(P_whole)
        
        treatment_whole = np.random.binomial(1,P_whole)
        #np.corrcoef(np.concatenate((treatment_whole,P_whole),axis=1).transpose()) 

        a1 = -np.piecewise(U, [U < -3, (U >= - 3)&(U<-1),(U >= -1)&(U<= 1),(U > 1)&(U<3),U >= 3], [1,2,-1,-2,-3])
        a2 = np.piecewise(U, [U < -4, (U >= - 4)&(U<-2),(U >= -2)&(U<= 0),(U > 0)&(U<2),(U >= 2)&(U<4), U >= 4], [-2,0.5,1,2,3,4])
        a3 = np.piecewise(U, [U < -3, (U >= - 3)&(U<3),U >= 3], [-1,2,3])
        
        
        a4 = np.sin(U)
        a5 = np.cos(U)*a1
        a6 = np.exp(-U/6)*a2
        
        
        Bias= np.concatenate((a1,a2,a3,a4,a5,a6),axis=1)
        #np.corrcoef(np.concatenate((U,Bias),axis=1).transpose())
        Bias = (Bias - np.mean(Bias,axis=0))/np.std(Bias,axis = 0)
        #pca = PCA(n_components=6)
        #pca.fit(Bias)
        #pca.components_.T
        #pca.explained_variance_ratio_
        
        
        X_whole = np.random.multivariate_normal(np.zeros(7), np.diag(np.ones(7)), N)
        
        noise_M = np.random.multivariate_normal(np.zeros(5), 0.5*np.diag(np.ones(5)), N)
        coef_M = rvs(dim=5)
        M_whole = 0.5*Bias[:,0:5] + [1,1,1,1,1]*treatment_whole + 0.3*np.matmul(X_whole[:,0:5],coef_M[:,0:5]) + 1*noise_M
        true_M_whole = 0.5*Bias[:,0:5] + [1,1,1,1,1]*treatment_whole + 0.3*np.matmul(X_whole[:,0:5],coef_M[:,0:5])
          
        noise_Y = np.random.multivariate_normal([0], [[0.3]], N)
        Y_whole = 0.5*treatment_whole + (2.5*Bias[:,5]).reshape(N,1) + np.matmul(M_whole,[0.5,0.5,0.5,0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1) + 1*noise_Y
        true_Y_whole = 0.5*treatment_whole + 2.5*Bias[:,5].reshape(N,1) + np.matmul(M_whole,[0.5,0.5,0.5,0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1)
 
        return X_whole, Y_whole, treatment_whole, M_whole
    
    