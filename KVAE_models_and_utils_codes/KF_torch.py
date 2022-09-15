
import torch
                         
# X : state
# P : covariance of state
# y : mesurement
# H : measurement matrix
# R : observation covariance
def kf_update(X, P, y, H, R):
    
    # Find innovation and innovation covariance
    innovationMean = y - torch.mm(H,X)
    H_transpose    = H.transpose(0,1)
    innovationCov  = torch.mm(H, torch.mm(P,H)) + R

    inverse_innovationCov = torch.inverse(innovationCov)
    
    # Kalman Gain
    kalmanGain = torch.mm(P, torch.mm(H_transpose, inverse_innovationCov))

    X = X + torch.mm(kalmanGain,innovationMean)
    
    # We also update the covariance estimation:
    covEstimatedTemp = torch.mm(kalmanGain, innovationCov)
    covEstimatedTemp = torch.mm(covEstimatedTemp, kalmanGain.transpose(0,1))
    P = P - covEstimatedTemp    
    
    return X, P
    
# X : state
# P : covariance of state
# A : transition matrix
# Q : prediction covariance matrix
# B : control matrix
# u : control vector (velocity)

def kf_predict(X,P,A,Q,B,u):
    
    X = torch.mm(A, X) + torch.mm(B,u)
    P = torch.mm(A, torch.mm(P, A.transpose(0,1))) + Q
       
    return X, P