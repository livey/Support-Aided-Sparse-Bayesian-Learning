function x_est=SBL_SL(y,A,sigma,partial_support,max_iter)
% this function implement the support aided SBL with support 
% learning algorithm 
% Inputs:  
% y ---- measurments 
% A ---- measurement matrix 
% sigma -- noise variance 
% partial_support --- the support prior knowledge 
% max_iter --- maximal iteration 

[m,n]=size(A);
iter=0;
iter_mx=1000;
D=eye(n);
sigma_new=sigma;
cov_new=inv(A'*A/sigma_new+D);
u_new=1/sigma_new*cov_new*A'*y;
u_old=ones(n,1);
a=1e-10;  
b=1e-15*ones(n,1);
c=0.1;
d=0.1;
indexb=partial_support;
while norm(u_new-u_old)/norm(u_old)>1e-5 && iter < max_iter
    iter=iter+1;
    u_old=u_new;
    cov_old=cov_new;
    var=diag(cov_old);
    E=u_new.^2+var;
    alpha_new=(1+2*a)./(E+2*b);
    b(indexb)=c./(alpha_new(indexb)+d);
    idx1=find(alpha_new>1e10);
    alpha_new(idx1)=1e10;
    D=diag(alpha_new);
    cov_new=inv(A'*A/sigma_new+D);
    u_new=1/sigma_new*cov_new*A'*y;
end

x_est=u_new;
