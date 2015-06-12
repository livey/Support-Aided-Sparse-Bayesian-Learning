function x_est=SBL_NSL(y,A,sigma,partial_support,max_iter)
% this function implement the support aided SBL with non support 
% learning algorithm 
% Inputs:  
% y ---- measurments 
% A ---- measurement matrix 
% sigma -- noise variance 
% partial_support --- the support prior knowledge 
% max_iter --- maximal iteration 
[m,n]=size(A);
indexb=partial_support;
tau=0.5;
a=1e-10;                 
bb=1e-10*ones(n,1);      
bb(indexb)=tau;
iter=0;
D=eye(n);
sigma_new=sigma;
cov_new=inv(A'*A/sigma_new+D);
z_new=1/sigma_new*cov_new*A'*y;
z_old=ones(n,1);
iter =0; 
while(norm(z_new-z_old)/norm(z_old)>1e-5 && iter < max_iter)
    iter=iter+1 ;
    z_old=z_new;
    sigma_old=sigma_new;
    cov_old=cov_new;
    var=diag(cov_new);
    E=(z_new.^2+var);
    alpha_new=(1+2*a)./(E+2*bb);
    idx1=find(alpha_new>1e10);
    alpha_new(idx1)=1e10;
    D=diag(alpha_new);   
    cov_new=inv(A'*A/sigma_new+D);
    z_new=1/sigma_new*cov_new*A'*y;
end
x_est=z_new;
