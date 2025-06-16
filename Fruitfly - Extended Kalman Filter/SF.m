function result = SF(x,u, Acl, Nolags, Nnet)

B = zeros(Nnet,Nolags);
for ll = 1:Nolags
    B(:,ll) = x(Nnet+1+(ll-1)*Nnet:Nnet+ll*Nnet);
end
K = zeros(Nolags,Nnet);
for ll = 1:Nolags
    K(ll,:) = x(Nnet+Nnet*Nolags+1+(ll-1)*Nnet:Nnet+Nnet*Nolags+(ll)*Nnet)';
end
BB = zeros(length(Acl),Nolags);
BB(1:Nnet,1:Nolags) = B;
KK = zeros(Nolags,length(Acl));
KK(1:Nolags,1:Nnet) = K;
result = zeros(length(x),1);
result(1:Nnet) = tanh(Acl*x(1:Nnet) - B*K*x(1:Nnet) + B*u - ((B'*x(1:Nnet))'*K)');
result(Nnet+1:length(result)) = x(Nnet+1:length(x));
% result=((Acl-[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)])*x ...
%     +[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*u) - [x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*x'*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)]' ;
end