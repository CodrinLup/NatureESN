function output = affinefnetwork(lensys, A,B,C,K,leninput,input)

state = zeros(lensys,leninput);
output = zeros(leninput,1);
state(:,1) = tanh(A*zeros(lensys,1)) + B*input(1);

for i = 2:leninput
    state(:,i) = tanh(A*state(:,i-1)) + B*input(i) + K'*C*state(:,i-1);
    output(i-1) = C*state(:,i-1);
end

output(i) = C*state(:,i);

end