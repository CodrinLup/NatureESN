function [output,state] = affinetarget(lensys, A,B,C,leninput,input)

state = zeros(lensys,leninput);
output = zeros(leninput,1);
state(:,1) = B*input(1);

for i = 2:leninput
    state(:,i) = A*state(:,i-1) + B*input(i);
    output(i-1) = C*state(:,i-1);
end

output(i) = C*state(:,i);

end