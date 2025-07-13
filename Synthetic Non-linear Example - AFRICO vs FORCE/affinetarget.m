function [output,state] = affinetarget(lensys, A,B,C,leninput,input)

state = zeros(lensys,leninput);
output = zeros(leninput,1);
state(:,1) = B*input(1);
a = 0.25;
b = 0.25;
for i = 2:leninput
    state(:,i) = A*state(:,i-1) + B*input(i);
    for j = 1:1:lensys-1
        output(i-1) = output(i-1) + a*(state(j+1,i)-state(j,i)^2)^2 - (b - state(j,i))^2;
    end
end

for j = 1:1:lensys-1
    output(i) = output(i) + a*(state(j+1,i)-state(j,i)^2)^2 - (b - state(j,i))^2;
end

end