function [output,state] = affinecnetworkstep(A,B,C,input,prevstate)

state = tanh(A*prevstate) + B*input;
output = C*state;

end