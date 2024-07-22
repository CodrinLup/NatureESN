function [output,state] = affinenetworkstep(A,B,C,K,input,prevstate)

state = tanh(A*prevstate) + B*input - B*K*prevstate;
output = C*state;

end