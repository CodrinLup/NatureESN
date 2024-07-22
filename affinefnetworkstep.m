function [output,state] = affinefnetworkstep(A,B,C,K,input,prevstate)

state = tanh(A*prevstate) + B*input + K'*C*prevstate;
output = C*state;

end