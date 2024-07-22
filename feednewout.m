function output = feednewout(tinput, toutput, A, B, C, prevstate)
    output = tinput + inv(C*B)*C*tanh(A*prevstate)-inv(C*B)*toutput;
end