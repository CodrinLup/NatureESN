function state = targetstategen(lensys,leninput,input,A,B)

state = zeros(lensys,leninput);
state(:,1) = B*input(1);
for i = 2:leninput
    state(:,i) = A*state(:,i-1) + B*input(i);
end

end