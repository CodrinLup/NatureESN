function nrmse = FORCE (Nnet, Anetwork, Bnetwork, input_training, output_training, input_testing, output_testing, tranz, mean_output_training)

state = zeros(Nnet, length(input_training));

% Computing the readout

state(:,1) = Bnetwork*input_training(:,1);
for i = 2:length(input_training)
    state(:,i) = Anetwork*state(:,i-1) + Bnetwork*input_training(:,i);
end

Cnetwork = output_training*pinv(state);

% Generating the output of the FORCE using the testing data

tnet = 0:1:length(input_testing)-1;
ynet = zeros(length(tnet),1);
state = zeros(Nnet,length(input_testing));
state(:,1) = Bnetwork*input_testing(:,1);
for i=2:length(tnet)
    state(:,i) = Anetwork*state(:,i-1) + Bnetwork*input_testing(:,i);
    ynet(i-1) = Cnetwork*state(:,i-1);
end
ynet(i) = Cnetwork*state(:,i);
ynet = ynet + mean_output_training;

% Computing the quality of the approximation with a
% transitory period

nrmse = sum((ynet-output_testing').^2)/sum((output_testing-mean(output_testing)).^2);

end