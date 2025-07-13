function [nrmse,output_tranz] = FORCEpp (Nnet, Anetwork, Bnetwork, input_training, output_training, input_testing, output_testing, input_validation, output_validation, rho, mean_output_training, tranz, trans,Ts)

state = zeros(Nnet, length(input_training));
orig_output_training = output_training;
% Computing the readout
state(:,1) = Bnetwork*input_training(:,1);
Cfnetwork = zeros(1,length(Anetwork(1,:)));
deltafb = 8;
lambdafb = 0.94;
% The recursive matrix
Pfb = 1/deltafb * eye(length(Anetwork(1,:)));
% The first state
state(:,1)=(Bnetwork*input_training(:,1));
% a = zeros(length(input_training),5);
% Recursive algorithm
for i=2:length(input_training)
   
    % Next state
    state(:,i)=(((Anetwork)*state(:,i-1))+Bnetwork*input_training(:,i)+Cfnetwork*state(:,i-1));
    outfb = Cfnetwork*state(:,i);
    % Error between the actual output and the target output when the main unknown to compute is K
    alfafb = outfb - output_training(i);
    % Gain matrix
    Pfb = 1/lambdafb*Pfb - 1/lambdafb*(Pfb*state(:,i)*state(:,i)'*Pfb)/(lambdafb+state(:,i)'*Pfb*state(:,i));
    % New recursive matrix
    Cfnetwork = Cfnetwork - alfafb*(Pfb*state(:,i))';

end
state = zeros(Nnet, length(input_testing));
state(:,1) = Bnetwork*input_testing(:,1);
outputpred = zeros(1,length(input_testing));
for i=2:length(input_testing)
    state(:,i)=(((Anetwork)*state(:,i-1))+Bnetwork*input_testing(:,i)+Cfnetwork*state(:,i-1));
    outputpred(i) = Cfnetwork*state(:,i);
end

outputpred = outputpred + mean_output_training;

% Compute the error on the testing data
transit = floor(length(input_testing)/4);
transit = 0;
tnet = 0:Ts:length(output_testing)*Ts-Ts-transit*Ts;
output_testing_tranz = output_testing(:,transit+1:length(output_testing));
output_tranz = outputpred(:,transit+1:length(outputpred),:);

nrmse = sum((output_tranz-output_testing_tranz).^2)/sum((output_testing_tranz-mean(output_testing_tranz)).^2)

% Plot the response of the network with the testing data


end