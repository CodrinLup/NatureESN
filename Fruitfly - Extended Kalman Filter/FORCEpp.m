function nrmse = FORCEpp (Nnet, Anetwork, Bnetwork, input_training, output_training, input_testing, output_testing, input_validation, output_validation, rho, mean_output_training, tranz, trans)

state = zeros(Nnet, length(input_training));
orig_output_training = output_training;
% Computing the readout
state(:,1) = Bnetwork*input_training(:,1);
Cfnetwork = zeros(1,length(Anetwork(1,:)));
deltafb = 80;
% The recursive matrix
Pfb = 1/deltafb * eye(length(Anetwork(1,:)));
% The first state
state(:,1)=(Bnetwork*input_training(:,1));
% a = zeros(length(input_training),5);
% Recursive algorithm
for i=2:length(input_training)
   
    % Next state
    state(:,i)=((Anetwork)*state(:,i-1))+Bnetwork*input_training(:,i)+Cfnetwork*state(:,i-1);
    outfb = Cfnetwork*state(:,i);
    % Error between the actual output and the target output when the main unknown to compute is K
    alfafb = outfb - output_training(i);
    % Gain matrix
    Pfb = Pfb - (Pfb*state(:,i)*state(:,i)'*Pfb)/(1+state(:,i)'*Pfb*state(:,i));
    % New recursive matrix
    Cfnetwork = Cfnetwork - alfafb*(Pfb*state(:,i))';

end

statepred = statepredgen(input_testing,length(input_testing),Nnet,Anetwork,Bnetwork);
outputpred = Cfnetwork*statepred;
outputpred = outputpred + mean_output_training;

% Compute the error on the testing data
trans = length(input_testing)/4;
tnet = 1:length(input_testing)-trans-1;
output_testing_tranz = output_testing(:,trans+1:length(output_testing)-1);
output_tranz = outputpred(:,trans+1:length(outputpred)-1,:);

nrmse = sum((output_tranz-output_testing_tranz).^2)/sum((output_testing_tranz-mean(output_testing_tranz)).^2)

% Plot the response of the network with the testing data

figure(11)
plot(tnet, output_testing_tranz, '-');             % Test signal
hold on
plot(tnet,  output_tranz, '--o');                 % ARLS signal
xlabel('KT','fontsize', 18)
ylabel('y(KT)','fontsize', 18)
set(gca,'FontSize',16)

end