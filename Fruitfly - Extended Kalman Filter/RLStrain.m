function Knetwork = RLStrain(Anetwork,Bnetwork,state_training,input_training,sout,lennet,factsout,lambdafb)
Knetwork = zeros(1,lennet);
% Initiliazation of the algorithm
% Initial state + vector of as many states as training points
state = zeros(length(Anetwork(1,:)),length(input_training));
% Forgetting factor
% Initialization factor for the recursive matrix using the standard
% deviation of the training states
deltafb = factsout*diag(sout.^2);
% The recursive matrix
Pfb = deltafb * eye(length(Anetwork(1,:)));
% The first state
state(:,1)=(Bnetwork*input_training(:,1));
% a = zeros(length(input_training),5);
% Recursive algorithm
for i=2:length(input_training)
    outfb = input_training(:,i) + Bnetwork'/norm(Bnetwork)^2*Anetwork*state(:,i-1)-Bnetwork'/norm(Bnetwork)^2*state_training(:,i);
    % Error between the actual output and the target output when the main unknown to compute is K
    alfafb = outfb - Knetwork*state(:,i-1);
    % Gain matrix for K
    gfb = Pfb*state(:,i-1)*inv(lambdafb+state(:,i-1)'*Pfb*state(:,i-1));
    % New recursive matrix for computing K
    Pfb = 1/lambdafb*Pfb-1/lambdafb*gfb*state(:,i-1)'*Pfb;
    % Update of the feedbackmatrix
    Knetwork = Knetwork + alfafb*gfb';
    % Next state
    state(:,i)=((Anetwork-Bnetwork*Knetwork)*state(:,i-1)+Bnetwork*input_training(:,i));
end
end