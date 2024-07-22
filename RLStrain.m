function [Knetwork,Bnetwork] = RLStrain(Anetwork,Bnetwork,state_training,input_training,sout,lennet,factsout,lambdafb, output_training)
Knetwork = rand(1,lennet);
% Initiliazation of the EKF algorithm
% Expanding system matrices
state = zeros(3*length(Anetwork(1,:)),length(input_training));
state(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)),1) = Bnetwork;
state(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)),1) = Knetwork;
state(:,1) = mean(state(:,1));
A = eye(3*length(Anetwork(1,:)), 3*length(Anetwork(1,:)));
A(1:length(Anetwork(1,:)),1:length(Anetwork(1,:))) = Anetwork;
B = zeros(3*length(Anetwork(1,:)),1);
B(1:length(Anetwork(1,:))) = Bnetwork;
K = zeros(1,3*length(Anetwork(1,:)));
K(1:length(Anetwork(1,:))) = Knetwork;
% Readout Jacobean
JH = zeros(1,3*length(Anetwork(1,:)));
JH(1,1) =  rand;
% Parameter initialization
Pk = eye(3*length(Anetwork(1,:)),3*length(Anetwork(1,:)));
Kk = zeros(3*length(Anetwork(1,:)),1);
% Recursive algorithm
for i=2:length(input_training)
    % model predictor
    tempstate=((A-B*K)*state(:,i-1)+B*input_training(:,i-1));
    % f Jacobean
    JF = A;
    for j = length(Anetwork(1,:))+1:2*length(Anetwork(1,:))
        JF(j-length(Anetwork(1,:)),j) = JF(j-length(Anetwork(1,:)),j) - K*state(:,i-1);
        for l = 2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:))
            JF(j-length(Anetwork(1,:)),l) = JF(j-length(Anetwork(1,:)),l) - B(j-length(Anetwork(1,:)))*state(l-2*length(Anetwork(1,:)),i-1);
        end
        
    end
    for j = 2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:))
        JF(j-2*length(Anetwork(1,:)),j) = JF(j-2*length(Anetwork(1,:)),j) - B(j-2*length(Anetwork(1,:)))*[ones(1,length(Anetwork(1,:))) zeros(1,2*length(Anetwork(1,:)))]*state(:,i-1);
    end
    Pkf = JF*Pk*JF';
    % step corrector
    
    Kk = Pkf*JH'*inv(JH*Pkf*JH');
    Pk = (eye(3*length(Anetwork(1,:)),3*length(Anetwork(1,:))) - Kk*JH)*Pkf;
    state(:,i) = tempstate + Kk*(output_training(i)-tempstate(1));
    B(1:length(Anetwork(1,:))) = state(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)),i);
    K(1,1:length(Anetwork(1,:))) = state(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)),i);
end
 Bnetwork = state(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)),i-1)';
 Knetwork = state(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)),i-1);
end