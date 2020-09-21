%% Lab 3 Exercise 3 and 4
%Shahezad
%Rinay
%Barry
clear all;
close all;
clc;
load winedata.mat;
%% Setting Parameters
x1 = [data(1:59,2); data(131:178, 2)]';
x2 = [data(1:59,3); data(131:178, 3)]';
t= [ones(59,1);-1*ones(48,1)]';  
theta = 0.001;
eta = 0.1;
epoch = 0;
wij = rand(2,3);
wkj = rand(1,3);
X1 = (x1 - (mean(x1)))/(std(x1));
X2 = (x2 - (mean(x2)))/(std(x2));
J = 1;
%% Neural Netowrk
while (J > theta)
    epoch = epoch + 1; 
    m = 0;
    %initializing the delta values
    delwij = zeros(2,3); delwkj = zeros(1,3);
    %Sending Information through Neural Network
    for m = 1:length(X1)
          x_in = [ 1 X1(m) X2(m) ]; 
       temp = zeros(1,2);
       hidden = zeros(1,2);
        for i = 1:2
           temp(i) = wij(i,:) * x_in';
           [hidden(i),y_deriv] = sigmoid(temp(i)); 
        end
        hidden = [ 1 hidden ];
        [Output(m),z_deriv] = sigmoid((wkj * hidden'));
        %Begin Back Propagation to refine result
        temp2 = zeros(1,2);
        for j = 1:2
            temp2(j) = (1 - (tanh(temp(j))).^2) * (t(m) - Output(m)) * z_deriv * wkj(j+1);
        end
        for i = 1:3
            for j = 1:2
                delwij(j,i) =  delwij(j,i) + eta * temp2(j) * x_in(i)';
            end
        end
        for l = 1:3
            delwkj(l) =   delwkj(l) + eta * (t(m) - Output(m)) * z_deriv * hidden(l)';
        end
    end
    %Apply changes calculated in back propogation to weights
    wij = wij + delwij;
    wkj = wkj + delwkj;
    %Check to see if the cost function has achieved a level below threshold
    Jp(epoch) = 0.5 * (((t' - Output')') * (t' - Output')); 
    if (epoch == 1)
        J = Jp(epoch);
    else
        J = Jp(epoch-1) - Jp(epoch);
    end
    J = abs(J);
end
%% Data Plotting and Results
% X-Axis Formation
x_axis = -5:0.005:5;
%Boundary Calculations
Boundary = (-wij(1,2)/wij(1,3)) * x_axis + (-wij(1,1)/wij(1,3));
Boundary1 = (-wij(2,2)/wij(2,3)) * x_axis + (-wij(2,1)/wij(2,3));
%Plotting the results
figure;
plot(X1(1:59),X2(1:59),'ko',X1(60:107),X2(60:107),'b*'); 
hold on;
plot(x_axis, Boundary,'b',x_axis, Boundary1, 'r');
xlabel('x_1');ylabel('x_2');legend('W1','W2','Boundary 1', 'Boundary 2');
grid;title('Results');
hold off;
%Plotting the learning curve
figure;
plot(Jp,'m');xlabel('Epoch'); ylabel('Error Rate');grid;title('Learning Curve');
%% Accuracy Calculations
correct = 0;
for i=1:length(X1)
    if ( (Output(i)<0 && t(i)==-1) || (Output(i)>0 && t(i) == 1) );
        correct = correct + 1;
    end
end
accuracy= correct/length(X1)*100
epoch