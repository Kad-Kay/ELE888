%% Lab 3 Exercise 1 and 2
%Shahezad
%Rinay
%Barry
clear all
close all
clc
%% Setting Parameters
x1 = [ -1 1 -1 1 ]; 
x2 = [ -1 1 1 -1 ]; 
t = [ -1 -1 1 1 ];
theta = 0.001;
eta = 0.1;
epoch = 0;
wij = rand(2,3);
wkj = rand(1,3);
X1 = (x1 - (mean(x1)))/(std(x1));
X2 = (x2 - (mean(x2)))/(std(x2));
J = 1;
%% Neural Network
while (J > theta)
    epoch = epoch + 1; 
    m = 0;
    %initializing the delta values
    delwij = zeros(2,3); delwkj = zeros(1,3);
    %Sending Information through Neural network
    for m = 1:4
          x_in = [ 1 X1(m) X2(m) ]; 
       temp = zeros(1,2);
       hidden = zeros(1,2);
        for i = 1:2
           temp(i) = wij(i,:) * x_in';
           [hidden(i), hidden_deriv] = sigmoid(temp(i)); 
        end
        hidden = [ 1 hidden ];
        [Output(m),z_deriv] = sigmoid((wkj * hidden'));
        %Begin Back Propagation to refine result
        temp2 = zeros(1,2);
        for i = 1:2
            temp2(i) = (1 - (tanh(temp(i))).^2) * (t(m) - Output(m)) * z_deriv * wkj(i+1);
        end
        for i = 1:3
            for j = 1:2
                delwij(j,i) =  delwij(j,i) + eta * temp2(j) * x_in(i)';
            end
        end
        for i = 1:3
            delwkj(i) =   delwkj(i) + eta * (t(m) - Output(m)) * z_deriv * hidden(i)';
        end
    end
    %Apply changes calculated in back propagation to the weights
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
x_axis = -1.5:0.001:1.5;
%Boundaries calculations
Boundary = (-wij(1,2)/wij(1,3)) * x_axis + (-wij(1,1)/wij(1,3));
Boundary1 = (-wij(2,1)/wij(2,3)) + (-wij(2,2)/wij(2,3)) * x_axis;
%Plotting the results
figure;
plot(x1(:,1:2),x2(:,1:2),'ko',x1(:,3:4),x2(:,3:4),'b*');
hold on;
plot(x_axis, Boundary,'b',x_axis, Boundary1, 'r');
xlabel('x_1');ylabel('x_2');legend('Class 1', 'Class 2','Bound 1', 'Bound 2');
grid;title('Results');
hold off;
%Plotting the learning curve
figure;
plot(Jp, 'm');xlabel('Epoch'); ylabel('error rate');grid;title('Learning Curve');
epoch