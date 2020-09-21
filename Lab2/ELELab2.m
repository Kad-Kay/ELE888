%% 
%----------------------------------------%
%----------------------------------------%
clear;
clc;
close all;

load("irisdata.mat");

Training_Percentage = 0.3;
n = 0.01;
theta = 0;
a1 = [0,0,1];
a2 = [0,0,1];

Data_set_A = irisdata_features (1:50, 2:3);
Data_set_B = irisdata_features (51:100, 2:3);
Data_set_C = irisdata_features (101:150, 2:3);

Training_A = Data_set_A(1:(50*Training_Percentage), 1:2);
Training_B = Data_set_B(1:(50*Training_Percentage), 1:2);
Training_C = Data_set_C(1:(50*Training_Percentage), 1:2);


Testing_A = Data_set_A((50*Training_Percentage):(50),1:2);
Testing_B = Data_set_B((50*Training_Percentage):(50),1:2);
Testing_C = Data_set_C((50*Training_Percentage):(50),1:2);


%% For Data Sets A and B
Y_AB(1:(size(Training_A)),2:3) = Training_A(:,:);
Y_AB((size(Training_A)+1):(size(Training_A)+size(Training_B)),2:3) = -1*Training_B;
Y_AB((size(Training_A)+1):(size(Training_A)+size(Training_B)),1) = -1;
Y_AB((1:(size(Training_A))),1) = 1;
Y_AB = Y_AB.';
Y_AB_Size = size(Y_AB);
Y_AB_Size = Y_AB_Size(1,2);

Delta_J = 0;

for i = 1:300
    AY = a1*Y_AB;
    
    for x = 1:Y_AB_Size
        if AY(1,x) <= theta
            Delta_J = Delta_J - Y_AB(:,x);
        end
    end
    if Delta_J == 0
        break
    else
        a1 = a1 - n*Delta_J.';
        Delta_J=0;
    end
end

disp(['The amount of iterations was ' num2str(i)]);

figure;

for i = 1:(size(Training_A))
    plot(Training_A(i,1), Training_A(i,2),'ob');
    hold on;
end

for i = 1:(size(Training_B))
    plot(Training_B(i,1), Training_B(i,2),'xr');
    hold on;
end

x = 1:5;
Boundary = (-1*a1(2)/a1(3))*x+a1(1)/a1(3);

grid;
plot(x,Boundary);
hold off;

%% Testing Data Set A and B

Correct_Classification = 0;
for i = 1:(size(Testing_A,1))
    Gx = a1(1) + a1(2)*Testing_A(i,1) + a1(3)*Testing_A(i,2);
    if Gx > 0
        Correct_Classification = Correct_Classification+1;
    end
    Gx = a1(1) + a1(2)*Testing_B(i,1) + a1(3)*Testing_B(i,2);
    if Gx < 0
        Correct_Classification = Correct_Classification+1;
    end
end

Accuracy = Correct_Classification/(size(Testing_A,1)+size(Testing_B,1));
disp(['The Accuracy is ' num2str(Accuracy*100) '%'])

%% For Data Sets B and C
Y_BC(1:(size(Training_B)),2:3) = Training_B(:,:);
Y_BC((size(Training_B)+1):(size(Training_B)+size(Training_C)),2:3) = -1*Training_B;
Y_BC((size(Training_B)+1):(size(Training_B)+size(Training_C)),1) = -1;
Y_BC((1:(size(Training_B))),1) = 1;
Y_BC = Y_BC.';
Y_BC_Size = size(Y_BC);
Y_BC_Size = Y_BC_Size(1,2);


Delta_J = 0;

for i = 1:300
    AY = a2*Y_BC;
    
    for x = 1:Y_AB_Size
        if AY(1,x) <= theta
            Delta_J = Delta_J - Y_BC(:,x);
        end
    end
    if Delta_J == 0
        break
    else
        a2 = a2 - n*Delta_J.';
        Delta_J=0;
    end
end

disp(['The amount of iterations was ' num2str(i)]);

figure;

for i = 1:(size(Training_B))
    plot(Training_B(i,1), Training_B(i,2),'ob');
    hold on;
end

for i = 1:(size(Training_C))
    plot(Training_C(i,1), Training_C(i,2),'xr');
    hold on;
end

x = 1:5;
Boundary = (-1*a2(2)/a2(3))*x+a2(1)/a2(3);

grid;
plot(x,Boundary);

%% Testing Data B and C

Correct_Classification = 0;
for i = 1:(size(Testing_C,1))
    Gx = a2(1) + a2(2)*Testing_B(i,1) + a2(3)*Testing_B(i,2);
    if Gx > 0
        Correct_Classification = Correct_Classification+1;
    end
    Gx = a2(1) + a2(2)*Testing_C(i,1) + a2(3)*Testing_C(i,2);
    if Gx < 0
        Correct_Classification = Correct_Classification+1;
    end
end

Accuracy = Correct_Classification/(size(Testing_B,1)+size(Testing_C,1));
disp(['The Accuracy is ' num2str(Accuracy*100) '%'])