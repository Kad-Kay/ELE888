%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Group Members:
% Pranay Patel: 500702502
% Barry Chong: 500508396
% Shahezad Kassam :500682174

%% Lab1.m Calculating Probabilities

function [posteriors_x,g_x]=lab1(x,Training_Data)

% x = individual sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    
 
column = 1; % feature column 1 for width, 2 for length
f=D(:,column);  % feature samples
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

disp('Prior probabilities:');
Pr1 = length(find(la(:) == 1))/M
Pr2 = length(find(la(:) == 2))/M

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
m11 = mean(D(find(la(:) == 1),column))  % mean of the class conditional density p(x/w1)
std11 = std(D(find(la(:) == 1),column)) % Standard deviation of the class conditional density p(x/w1)

m12 = mean(D(find(la(:) == 2),column))  % mean of the class conditional density p(x/w2)
std12= std(D(find(la(:) == 2),column)) % Standard deviation of the class conditional density p(x/w2)


disp(['Conditional probabilities for x=' num2str(x)]);
cp11= 1/(sqrt(2*pi*std11^2))*exp(-(x-m11)^2/(2*(std11)^2)) % use the above mean, std and the test feature to calculate p(x/w1)

cp12= 1/(sqrt(2*pi*std12^2))*exp(-(x-m12)^2/(2*(std12)^2)) % use the above mean, std and the test feature to calculate p(x/w2)

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');

pos11= (cp11*Pr1)/(cp11*Pr1 + cp12*Pr2) % p(w1/x) for the given test feature value

pos12= (cp12*Pr2)/(cp11*Pr1 + cp12*Pr2) % p(w2/x) for the given test feature value

posteriors_x= [ pos11 pos12]

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x= pos11 - pos12 % compute the g(x) for min err rate classifier.
%% 
if(g_x<0) 
    disp(sprintf('This sample is Iris Versicolor\n--------------------------'));
else
    disp(sprintf('This sample is Iris Setosa\n--------------------------'));
    end
%% find threshold

syms thresh

eqn = 1/(sqrt(2*pi)*std11)*exp(-0.5*((thresh-m11)/std11)^2) * Pr1 == 1/(sqrt(2*pi)*std12)*exp(-0.5*((thresh-m12)/std12)^2)* Pr2;
thresh = solve(eqn, thresh);
thresh = double(thresh);
th1 = thresh(thresh>0)
hold on;
plot(thresh)
th_plot = thresh;

%TH with a higher penalty
th2 =thresh(thresh> ((0.25/0.75)*(Pr2/Pr1)))


