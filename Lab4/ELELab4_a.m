%% ELE888 Lab4 Unsupervised Learning
%Shahezad
%Rinay
%Barry
clc
clear all
close all
%% Setting Up Parameters
Image = imread('Me.png');
X = double(reshape(Image, 667*664, 3));
J = [];
M = rand([2 3]) * 256;
M_pre(:,:,1) = [M(1,:)];
M_pre(:,:,2) = [M(2,:)];
M_check = zeros(size(M));
%% The algorithm
while(M_check ~= M)
    M_check = M;
    % Calculating distance to closest cluster mean
    J1 = sum((X - repmat(M(1,:), size(X,1), 1)).^2, 2);
    J2 = sum((X - repmat(M(2,:), size(X,1), 1)).^2, 2);
    % Recalculating mean values of Clusters
    cluster(:,1) = J1 < J2;
    cluster(:,2) = ~cluster(:,1);
    M(1,:) = sum(X(cluster(:,1), :))/sum(cluster(:,1));
    M(2,:) = sum(X(cluster(:,2), :))/sum(cluster(:,2));
    % Calculating J Value
    J = [J sum(min(J1,J2))];
    M_pre(size(M_pre,1)+1,:,1) = M(1,:);
    M_pre(size(M_pre,1),:,2) = M(2,:);
end
% Plotting error criterion
figure;plot(J);grid;title('Error Criterion');
% Cluster means for different stages of cluster process
figure; plot3(M_pre(:,1,1)/256, M_pre(:,2,1)/256, M_pre(:,3,1)/256,'-.*');hold;
plot3(M_pre(:,1,2)/256, M_pre(:,2,2)/256, M_pre(:,3,2)/256, '-.*');grid;hold;
xlabel('Red Mean');ylabel('Green Mean');zlabel('Blue Mean');
title('Cluster Mean Values');
% Plotting Data Points by Cluster Colors
X_Cluster1 = X(cluster(:,1),:);
X_Cluster2 = X(cluster(:,2),:);
figure;plot3(X_Cluster1(:,1), X_Cluster1(:,2), X_Cluster1(:,3),'.','Color', M(1,:)/256);
hold;plot3(X_Cluster2(:,1), X_Cluster2(:,2), X_Cluster2(:,3),'.','Color', M(2,:)/256);hold;
grid;title('Pixels Classified by Cluster');
% Picture Comparison
Rebuild = repmat(M(1,:), size(X,1), 1) .* repmat(cluster(:,1), 1, size(X,2));
Rebuild = Rebuild + repmat(M(2,:), size(X,1), 1) .* repmat(cluster(:,2), 1, size(X,2));
Rebuild = reshape(Rebuild, size(Image, 1), size(Image, 2), 3);
subplot(1,2,1);imshow(Image);
subplot(1,2,2);imshow(Rebuild/256);