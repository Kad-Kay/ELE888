%% ELE888 Lab4 Unsupervised Learning
%Shahezad
%Rinay
%Barry
clc
clear all
close all
XX = []
%% Setting Up Parameters
Image = imread('Me.png');
X = double(reshape(Image, 667*664, 3));
c=8;
J = zeros(size(X,1), c);
M_init = rand([c 3])*256
M = M_init;
M_check = zeros(size(M));
while (M_check ~= M)
    M_check = M;
    for i = 1:c
        J(:,i) = sum((X - repmat(M(i,:), size(X,1), 1)).^2, 2);
    end
    [garbo, cluster_temp] = min(J, [], 2);
    for i = 1:c
        cluster = (cluster_temp==i);
        M(i, :) = sum(X(cluster, :)) / sum(cluster);
    end
    figure;
    XX = zeros(size(X));
for i = [1:c]
    this_cluster = (cluster==i);
    XX = XX + repmat(M(i,:), size(X,1), 1) .* repmat(this_cluster, 1, size(X,2));
end
    XX = reshape(XX, size(Image, 1), size(Image, 2), 3);
    subplot(1,2,1);imshow(Image)
    subplot(1,2,2);imshow(XX/256)
end
M
figure; hold;
for i = 1:c
    cluster = (cluster_temp==i);
    Xplot = X(cluster, :);
    plot3(Xplot(:,1), Xplot(:,2), Xplot(:,3),'.','Color', M(i,:)/256)
end
hold;grid;title('Clustered Pixel Samples');xlabel('Red');ylabel('Green');zlabel('Blue');
N = size(X,1);
Xb1 = 0;
for i = 1:c
    cluster = (cluster_temp==i);
    Xi = X(cluster, :);
    mui_j = sort(sum((M - repmat(M(i,:), c, 1)).^2, 2).^.5);
    Xb1 = Xb1 + sum(sum((Xi - repmat(M(i,:), size(Xi,1), 1)).^2, 2).^.5) / mui_j(2);
end
Xb1 = Xb1 / N
%% Setting Up Parameters
Image = imread('Me.png');
X = double(reshape(Image, 667*664, 3));
J = zeros(size(X,1), c);
M_init = rand([c 3])*256
M = M_init;
M_check = zeros(size(M));
while (M_check ~= M)
    M_check = M;
    for i = [1:c]
        J(:,i) = sum((X - repmat(M(i,:), size(X,1), 1)).^2, 2);
    end
    [garbo, cluster_temp] = min(J, [], 2);
    for i = [1:c]
        cluster = (cluster_temp==i);
        M(i, :) = sum(X(cluster, :)) / sum(cluster);
    end
    figure;
        XX = zeros(size(X));
for i = [1:c]
    this_cluster = (cluster==i);
    XX = XX + repmat(M(i,:), size(X,1), 1) .* repmat(this_cluster, 1, size(X,2));
end
XX = reshape(XX, size(Image, 1), size(Image, 2), 3);
subplot(1,2,1);imshow(Image)
subplot(1,2,2);imshow(XX/256)
end
M
figure; hold;
for i = [1:c]
    cluster = (cluster_temp==i);
    Xplot = X(cluster, :);
    plot3(Xplot(:,1), Xplot(:,2), Xplot(:,3),'.','Color', M(i,:)/256)
end
hold;grid;title('Clustered Pixel Samples');xlabel('Red');ylabel('Green');zlabel('Blue');
N = size(X,1);
Xb2 = 0;
for i = 1:c
    cluster = (cluster_temp==i);
    Xi = X(cluster, :);
    mui_j = sort(sum((M - repmat(M(i,:), c, 1)).^2, 2).^.5);
    Xb2 = Xb2 + sum(sum((Xi - repmat(M(i,:), size(Xi,1), 1)).^2, 2).^.5) / mui_j(2);
end
Xb2 = Xb2 / N