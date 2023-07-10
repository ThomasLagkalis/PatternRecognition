clear all; clc;

%% Question 1: Epochs-400
accuracy = [0.8796, 0.9353]; %train, test
X1 = categorical({'test', 'train'});
figure(1);
b = bar(X1, accuracy);
b.FaceColor = 'flat';
b.CData(2,:) = [.5 0 .5];
ylabel('Accuracy')

%For epoch 400 (last one)
loss = [2.0513, 0.0190];
X2 = categorical({'validation set', 'train set'});
figure(2)
b = bar(X2, loss);
b.FaceColor = 'flat';
b.CData(2,:) = [.5 0 .5];
ylabel('loss')
title('Loss at epoch 400 / last epoch')

%% Question 2: 
% Accuracy for each optimazation algorithm
adam = 0.8891;
sgd = 0.8772;
rmsprop = 0.8769;
nadam = 0.8889;
adamax = 0.8881;
ftrl = 0.8043;
accuracy = [adam, sgd, rmsprop, nadam, adamax, ftrl];
X = categorical({'adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'ftrl'});
figure(3)
bar(X, accuracy)
set(gca, 'YScale', 'log')
ylabel('Accuracy')
title('Accuracy of the model with different optimizers')
yMin = min(accuracy); % Minimum value of data
yMax = max(accuracy); % Maximum value of data
ylim([yMin-0.03 yMax+0.02]) % Set the y-axis limits based on the minimum and maximum values of data

%% Question 3:
% Val loss untul epoch 20 (of CNN)
Val_loss_CNN = [0.2807, 0.2485, 0.2435, 0.2417, 0.2472, 0.2347, 0.2376, 0.2432, 0.2597, 0.2620, 0.3102, 0.3895, 0.3230, 0.3624, 0.3688, 0.3423, 0.3873, 0.4221, 0.4582, 0.4973];
Val_loss_ANN = [0.4311, 0.3857, 0.3943, 0.3706, 0.3553, 0.3627, 0.3471, 0.3360, 0.3451, 0.3471, 0.3314, 0.3568, 0.3427, 0.3338, 0.3410, 0.3482, 0.3701, 0.3506, 0.3775, 0.3418, 0.3503, 0.3700, 0.3682, 0.4157, 0.4019, 0.3748, 0.3957, 0.4053, 0.3987, 0.4038, 0.4268, 0.4234, 0.4232, 0.4272, 0.4357, 0.4441, 0.4805, 0.4752, 0.4496, 0.4503];
X1 = [1, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 17, 18 ,19 , 20];
X2 = [1, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 17, 18 ,19 , 20, 21, 22, 23, 24 ,25,26 ,27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37 ,38, 39,40];

figure(4)
plot(X2, Val_loss_ANN, 'blue', LineWidth=2)
hold on;
plot(X1, Val_loss_CNN, 'red', LineWidth=2)
legend('Simple ANN', 'CNN' )
xlabel('Epoch')
ylabel('Validation loss')