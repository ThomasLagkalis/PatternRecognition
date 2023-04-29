close all;
clear;
clc;

data_file = './data/mnist.mat';

data = load(data_file);

% Read the train data
[train_C1_indices, train_C2_indices,train_C1_images,train_C2_images] = read_data(data.trainX,data.trainY.');

% Read the test data
[test_C1_indices, test_C2_indices,test_C1_images,test_C2_images] = read_data(data.testX,data.testY.');


%% Compute Aspect Ratio


% Compute the aspect ratios of all images and store the value of the i-th image in aRatios(i)

%Computing the size of aRatios vector.
sizeOfaRatios = size(train_C1_images, 1) + size(train_C2_images, 1);
aRatios(sizeOfaRatios)=0;
aRatiosC1Train(size(train_C1_images, 1))=0;
aRatiosC2Train(size(train_C2_images, 1))=0;
for i=1:size(train_C1_images, 1)
    aRatiosC1Train(i) = computeAspectRatio(reshape(train_C1_images(i, :, :), 28, 28));
    aRatios(i) = computeAspectRatio(reshape(train_C1_images(i, :, :), 28, 28));
end

for i=1:size(train_C2_images, 1)
  aRatiosC2Train(i) = computeAspectRatio(reshape(train_C2_images(i, :, :), 28, 28));
  aRatios(i + size(train_C1_images, 1)) = computeAspectRatio(reshape(train_C2_images(i, :, :), 28, 28));
end

for i=1:size(test_C1_images, 1)
  aRatiosC1Test(i) = computeAspectRatio(reshape(test_C1_images(i, :, :), 28, 28));
  %aRatios(i + size(train_C1_images, 1) + size(train_C2_images)) = computeAspectRatio(reshape(test_C1_images(i, :, :), 28, 28));
end

for i=1:size(test_C2_images, 1)
  aRatiosC2Test(i) = computeAspectRatio(reshape(test_C2_images(i, :, :), 28, 28));
  %aRatios(i + size(train_C1_images, 1) + size(train_C2_images, 1) + size(test_C1_images, 1)) = computeAspectRatio(reshape(test_C2_images(i, :, :), 28, 28));
end

minAspectRatio = min(aRatios)
maxAspectRatio = max(aRatios)

figure(1); displayAspectRatio(reshape(train_C2_images(31, :, :), 28, 28));
figure(2); displayAspectRatio(reshape(train_C1_images(1, :, :), 28, 28));

%% Bayesian Classifier


% Prior Probabilities
PC1 = size(train_C1_images, 1)/(size(train_C1_images, 1)+size(train_C2_images, 1))
PC2 = size(train_C2_images, 1)/(size(train_C1_images, 1)+size(train_C2_images, 1))


% Likelihoods
x = [0:0.001:6];
mu1 = sum(aRatiosC1Train)/size(train_C1_images, 1);
mu2 = sum(aRatiosC2Train)/size(train_C2_images, 1);
sigma1 = sqrt(sum((aRatiosC1Train-mu1).^2)/size(train_C1_images, 1));
sigma2 = sqrt(sum((aRatiosC2Train-mu2).^2)/size(train_C2_images, 1));
PgivenC1 = normpdf(x, mu1, sigma1); %likelihood of C1.
PgivenC2 = normpdf(x, mu2, sigma2); %likelihood of C2.
figure(3); plot(x, PgivenC1, x, PgivenC2)


% Posterior Probabilities
Px = PgivenC1.*PC1 + PgivenC2.*PC2;
PC1givenL = (PgivenC1.*PC1)./Px;
PC2givenL = (PgivenC2.*PC2)./Px;
diff = (PgivenC2 - PgivenC1);
idx =find(diff(1:end-1).*diff(2:end)<0);
boundary = x(idx)
% Classification r  esult
testsRatios = [aRatiosC1Test aRatiosC2Test]; %Merging the two test aspect ratio classes.
Classes = [ones(1, size(aRatiosC1Test, 2)) 2*ones(1, size(aRatiosC2Test, 2))];

%BayesClass = ((boundary(1)< testsRatios) & (testsRatios < boundary(2))) +1;
BayesClass = (testsRatios < boundary(2)) +1;

% Count misclassified digits
count_errors = sum(BayesClass ~= Classes)

% Total Classification Error (percentage)
Error = count_errors/(size(aRatiosC1Test, 2) + size(aRatiosC2Test, 2))


%Function to display the rectangle of aspect ratio in two random number
%images from C1 and C2.

function displayAspectRatio(im)
    nonzeros_rows = find(any(im, 2));
    nonzeros_columns = find(any(im,1)); 
    image(im)
    colormap("gray")
    box = [min(nonzeros_columns)-1, min(nonzeros_rows)-1,  size(nonzeros_columns, 2)+1, size(nonzeros_rows, 1)+1];
    hold on;
    rectangle('Position', box, 'EdgeColor','r', 'LineWidth', 3 )
end

