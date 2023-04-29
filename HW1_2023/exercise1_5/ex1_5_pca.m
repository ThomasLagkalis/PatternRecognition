%% Pattern Recognition 2023
%  Exercise 1.5 | Principle Component Analysis
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     myPCA.m
%     projectData.m
%     recoverData.m
%     featureNormalize.m
%
%  Add your code in this file as requested in exercise 1 of HW1,
%  or any other given files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% ============ Part 1: PCA on 2D Samples ===================
%  We use a small dataset that is easy to visualize
%
fprintf('Readinging and Visualizing initial 2D dataset.\n\n');

%  The following command loads the dataset. You should now have the 
%  variable X in your environment
load ('data/data1.mat');

%  Visualize the samples (Input your code)
figure(1);subplot(1,3,1); scatter(X(:,1), X(:,2), 15, "filled");title('Pure Data')
%plot the samples  
axis([0 6.5 2 8]); axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;


%  Before running PCA, it is important to first normalize X
%  Add your code in the file featureNormalize.m
%

[X_norm, mu, sigma] = featureNormalize(X);
%plot the samples 
subplot(1,3,3); scatter(X(:,1), X(:,2), 15, "filled","MarkerFaceColor",'r')
hold on; scatter(X_norm(:,1), X_norm(:,2), 15, "filled","MarkerFaceColor",'b')
axis([-2.5 6.5 -2.5 8]); axis square;
legend("Pure Data", "Normalized Data", 'location', "northwest")
hold off;
subplot(1,3,2); scatter(X_norm(:,1), X_norm(:,2), 15, "filled");title('Normalized Data') 
axis([-2.5 3 -2.5 4]); axis square;


%  Extract Principal Components 
%  Add your code in the file myPCA.m
[U, S] = myPCA(X_norm);
eigenval=diag(S); %Vector of eigenvalues
[eigenval,ind]=sort(eigenval,1,'descend'); %Sort them
U=U(:,ind); %Corresponding eigenvectors
S=eigenval.*S;


%  Compute mu, the mean of each feature
%mu = mean(X)

%  Draw the eigenvectors centered at the mean of samples. These lines show the
%  directions of maximum variations in the dataset.
hold on;
line([0;U(1,1)],[0; U(2,1)]) % eigenvector
line([0;U(1,2)],[0; U(2,2)]) % eigenvector
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(1,1) * -U(:,1)', '-k', 'LineWidth', 2);
%drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n');


fprintf('Program paused. Press enter to continue.\n');
pause;
%%
%  First estimate the variance contribution of each principal component

PCvariance = eigenval/sum(eigenval);
fprintf(' PCvariance = %f %f \n', PCvariance(1,1), PCvariance(2,1));
fprintf('\n(you should expect to see 0.8678 and 0.1322)\n');

%  Apply PCA to perform Dimensionality Reduction
%  You should complete the code in projectData.m to perform the projection
%  in a lower space based on the first k eigenvectors
%
%
fprintf('\nDimension reduction of dataset samples.\n\n');

%  Plot the normalized dataset (returned from myPCA)
figure(2)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square

%  Project the data onto K = 1 dimension
%  Add your code to complete the projectData() function
K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');

%  To visualize the samples of this reduced dimensional space 
%  we have to recostruct (project) them back to the original one.  
%  This will show you what the data looks like when 
%  using only the corresponding eigenvectors to reconstruct it.
%  Add your code to complete the recoverData() function
X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');

%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 2: PCA on Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('data/faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%  PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = myPCA(X_norm);

%  Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%  Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%  Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;




