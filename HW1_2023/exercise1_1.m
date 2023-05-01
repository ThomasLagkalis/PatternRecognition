clear all; 
 
%Defining the parameters for Gaussian
mu1 = [3 ,3];
mu2 = [6, 6];
sigma1 = [1.2 , -0.4; -0.4, 1.2];
sigma2 = [1.2, 0.4; 0.4, 1.2];
 
 %Defining a and b values (a.k.a features) for the plot.
a = linspace(-1, 10, 1000);
b = linspace(-1, 10, 1000); 
[X, Y] = meshgrid(a, b);

%Create the first 2D Gaussian.
Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
Z1 = reshape(Z1, size(X));
Z2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
Z2 = reshape(Z2, size(X));

%Plot the 3D Gaussian.
figure(1)
subplot(1,2,1)
surf(X, Y, Z1);
xlabel('x');
ylabel('y');
zlabel('z');
title('3D Gaussian Functions');
hold on;
surf(X, Y, Z2);
hold off;

%Boundaries
t=[0:0.01:10];
PC1 = 0.1;
y1 = (28.8 + 2.56*log(PC1/(1-PC1)))./(1.6.*t);
PC1 = 0.25;
y2 = (28.8 + 2.56*log(PC1/(1-PC1)))./(1.6.*t);
PC1 = 0.5;
y3 = (28.8 + 2.56*log(PC1/(1-PC1)))./(1.6.*t);
PC1 = 0.75;
y4 = (28.8 + 2.56*log(PC1/(1-PC1)))./(1.6.*t);
PC1 = 0.9;
y5 = (28.8 + 2.56*log(PC1/(1-PC1)))./(1.6.*t);


%Plot the procection of Gaussian functions onto x-y plane.
subplot(1,2,2)
plot(t, y1,'linewidth', 2, t, y2, 'linewidth', 2, t, y3, 'linewidth', 2, t, y4,'linewidth', 2, t, y5, 'linewidth', 2)
legend('Pc1=0.1','Pc1=0.25','Pc1=0.5','Pc1=0.75','Pc1=0.9')
hold on;
contour(X, Y, Z1, 20);
hold on;
contour(X, Y, Z2, 20);

xlabel('x');
ylabel('y');
title('Procections of Gaussian functions onto x-y plane');

%% For Sigma1=Sigma2

%Defining the parameters for Gaussian
mu1 = [3 ,3];
mu2 = [6, 6];
sigma1 = [1.2, 0.4; 0.4, 1.2];
sigma2 = [1.2, 0.4; 0.4, 1.2];
 
 %Defining a and b values (a.k.a features) for the plot.
a = linspace(-1, 10, 1000);
b = linspace(-1, 10, 1000); 
[X, Y] = meshgrid(a, b);

%Create the first 2D Gaussian.
Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
Z1 = reshape(Z1, size(X));
Z2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
Z2 = reshape(Z2, size(X));

%Plot the 3D Gaussian.
figure(2)
subplot(1,2,1)
surf(X, Y, Z1);
xlabel('x');
ylabel('y');
zlabel('z');
title('3D Gaussian Functions');
hold on;
surf(X, Y, Z2);
hold off;

%Boundaries
t=[0:0.01:10];
PC1 = 0.1;
y1 = -t+9+1.066*log(PC1/(1-PC1));
PC1 = 0.25;
y2 = -t+9+1.066*log(PC1/(1-PC1));
PC1 = 0.5;
y3 = -t+9+1.066*log(PC1/(1-PC1));
PC1 = 0.75;
y4 = -t+9+1.066*log(PC1/(1-PC1));
PC1 = 0.9;
y5 = -t+9+1.066*log(PC1/(1-PC1));


%Plot the procection of Gaussian functions onto x-y plane.
subplot(1,2,2)
plot(t, y1,'linewidth', 2, t, y2, 'linewidth', 2, t, y3, 'linewidth', 2, t, y4,'linewidth', 2, t, y5, 'linewidth', 2)
legend('Pc1=0.1','Pc1=0.25','Pc1=0.5','Pc1=0.75','Pc1=0.9')
hold on;
contour(X, Y, Z1, 20);
hold on;
contour(X, Y, Z2, 20);
xlabel('x');
ylabel('y');
title('Procections of Gaussian functions onto x-y plane');
hold off;

