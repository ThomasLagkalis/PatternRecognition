function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    
    [NumSamples NumFeatures] = size(Samples);
	A=zeros(NumFeatures,NewDim);
    
	
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    %For each class i
	%Find the necessary statistics
    
    %Calculate the Class Prior Probability
    for i=1:NumClasses
        classi = find(Labels==Classes(i));
	    P(i)=length(classi)/NumLabels;
    end
    %Calculate the Class Mean 
    for i=1:NumClasses
	    classi = find(Labels==Classes(i));
        xi = Samples(classi, :);
        mu(i,:) = mean(xi);
    end
    
    %Calculate the Global Mean
	m0=mean(Samples);
    
    %Calculate the Within and Between Class Scatter Matrix
    Sw = zeros(NumFeatures, NumFeatures);
    Sb=zeros(NumFeatures, NumFeatures);
	for i=1:NumClasses
        classi = find(Labels==Classes(i));
        xi = Samples(classi, :);
        xi = xi-mu(i,:);
        Sw=Sw + xi'*xi;
        Sb=Sb + length(classi)*(mu(i,:)-m0)'*(mu(i,:)-m0);
    end
  
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [Q, L] = eig(EigMat);
    [junk, ind] = sort(diag(L), 'descend');
    W = Q(:,ind(1:NewDim));
    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
    A=A+W;

