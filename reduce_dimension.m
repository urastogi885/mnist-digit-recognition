function weights = reduce_dimension(method, train_data, train_labels)
%Methods: 0:PCA, 1:LDA
if method == 0
    p = 9;
    COEFF = pca(train_data, 'NumComponents', p);
    weights = COEFF(:, 1:p);
elseif method == 1
    num_features = size(train_data, 2);
    labels = unique(train_labels);
    num_labels = length(labels);
    % mean
    mu = zeros(num_labels, num_features);
    for i=1:num_labels
        mu(i,:) = mean( train_data((train_labels==labels(i)),:) ); %row vector
    end
    mu_all = mean(train_data); %row vector

    % Within scatter matrix
    delta = 0.1;  %SW singularittrain_labels
    SW = zeros(num_features,num_features);
    for i=1:num_labels
        S = cov(train_data);
        if(det(S)==0)
            S = S + delta * eye(num_features);
        end
        SW  = SW + S;
    end

    if(det(SW)==0)
        disp('singular');
        pause;
    end

    % Between scatter matrix
    SB = zeros(num_features,num_features);
    for i=1:num_labels
       Ni = sum( train_labels==labels(i) );
       SB = SB + Ni * ( mu(i,:) - mu_all )' * ( mu(i,:) - mu_all );      
    end
    [weights,~] = eigs(SB,SW, num_labels-1);
else
    disp('invalid dimension reduction method selection');
    return;
end
end