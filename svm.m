function accuracy = svm(kernel, train_data, test_data, label_train, ...
    label_test, len_test)
%kernel 0:linear, 1:polynomial, 2:rbf
num_labels = 10;
model = cell(num_labels,1);

% Start training
for i = 1:num_labels
    if kernel == 0
        model{i} = svm_train(double(label_train==(i-1)), ...
            train_data', '-s 0 -t 0 -b 1 -h 0');
    elseif kernel == 1
        model{i} = svm_train(double(label_train==(i-1)), ...
            train_data', '-s 0 -t 1 -b 1 -h 0');
    elseif kernel == 2
        model{i} = svm_train(double(label_train==(i-1)), ...
            train_data', '-s 0 -t 2 -b 1 -h 0');
    else
        disp('Invalid kernel choice!');
        return;
    end
end

% Start testing
prob = zeros(length(label_test),num_labels);
for i = 1:num_labels
    disp(i);
    [~,~,p] = svm_predict(double(label_test==(i-1)), test_data', ...
        model{i}, '-b 1');
    prob(:,i) = p(:, model{i}.Label==1);
end

% Find predictions with maximum probabilities
[~,pred] = max(prob, [], 2);
accuracy = 100 * sum((pred-1) == label_test) / len_test;
end