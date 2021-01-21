%% Code to import training data from MNIST dataset
function accuracy = CNN(train_data, test_data, train_labels, ...
    test_labels, len_train, len_test)
%% Reshape the data
train_data = reshape((train_data),[28,28,1,len_train]);
train_labels = ((train_labels));


test_data_cnn = reshape((test_data),[28,28,1,len_test]);
test_labels_cnn = ((test_labels));

%% Code to generate 10 fold train data

train_images_fold=zeros(28,28,1,6000,10);
train_labels_fold=zeros(6000,1,10);

for i=(1:10)
train_images_fold(:,:,:,:,i)=train_data(:,:,:,(6000*(i-1))+1:6000*i);
train_labels_fold(:,1,i)=train_labels((6000*(i-1))+1:6000*i,1);
end

train_images_data=zeros(28,28,1,54000,10);
train_labels_data=zeros(54000,1,10);
cross_valid_images=zeros(28,28,1,6000,10);
cross_valid_labels=zeros(6000,1,10);

for i=1:10
    cross_valid_images(:,:,:,:,i)=train_images_fold(:,:,:,:,i);
    cross_valid_labels(:,:,i)=train_labels_fold(:,:,i);
    k=1;
    for j=(1:10)
        if (j~=i)
           train_images_data (:,:,:,(6000*(k-1))+1:6000*k,i)=train_images_fold(:,:,:,:,j);
           train_labels_data ((6000*(k-1))+1:6000*k,1,i)=train_labels_fold(:,:,j);
           k=k+1;
        end
    end
end

%% Code to generate the ANN layers

inputlayer = imageInputLayer([28 28 1],'DataAugmentation','none',...
    'Normalization','none','Name','input');

%%volume 28*28*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                Layer 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

convlayer1 = convolution2dLayer(4,32,'Stride',1,'Padding',0, ...
    'BiasLearnRateFactor',2,'NumChannels',1,...
    'WeightLearnRateFactor',2, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'Name','conv1');

%%volume 25*25*32

convlayer1.Weights = randn([4 4 1 32])*0.1;
convlayer1.Bias = randn([1 1 32])*0.1;

relulayer1 = reluLayer('Name','relu1');

localnormlayer1 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm1','Alpha',0.0001,'Beta',0.75,'K',2);

maxpoollayer1 = maxPooling2dLayer(3,'Stride',3,'Name','maxpool1','Padding',1);

%volume 9*9*32 

droplayer1 = dropoutLayer(0.35);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Layer 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

convlayer2 = convolution2dLayer(3,16,'Stride',1, 'Padding',0,...
    'BiasLearnRateFactor',1,'NumChannels',32,...
    'WeightLearnRateFactor',1, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'Name','conv2');

%7*7*16

convlayer2.Weights = randn([3 3 32 16])*0.0001;
convlayer2.Bias = randn([1 1 16])*0.00001;

relulayer2 = reluLayer('Name','relu2');

localnormlayer2 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm2','Alpha',0.0001,'Beta',0.75,'K',2);
    
droplayer2 = dropoutLayer(0.25);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Output Layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fullconnectlayer = fullyConnectedLayer(10,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','fullconnect1');
fullconnectlayer.Weights = randn([10 784])*0.0001;
fullconnectlayer.Bias = randn([10 1])*0.0001+1;

smlayer = softmaxLayer('Name','sml1');

coutputlayer = classificationLayer('Name','coutput');

 %% Code to make the network
layers =[inputlayer, convlayer1, relulayer1,localnormlayer1, ...
   maxpoollayer1, droplayer1,...
   convlayer2, relulayer2, localnormlayer2,droplayer2,...
   fullconnectlayer, smlayer, coutputlayer]; 

%% Code to define training parameters
options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.75,... 
      'LearnRateDropPeriod',1,'L2Regularization',0.0001,... 
      'MaxEpochs',16,'Momentum',0.9,'Shuffle','once',... 
      'MiniBatchSize',15,'Verbose',1,...
      'CheckpointPath','checkpoints','InitialLearnRate',0.043);

%% Train the ANN
trained_net = cell(10,1);
score = zeros(10,1);
for i=1:10
   train_im = train_images_data(:,:,:,:,i);
   train_lb = categorical(train_labels_data(:,:,i));
   cross_valid_im = cross_valid_images(:,:,:,:,i);
   cross_valid_lb = categorical(cross_valid_labels(:,:,i));
   trained_net(i) = trainNetwork(train_im, train_lb, layers, options);
   [Ypred,~] = classify(trained_net(i),cross_valid_im);
   score(i) = sum((Ypred==cross_valid_lb))/numel(cross_valid_lb);
end

%% Get the best model and test the accuracy of the network
[~,index] = max(score);
best_model=trained_net(index);

[Ypred, ~] = classify(best_model, test_data_cnn);
accuracy = 100 * sum( (Ypred == categorical(test_labels_cnn)) ) / ...
    numel(test_labels_cnn);

end