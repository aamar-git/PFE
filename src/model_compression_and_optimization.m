%% Taylor Pruning

% Calculate validation and training accuracy
validationAccuracy = mean(YValidation == TValidation);
trainAccuracy = mean(YTrain == TTrain);

% Display accuracy
disp(["Training Accuracy: " + trainAccuracy*100 + "%";"Validation Accuracy: " + validationAccuracy*100 + "%"])

% Create a figure for confusion matrix
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);

% Generate confusion matrix with labels and title
cm = confusionchart(TValidation,YValidation, ...
  'Title','Original Network', ...
  'ColumnSummary','column-normalized','RowSummary','row-normalized');

% Sort confusion matrix classes in a specific order
sortClasses(cm,[commands,"unknown","background"])
%%

% Create datastores for training and validation with data augmentation
dsTrain = augmentedImageDatastore([98 50], XTrain, TTrain);
dsValidation = augmentedImageDatastore([98 50], XValidation, TValidation);

% Set mini-batch size
miniBatchSize = 50;

% Choose execution environment based on available hardware
executionEnvironment = "auto";

% Create mini-batch queues with preprocessing function and specific format
mbqTrain = minibatchqueue(dsTrain, ...
  MiniBatchSize=miniBatchSize, ...
  MiniBatchFcn=@preprocessMiniBatch,...
  MiniBatchFormat = ["SSCB",""],...
  PartialMiniBatch="discard");

mbqValidation = minibatchqueue(dsValidation, ...
  MiniBatchSize=miniBatchSize, ...
  MiniBatchFcn=@preprocessMiniBatch,...
  MiniBatchFormat = ["SSCB",""],...
  PartialMiniBatch="discard");

%%

% Convert the trained network to layer graph and dlnetwork
lgraph = layerGraph(trainedNet);
dlnet = dlnetwork(lgraph);

%%

% Create a prunable network from the dlnetwork
prunableNet = taylorPrunableNetwork(dlnet);

% Set parameters for pruning iterations
maxPruningIterations = 16; 
maxToPrune = 8; 
maxPrunableFilters = prunableNet.NumPrunables;
numTest = size(TValidation,1);
minPrunables = 5;

% Learning rate, momentum and other training parameters
learnRate = 1e-2;
momentum = 0.9;
numMinibatchUpdates = 50;
validationFrequency = 1;

%%

% Perform Taylor pruning loop with validation and progress plots
prunableNet = taylorPruningLoop(prunableNet, mbqTrain, mbqValidation, classes, numTest, maxPruningIterations, ...
                maxPrunableFilters, maxToPrune, minPrunables, learnRate, ...
                momentum, numMinibatchUpdates, validationFrequency,trainAccuracy);

%%

% Reassemble the pruned network with classification layer
prunedLayerGraph = reassembleTaylorNetwork(prunableNet, classes);

%%

% Set mini-batch size and validation frequency for fine-tuning
miniBatchSize = 128;
validationFrequency = floor(numel(TTrain)/miniBatchSize);

% Training options for fine-tuning the pruned network
options = trainingOptions("sgdm", ...
  InitialLearnRate=3e-4, ...
  MaxEpochs=15, ...
  MiniBatchSize=miniBatchSize, ...
  Shuffle="every-epoch", ...
  Plots="training-progress", ...
  Verbose=false, ...
  ValidationData={XValidation,TValidation}, ...
  OutputNetwork="best-validation-loss",...
  ValidationFrequency=validationFrequency);

% Fine-tune the pruned network
trainedNetPruned = trainNetwork(XTrain,TTrain,prunedLayerGraph,options);

%%

% Classify validation and training data with the fine-tuned network
YValidation = classify(trainedNetPruned,XValidation);
validationAccuracy = mean(YValidation == TValidation);
YTrain = classify(trainedNetPruned,XTrain);
trainAccuracy = mean(YTrain == TTrain);

% Display accuracy
disp(["Training Accuracy: " + trainAccuracy*100 + "%";"Validation Accuracy: " + validationAccuracy*100 + "%"])

% Create confusion matrix for the pruned network
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(TValidation,YValidation, ...
  'Title','Pruned Network', ...

%% Neural Net Quantization

% Create quantization object with execution environment
dlquantObj = dlquantizer(trainedNetPruned,ExecutionEnvironment='GPU');

% Generate calibration data
calData = createCalibrationSet(XTrain, TTrain, 36, ["yes","no","up","down","left","right","on","off","stop","go","unknown","background"]);

% Calibrate the quantization object with calibration data
calibrate(dlquantObj, calData);

%%

% Quantize the network with specified scheme
qnetPruned = quantize(dlquantObj,'ExponentScheme','Histogram');

% Save the quantized network
save("qnet","qnetPruned")

% Get details of the quantized network
qDetails = quantizationDetails(qnetPruned)

%%

% Classify validation and training data with the quantized network
YValidation = classify(qnetPruned,XValidation);
validationAccuracy = mean(YValidation == TValidation);
YTrain = classify(qnetPruned,XTrain);
trainAccuracy = mean(YTrain == TTrain);

% Display accuracy
disp(["Training Accuracy: " + trainAccuracy*100 + "%";"Validation Accuracy: " + validationAccuracy*100 + "%"])

% Create confusion matrix for the quantized network
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(TValidation,YValidation, ...
  'Title','Pruned and Quantized Network', ...
  'ColumnSummary','column-normalized','RowSummary','row-normalized');
sortClasses(cm,[commands,"unknown","background"])

%%

% Estimate network metrics for original, Taylor pruned and quantized networks
originalNetMetrics = estimateNetworkMetrics(trainedNet);
taylorNetMetrics = estimateNetworkMetrics(trainedNetPruned);
quantizedNetMetrics = estimateNetworkMetrics(qnetPruned);

% Plot the number of learnable parameters
figure
learnables = [sum(originalNetMetrics.NumberOfLearnables)
    sum(taylorNetMetrics.NumberOfLearnables)
    sum(quantizedNetMetrics.NumberOfLearnables)];

x = categorical({'Original','Taylor Pruned','Quantized'});
x = reordercats(x, string(x));
plotResults(x, learnables)
ylabel("Number of Learnables")
title("Number of Learnables in Network")

% Plot parameter memory usage
figure;
memory = [sum(originalNetMetrics.("ParameterMemory (MB)"))
    sum(taylorNetMetrics.("ParameterMemory (MB)"))
    sum(quantizedNetMetrics.("ParameterMemory (MB)"))];
  
plotResults(x, memory)
ylabel("Parameter Memory (MB)")
title("Parameter Memory of Network")

%% Export Network to ONNX

% Export the trained network to ONNX format for portability (optional)
exportONNXNetwork(qnetPruned, "quanetPrunedCNNLSTMONNX", OpsetVersion=13);
exportONNXNetwork(trainedNetPruned, "prunedCNNLSTM-ONNX", OpsetVersion=13);

%% Export Network to TensorFlow

% Export the trained network to TensorFlow format for flexibility (optional)
exportNetworkToTensorFlow(qnetPruned, "qnetPrunedCNNLSTMONNX");
exportNetworkToTensorFlow(trainedNetPruned, "prunCNNLSTM-ONNX");

%% Helper Functions

function [X, Y] = preprocessMiniBatch(XCell, YCell)

  X = cat(4, XCell{:});

  Y = cat(2, YCell{:});

  % One-hot encode labels.
  Y = onehotencode(Y, 1);

end

%% Helper Functions
function [X, Y] = preprocessMiniBatch(XCell, YCell)

 X = cat(4, XCell{:});

 Y = cat(2, YCell{:});

 % One-hot encode labels.
 Y = onehotencode(Y, 1);

end

function prunableNet = taylorPruningLoop(prunableNet, mbqTrain, mbqValidation, classes, ...
  numTest, maxPruningIterations, maxPrunableFilters, maxToPrune, minPrunables, learnRate, ...
  momentum, numMinibatchUpdates, validationFrequency,trainAccuracy)
  %Initialize plots used and perform Taylor pruning with custom loop

  accuracyOfOriginalNet = trainAccuracy*100;

  %Initialize Progress Plots
  figure("Position",[10,10,700,700])
  tl = tiledlayout(3,1);
  lossAx = nexttile;
  lineLossFinetune = animatedline(Color=[0.85 0.325 0.098]);
  ylim([0 inf])
  xlabel("Fine-Tuning Iteration")
  ylabel("Loss")
  grid on
  title("Mini-Batch Loss During Pruning")
  xTickPos = [];

  accuracyAx = nexttile;
  lineAccuracyPruning = animatedline(Color=[0.098 0.325 0.85],LineWidth=2,Marker="o");
  ylim([50 100])
  xlabel("Pruning Iteration")
  ylabel("Accuracy")
  grid on
  addpoints(lineAccuracyPruning,0,accuracyOfOriginalNet)
  title("Validation Accuracy After Pruning")

  numPrunablesAx = nexttile;
  lineNumPrunables = animatedline(Color=[0.4660 0.6740 0.1880],LineWidth=2,Marker="^");
  ylim([0 maxPrunableFilters])
  xlabel("Pruning Iteration")
  ylabel("Prunable Filters")
  grid on
  addpoints(lineNumPrunables,0,double(maxPrunableFilters))
  title("Number of Prunable Convolution Filters After Pruning")

  start = tic;
  iteration = 0;

  for pruningIteration = 1:maxPruningIterations

    % Shuffle data.
    shuffle(mbqTrain);

    % Reset the velocity parameter for the SGDM solver in every pruning
    % iteration.
    velocity = [];

    % Loop over mini-batches.
    fineTuningIteration = 0;
    while hasdata(mbqTrain)
      iteration = iteration + 1;
      fineTuningIteration = fineTuningIteration + 1;

      % Read mini-batch of data.
      [X, T] = next(mbqTrain);

      % Evaluate the pruning activations, gradients of the pruning
      % activations, model gradients, state, and loss using the dlfeval and
      % modelLossPruning functions.
      [loss,pruningActivations, pruningGradients, netGradients, state] = ...
        dlfeval(@modelLossPruning, prunableNet, X, T);

      % Update the network state.
      prunableNet.State = state;

      % Update the network parameters using the SGDM optimizer.
      [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, velocity, learnRate, momentum);

      % Compute first-order Taylor scores and accumulate the score across
      % previous mini-batches of data.
      prunableNet = updateScore(prunableNet, pruningActivations, pruningGradients);

      % Display the training progress.
      D = duration(0,0,toc(start),Format="hh:mm:ss");
      addpoints(lineLossFinetune, iteration, double(loss))
      title(tl,"Processing Pruning Iteration: " + pruningIteration + " of " + maxPruningIterations + ...
        ", Elapsed Time: " + string(D))
      % Synchronize the x-axis of the accuracy and numPrunables plots with the loss plot.
      xlim(accuracyAx,lossAx.XLim)
      xlim(numPrunablesAx,lossAx.XLim)
      drawnow

      % Stop the fine-tuning loop when numMinibatchUpdates is reached.
      if (fineTuningIteration > numMinibatchUpdates)
        break
      end
    end

    % Prune filters based on previously computed Taylor scores.
    prunableNet = updatePrunables(prunableNet, MaxToPrune = maxToPrune);

    % Show results on the validation data set in a subset of pruning iterations.
    isLastPruningIteration = pruningIteration == maxPruningIterations;
    if (mod(pruningIteration, validationFrequency) == 0 || isLastPruningIteration)
      accuracy = modelAccuracy(prunableNet, mbqValidation, classes, numTest);
      addpoints(lineAccuracyPruning, iteration, accuracy)
      addpoints(lineNumPrunables,iteration,double(prunableNet.NumPrunables))
    end
   
    % Set x-axis tick values at the end of each pruning iteration.
    xTickPos = [xTickPos, iteration]; %#ok<AGROW>
    xticks(lossAx,xTickPos)
    xticks(accuracyAx,[0,xTickPos])
    xticks(numPrunablesAx,[0,xTickPos])
    xticklabels(accuracyAx,["Unpruned",string(1:pruningIteration)])
    xticklabels(numPrunablesAx,["Unpruned",string(1:pruningIteration)])
    drawnow

    % Break if number of prunables is less than parameter
    if (prunableNet.NumPrunables < minPrunables)
      break
    end

  end
end

function [loss,pruningGradient,pruningActivations,netGradients,state] = modelLossPruning(prunableNet, X, Y)

  %Forward pass
  [pred,state,pruningActivations] = forward(prunableNet,X);

  %Compute cross-entropy
  loss = crossentropy(pred,Y);

  [pruningGradient,netGradients] = dlgradient(loss,pruningActivations,prunableNet.Learnables);
end

function accuracy = modelAccuracy(net, mbq, classes, numObservations)

  totalCorrect = 0;
  reset(mbq);

  while hasdata(mbq)
    [dlX, Y] = next(mbq);

    dlYPred = extractdata(predict(net, dlX));

    YPred = onehotdecode(dlYPred,classes,1);
    YReal = onehotdecode(Y,classes,1);

    miniBatchCorrect = nnz(YPred == YReal);

    totalCorrect = totalCorrect + miniBatchCorrect;
  end

  accuracy = totalCorrect / numObservations * 100;
end

function prunedLayerGraph = reassembleTaylorNetwork(prunableNet, classes)
   
  prunedNet = dlnetwork(prunableNet);
  prunedLayerGraph = layerGraph(prunedNet);

  %add classification layer from classes defined in training data
  lgraphUpdated = addLayers(prunedLayerGraph, classificationLayer(Classes=classes));
  prunedLayerGraph = connectLayers(lgraphUpdated,prunedNet.OutputNames{1},string(lgraphUpdated.OutputNames{1}));

end

function XCalibration = createCalibrationSet(XTrain, TTrain, n, labels)
XCalibration = [];
for i=1:numel(labels)
  %Find logical index of label
  idx = (TTrain == labels(i));
  %Create subset data corresponding to logical indices
  label_subset = XTrain(:,:,:,idx);
  first_n_labels = label_subset(:,:,:,1:n);
  %Concatenate
  XCalibration = cat(4, XCalibration, first_n_labels);
end
end

function plotResults(x, data)
  b = bar(x, data);
  b.FaceColor = 'flat';
  b.CData(1, :) = [0 0.9 1];
  b.CData(2, :) = [0 0.8 0.8];
  b.CData(3, :) = [0.8 0 0.8];
end