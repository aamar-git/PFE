%% Data Preparation

% Set random seed for reproducibility
rng('default');

% Define temporary directory for dataset
% % dataFolder = tempdir;
dataFolder = "C:\Users\Aamar\Documents\MATLAB\Examples\R2023b\supportfiles\";
% Define dataset path
dataset = fullfile(dataFolder, "google_speech");

% Function to augment dataset with background noise (defined later)
augmentDataset(dataset);

%% Load and Prepare Audio Data

% Load training data
adsTrain = audioDatastore(fullfile(dataset, "train"), ...
  'IncludeSubfolders', true, ...
  'FileExtensions', ".wav", ...
  'LabelSource', "foldernames");

% Define command categories
commands = categorical(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]);

% Define background category
background = categorical("background");

% Identify command, background, and unknown samples based on labels
isCommand = ismember(adsTrain.Labels, commands);
isBackground = ismember(adsTrain.Labels, background);
isUnknown = ~(isCommand | isBackground);

% Fraction of unknowns to include for training
includeFraction = 0.2;

% Randomly select unknown samples to include
idx = find(isUnknown);
idx = idx(randperm(numel(idx), round((1 - includeFraction) * sum(isUnknown))));
isUnknown(idx) = false;

% Assign unknown label to remaining unknown samples
adsTrain.Labels(isUnknown) = categorical("unknown");

% Select training data with commands, unknowns, and background
adsTrain = subset(adsTrain, isCommand | isUnknown | isBackground);

% Remove extra categories from labels
adsTrain.Labels = removecats(adsTrain.Labels);

% Load validation data (similar process as training data)
adsValidation = audioDatastore(fullfile(dataset, "validation"), ...
  'IncludeSubfolders', true, ...
  'FileExtensions', ".wav", ...
  'LabelSource', "foldernames");

isCommand = ismember(adsValidation.Labels, commands);
isBackground = ismember(adsValidation.Labels, background);
isUnknown = ~(isCommand | isBackground);
includeFraction = 0.2;
idx = find(isUnknown);
idx = idx(randperm(numel(idx), round((1 - includeFraction) * sum(isUnknown))));
isUnknown(idx) = false;
adsValidation.Labels(isUnknown) = categorical("unknown");
adsValidation = subset(adsValidation, isCommand | isUnknown | isBackground);
adsValidation.Labels = removecats(adsValidation.Labels);

% Visualize label distribution for training and validation data
figure('Units', 'normalized', 'Position', [0.2, 0.2, 0.5, 0.5]);
tiledlayout(2, 1);
nexttile;
histogram(adsTrain.Labels);
title('Training Label Distribution');
ylabel('Number of Observations');
grid on;
nexttile;
histogram(adsValidation.Labels);
title('Validation Label Distribution');
ylabel('Number of Observations');
grid on;

% Check for parallel pool and enable if available
if canUseParallelPool && ~speedupExample
  useParallel = true;
  gcp;
else
  useParallel = false;
end

%% Feature Extraction

% Define audio sample rate
fs = 16e3;

% Define segment and frame durations, hop length, and FFT length
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
FFTLength = 512;

% Calculate number of samples for segment, frame, and hop
numBands = 50;
segmentSamples = round(segmentDuration * fs);
frameSamples = round(frameDuration * fs);
hopSamples = round(hopDuration * fs);
overlapSamples = frameSamples - hopSamples;

% Create audio feature extractor object
afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);

% Set bark spectrum num bands (within setExtractorParameters)
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);

%% Feature Transformation
% Function to pad audio for segmentation (defined later)
transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);

% Function to extract features using audio feature extractor (defined later)
transform2 = transform(transform1,@(x)extract(afe,x));

% Function to apply log transformation (defined later)
transform3 = transform(transform2,@(x){log10(x+1e-6)});

% Apply feature transformation functions on training data (with parallelization)
XTrain = readall(transform3,UseParallel=useParallel);

% Combine transformed features from multiple files into a 4D tensor
XTrain = cat(4, XTrain{:});

% Get dimensions of transformed features
[numHops, numBands, numChannels, numFiles] = size(XTrain);

% Apply feature transformation functions on validation data (similar to training data)
transform1 = transform(adsValidation,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XValidation = readall(transform3,UseParallel=useParallel);
XValidation = cat(4, XValidation{:});

% Extract labels from training and validation data
TTrain = adsTrain.Labels;
TValidation = adsValidation.Labels;

%% Feature Normalization and Visualization

% Find minimum and maximum values across training features
specMin = min(XTrain, [], "all");
specMax = max(XTrain, [], "all");

% Select random samples for visualization
idx = randperm(numel(adsTrain.Files), 3);

% Create figure for visualization
figure('Units', 'normalized', 'Position', [0.2, 0.2, 0.6, 0.6]);
tlh = tiledlayout(2, 3);

for ii = 1:3
% Read audio data from file
[x, fs] = audioread(adsTrain.Files{idx(ii)});
colormap(hot)
% Plot raw waveform
nexttile(tlh, ii);
plot(x);
axis tight;
title(string(adsTrain.Labels(idx(ii))));

% Extract mel spectrogram from transformed features
spect = XTrain(:, :, 1, idx(ii))';

% Plot mel spectrogram with colormap and normalization
nexttile(tlh, ii + 3);
pcolor(spect);
clim([specMin specMax]);
shading flat;

% Play audio for reference while visualizing
sound(x, fs);
pause(2);
end

%% Deep Neural Network Architecture

% Define classes from training labels
classes = categories(TTrain);

% Calculate class weights to address imbalanced classes (optional)
classWeights = 1 ./ countcats(TTrain);
classWeights = classWeights' / mean(classWeights);

% Define number of classes
numClasses = numel(classes);

% Define time pool size for recurrent layer
timePoolSize = ceil(numHops / 8);

% Define hidden units, heads, and channels for layers
numHiddenUnits = 128;
numHeads = 8;
numKeyChannels = 64;

% Define dropout probability for regularization
dropoutProb = 0.2;

% Define number of filters for convolutional layers
numF = 20;

% Define layers for the deep neural network
layers = [
    imageInputLayer([numHops, afe.FeatureVectorLength])
    
    convolution2dLayer(3, numF, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, Stride=2, Padding="same")
    
    convolution2dLayer(3, 2 * numF, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, Stride=2, Padding="same")
    
    convolution2dLayer(3, 4 * numF, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, Stride=2, Padding="same")
    
    convolution2dLayer(3, 4 * numF, Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 4 * numF, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize, 1])

    flattenLayer
    lstmLayer(numHiddenUnits, "OutputMode", "sequence")

	fullyConnectedLayer(numClasses)
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(numClasses)
    reluLayer
    softmaxLayer('Name','classoutput')
	];
%% Network Training

% Define mini-batch size
miniBatchSize = 128;

% Define validation frequency
validationFrequency = floor(numel(TTrain) / miniBatchSize);

% Define training options with parameters
options = trainingOptions("adam", ...
  'InitialLearnRate', 3e-4, ...
  'MaxEpochs', 15, ...
  'MiniBatchSize', miniBatchSize, ...
  'Shuffle', 'every-epoch', ...
  'Plots', 'training-progress', ...
  'Verbose', false, ...
  'ValidationData', {XValidation, TValidation}, ...
  'ValidationFrequency', validationFrequency, ...
  'Metrics', ["accuracy", "fscore"], ...
  'L2Regularization', 0.003); % L2 regularization

% Train the deep neural network
trainedNet = trainnet(XTrain, TTrain, layers, @(Y, T) crossentropy(Y, T, classWeights(:), WeightsFormat="C"), options);

%% Network Evaluation

% Analyze the trained network for structure visualization (optional)
analyzeNetwork(trainedNet);

% Predict on validation data using mini-batch
scores = minibatchpredict(trainedNet, XValidation, miniBatchSize);
YValidation = scores2label(scores, classes, "auto");

% Calculate validation error
validationError = mean(YValidation ~= TValidation);

% Predict on training data using mini-batch (similar to validation data)
scores = minibatchpredict(trainedNet, XTrain, miniBatchSize);
YTrain = scores2label(scores, classes, "auto");
trainError = mean(YTrain ~= TTrain);

% Display training and validation errors
disp(["Training error: " + trainError * 100 + " %"; "Validation error: " + validationError * 100 + " %"]);

% Convert ground truth labels for validation data to categorical
TValidation = categorical(TValidation);
YValidation = categorical(YValidation);

% Plot confusion matrix for validation data
figure('Units', 'normalized', 'Position', [0.2, 0.2, 0.5, 0.5]);
cm = confusionchart(TValidation, YValidation, ...
    'Title', 'Confusion Matrix for Validation Data', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
sortClasses(cm, [commands, "unknown", "background"]);

% Measure prediction time on CPU for a single image (optional)
for ii = 1:100
    x = randn([numHops, numBands]);
    predictionTimer = tic;
    y = predict(trainedNet, x);
    time(ii) = toc(predictionTimer);
end

disp(["Network size: " + whos("trainedNet").bytes / 1024 + " kB"; ...
      "Single-image prediction time on CPU: " + mean(time(11:end)) * 1000 + " ms"]);

%% Testing on Unseen Data

% Load test data
adsTest = audioDatastore(fullfile(dataset, "test"), ...
    'IncludeSubfolders', true, ...
    'FileExtensions', ".wav", ...
    'LabelSource', "foldernames");

% Identify command, background, and unknown samples based on labels
isCommand = ismember(adsTest.Labels, commands);
isBackground = ismember(adsTest.Labels, background);
isUnknown = ~(isCommand | isBackground);
includeFraction = 0.2;

% Randomly select unknown samples to include
idx = find(isUnknown);
idx = idx(randperm(numel(idx), round((1 - includeFraction) * sum(isUnknown))));
isUnknown(idx) = false;

% Assign unknown label to remaining unknown samples
adsTest.Labels(isUnknown) = categorical("unknown");

% Select test data with commands, unknowns, and background
adsTest = subset(adsTest, isCommand | isUnknown | isBackground);

% Remove extra categories from labels
adsTest.Labels = removecats(adsTest.Labels);

% Visualize label distribution for test data
figure('Units', 'normalized', 'Position', [0.2, 0.2, 0.5, 0.5]);
tiledlayout(2, 1);
nexttile;
histogram(adsTest.Labels);
title('Testing Label Distribution');
ylabel('Number of Observations');
grid on;

% Apply feature transformation functions on test data (similar to training and validation)
transform1 = transform(adsTest,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XTest = readall(transform3,UseParallel=useParallel);
XTest = cat(4, XTest{:});

% Extract labels from test data
TTest = adsTest.Labels;

% Perform predictions on test data using mini-batch
scoresTest = minibatchpredict(trainedNet, XTest, miniBatchSize);
YTest = scores2label(scoresTest, classes, "auto");

% Convert ground truth labels for test data to categorical
TTest = categorical(TTest);
YTest = categorical(YTest);

% Calculate test error
testError = mean(YTest ~= TTest);

% Display test error
disp("Test error: " + testError * 100 + " %");

% Plot confusion matrix for test data
figure('Units', 'normalized', 'Position', [0.2, 0.2, 0.5, 0.5]);
cmTest = confusionchart(TTest, YTest, ...
    'Title', 'Confusion Matrix for Test Data', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
sortClasses(cmTest, [commands, "unknown", "background"]);

%% Export Network to ONNX

% Export the trained network to ONNX format for portability (optional)
exportONNXNetwork(trainedNet, "CNNLSTM-ONNX", OpsetVersion=13);

%% Export Network to TensorFlow

% Export the trained network to TensorFlow format for flexibility (optional)
exportNetworkToTensorFlow(trainedNet, "CNNLSTM-TF2");


%% Function to Augment Dataset with Background Noise
function augmentDataset(datasetloc)
adsBkg = audioDatastore(fullfile(datasetloc,"background"));
fs = 16e3; % Known sample rate of the data set
segmentDuration = 1;
segmentSamples = round(segmentDuration*fs);

volumeRange = log10([1e-4,1]);

numBkgSegments = 4000;
numBkgFiles = numel(adsBkg.Files);
numSegmentsPerFile = floor(numBkgSegments/numBkgFiles);

fpTrain = fullfile(datasetloc,"train","background");
fpValidation = fullfile(datasetloc,"validation","background");
fpTest = fullfile(datasetloc,"test","background");

if ~exist(fpTrain)

    % Create directories
    mkdir(fpTrain)
    mkdir(fpValidation)
    mkdir(fpTest)

    for backgroundFileIndex = 1:numel(adsBkg.Files)
        [bkgFile,fileInfo] = read(adsBkg);
        [~,fn] = fileparts(fileInfo.FileName);

        % Determine starting index of each segment
        segmentStart = randi(size(bkgFile,1)-segmentSamples,numSegmentsPerFile,1);

        % Determine gain of each clip
        gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numSegmentsPerFile,1) + volumeRange(1));

        for segmentIdx = 1:numSegmentsPerFile

            % Isolate the randomly chosen segment of data.
            bkgSegment = bkgFile(segmentStart(segmentIdx):segmentStart(segmentIdx)+segmentSamples-1);

            % Scale the segment by the specified gain.
            bkgSegment = bkgSegment*gain(segmentIdx);

            % Clip the audio between -1 and 1.
            bkgSegment = max(min(bkgSegment,1),-1);

            % Create a file name.
            afn = fn + "_segment" + segmentIdx + ".wav";

            % Randomly assign background segment to either the train or
            % validation set.
            if rand > 0.85 % Assign 15% to validation
                dirToWriteTo = fpValidation;
                augmentTest = tempdir;
            else % Assign 85% to train set.
                dirToWriteTo = fpTrain;
                augmentTest = fpTest;
            end

            % Write the audio to the file location.
            ffn = fullfile(dirToWriteTo,afn);
            audiowrite(ffn,bkgSegment,fs)
            ffn = fullfile(augmentTest,afn);
            audiowrite(ffn,bkgSegment,fs)

        end

        % Print progress
        fprintf('Progress = %d (%%)\n',round(100*progress(adsBkg)))

    end
end
end
%% Helper Functions

function paddedAudio = padForSegmentation(audio, segmentSamples)
  % This function pads an audio signal to a specified segment duration for segmentation.
  paddedAudio = [zeros(floor((segmentSamples - size(audio, 1)) / 2), 1); audio; zeros(ceil((segmentSamples - size(audio, 1)) / 2), 1)];
end

function labels = scores2label(scores, classes, method)
% Convert raw scores to class labels
% Inputs:
%   scores: Raw network output scores (matrix or array)
%   classes: Cell array of class labels

    if nargin < 3
        method = 'auto';
    end
    % Determine labels based on method
    switch lower(method)
        case 'auto'
            [~, idx] = max(scores, [], 2);
            labels = classes(idx);
        otherwise
            error('Invalid method: %s', method);
    end
end
function [predictions, labels] = minibatchpredict(net, data, miniBatchSize)
    % Initialize arrays to store predictions and labels
    predictions = [];
    labels = [];
    
    % Determine the number of data samples
    numSamples = size(data, 4);
    
    % Iterate through the data in mini-batches
    for i = 1:miniBatchSize:numSamples
        % Determine the indices of the current mini-batch
        idx = i:min(i+miniBatchSize-1, numSamples);
        
        % Extract the current mini-batch from the data
        miniBatchData = data(:,:,:,idx);
        
        % Perform predictions for the current mini-batch
        miniBatchPredictions = predict(net, miniBatchData);
        
        % Append predictions and labels to the arrays
        predictions = [predictions; miniBatchPredictions];
        labels = [labels; idx(:)];
    end
end