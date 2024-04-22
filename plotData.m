lossH = jsondecode(fileread(fullfile(pwd,'data\lossH.json')));
predictionsH = jsondecode(fileread(fullfile(pwd,'data\predictionsH.json')));
% lossK = jsondecode(fileread(fullfile(pwd,'data\lossK.json')));
epochCorrectValsK = jsondecode(fileread(fullfile(pwd,'data\predictionsK.json')));
p_Y = jsondecode(fileread(fullfile(pwd,'data\p_Y.json')));
epochs = jsondecode(fileread(fullfile(pwd,'data\epochs.json')));
datasetSize = size(p_Y,1);

x = 1:size(predictionsH,1);
y = repmat(p_Y, epochs, 1)';
absPredsH = round(predictionsH);
% absPredsK = round(predictionsK);
correctValsH = (absPredsH == y')*1;
% correctValsK = (absPredsK == y')*1;
countCorrectValsH = sum(correctValsH(:) == 1);
% countCorrectValsK = sum(correctValsK(:) == 1);

epochAccuracyH = arrayfun(@(i) sum(correctValsH(i:i+epochs-1)), ...
    1:datasetSize:length(correctValsH)-datasetSize+1)'/datasetSize;
epochAccuracyK = arrayfun(@(i) mean(epochCorrectValsK(i:i+epochs-1)), ...
     1:datasetSize:length(epochCorrectValsK)-datasetSize+1)'/datasetSize;

figure
plot(x, lossH,'b')
hold on
plot(x, lossK,'r')
xlabel('Iterations')
ylabel('Loss')
legend('Homemade','Keras')
hold off

figure
plot(x, predictionsH,'b')
hold on
plot(x, predictionsK,'r')
xlabel('Iterations')
ylabel('Predictions')
legend('Homemade','Keras')
hold off

figure
plot(1:epochs, epochAccuracyH,'b')
hold on
plot(1:epochs, epochAccuracyK,'r')
xlabel('Iterations')
ylabel('Accuracy')
legend('Homemade','Keras')
hold off

