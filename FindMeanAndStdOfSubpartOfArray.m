function [means, stds] = FindMeanAndStdOfSubpartOfArray(array, arrayParameters, uniqueParameters)

means = [];
stds = [];
for j = 1:length(uniqueParameters)
    selectedElements = array(arrayParameters == uniqueParameters(j));
    meanValue = mean(selectedElements);
    stdValue = std(selectedElements);
    means = [means, meanValue];
    stds = [stds, stdValue];
end

end