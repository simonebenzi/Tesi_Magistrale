
function [mean_abn, std_abn, thresh] = FindMeanStdAndThreshForAbn(abn, stdTimes, repetitions, std_factor)

[abn] = CutArraOutOfMeanStdThresh(abn, stdTimes, repetitions);
mean_abn = mean(abn);
std_abn  = std(abn);
thresh   = mean_abn + std_factor*std_abn;

end