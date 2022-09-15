
function [array] = CutArraOutOfMeanStdThresh(array, stdTimes, repetitions)

for i = 1:repetitions
    
    abn_mean = mean(array);
    abn_std  = std(array);

    above_thresh = abn_mean + stdTimes*abn_std;
    below_thresh = abn_mean - stdTimes*abn_std;

    array(array>above_thresh) = above_thresh;
    array(array<below_thresh) = below_thresh;

end

end