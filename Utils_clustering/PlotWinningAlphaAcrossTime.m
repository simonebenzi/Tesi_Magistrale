function [] = PlotWinningAlphaAcrossTime(lastValueToPrintForClusters, ...
    odometryVocabularyNew, clusterAssignments)

figure
lastValueToPrintForClusters = 2000;
winningCenters = odometryVocabularyNew.nodesMean(clusterAssignments, :);
scatter3(winningCenters(1:lastValueToPrintForClusters, 1), ...
    winningCenters(1:lastValueToPrintForClusters, 2), 1:1:lastValueToPrintForClusters);
scatter3(winningCenters(1:lastValueToPrintForClusters, 1), ...
    winningCenters(1:lastValueToPrintForClusters, 2), 1:1:lastValueToPrintForClusters);
title('Winning cluster across time for video')

end