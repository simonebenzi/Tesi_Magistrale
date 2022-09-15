function [] = PlotVocabularyErrorEllipses(net)

for clusterIndex = 1:net.N
    hold on
    PlotSingleClusterEllipseFromVocabulary(net, clusterIndex)
end

end