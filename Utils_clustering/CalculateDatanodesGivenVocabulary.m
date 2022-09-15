function [net] = CalculateDatanodesGivenVocabulary(net, data)

datanodes = cell(1,net.N);
for i = 1:net.N
    datanodes{1,i}   = [];
end
% Extract datanodes
for i = 1:size(data, 1)
    currentCluster   = net.dataColorNode(i);
    currentDatapoint = data(i,:);
    datanodes{1,currentCluster}   = [datanodes{1,currentCluster}; currentDatapoint];
end
net.datanodes = datanodes;

end