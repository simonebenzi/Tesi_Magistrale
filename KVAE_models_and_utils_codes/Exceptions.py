
# This is a class of custom exceptions

###############################################################################
# For ClusteredDataExtractor:
    
class ClusterDataExtractorUnmatchedLengthsError(Exception):

    def __init__(self, dataHolderLength, clusteringLength):
        self.dataHolderLength = dataHolderLength
        self.clusteringLength = clusteringLength
    
    def __str__(self):
        message = 'The length of the datapoints in the dataHolderNoSequence object ' + \
                  '( ' + str(self.dataHolderLength) + ' ) ' + \
                  'is different from the length of values assigned to clusters ' + \
                  'in the clusteringGraph object ' + \
                  '( ' + str(self.clusteringLength) + ' ) '
                  
        return message
            
class ClusterDataExtractorClusterNotPresent(Exception):

    def __init__(self, chosenCluster, numberOfClusters):
        self.chosenCluster    = chosenCluster
        self.numberOfClusters = numberOfClusters
    
    def __str__(self):
        message = 'Trying to access cluster number ' + str(self.chosenCluster ) + \
                  ' (numbering starting from 0) ' + \
                  ' but the graph only holds ' + str(self.numberOfClusters) + ' clusters '
                  
        return message