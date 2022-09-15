
% Note: each ROW is normalized.
function[dataNorm] = NormalizeAllValuesInCellArray(data, dataMin, dataMax)

% Number of rows and columns 
[nRows, nCols] = size(data);

dataNorm = data;

for i = 1:nRows
    for j = 1:nCols
        currentCell     = data{i,j};
        currentCellNorm = (currentCell - dataMin)./repmat(dataMax-dataMin,size(currentCell,1),1);
        dataNorm{i,j}   = currentCellNorm;
    end
end

end