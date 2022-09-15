
% Note: data in each cell is along ROWS.
% So output is one value for each COLUMN.
function [dataMin, dataMax] = LookForMinAndMaxValuesInCellArray(data)

% Number of rows and columns 
[nRows, nCols] = size(data);
% Number of dimensions
nDimensions    = size(data{1,1},2);

% Set min and max
dataMinValue = 1000000;
dataMaxValue = -dataMinValue;
dataMin = ones(1,nDimensions)*dataMinValue;
dataMax = ones(1,nDimensions)*dataMaxValue;

for i = 1:nRows
    for j = 1:nCols
        % Select current cell and find min and max of it for each column
        currentCell = data{i,j};
        currentDataMin = min(currentCell,[],1);
        currentDataMax = max(currentCell,[],1);
        % Is this the min/max value until now?
        whereisLessThanMin = currentDataMin < dataMin;
        whereisMoreThanMax = currentDataMax > dataMax;
        dataMin(whereisLessThanMin) = currentDataMin(whereisLessThanMin);
        dataMax(whereisMoreThanMax) = currentDataMax(whereisMoreThanMax);
    end
end

end