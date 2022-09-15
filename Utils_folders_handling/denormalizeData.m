
% Function to denormalize the data knowing original data maximum and
% minimum.
% INPUTS:
% - xNormalized: normalized data;
% - xMin: minimum;
% - xMax: maximum;
% OUTPUTS:
% - xDenormalized: denormalized data
function [xDenormalized] = denormalizeData(xNormalized, xMin, xMax)

    xDenormalized = xNormalized.*(xMax - xMin) + xMin;

end