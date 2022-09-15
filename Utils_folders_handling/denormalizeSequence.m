
% Function to denormalize the data knowing original data maximum and
% minimum.
% INPUTS:
% - normalizedSequence: normalized sequence;
%             shape: [num of time instants in sequence, number of params]
% - realMins: minimum;
% - realMaxs: maximum;
% OUTPUTS:
% - denormalizedSequence: denormalized sequence.
%             shape: [num of time instants in sequence, number of params]
function [denormalizedSequence] = denormalizeSequence(normalizedSequence, realMins, realMaxs)

    % Create array over sequence
    denormalizedSequence           = zeros(size(normalizedSequence,1),size(normalizedSequence,2));
    % Loop over sequence
    for i = 1:size(normalizedSequence, 1)
        % Take current value of the sequence
        currentValue               = normalizedSequence(i,:);
        % Denormalize the value
        denormalizedSequence(i,:)  = denormalizeData(currentValue, realMins, realMaxs);
    end

end