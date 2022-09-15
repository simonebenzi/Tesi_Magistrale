
% Function to denormalize the data knowing original data maximum and
% minimum.
% INPUTS:
% - denormalizeArrayOfSequences: normalized array of sequences;
%             shape: [num of time instants in sequence, number of sequences, number of params]
% - realMins: minimum;
% - realMaxs: maximum;
% OUTPUTS:
% - denormalizedSequence: denormalized array of sequences.
%             shape: [num of time instants in sequence, number of sequences, number of params]

function [denormalizedArrayOfSequences] = denormalizeArrayOfSequences(normalizedArrayOfSequences, realMins, realMaxs)

    % Create array over sequence
    denormalizedArrayOfSequences = zeros(size(normalizedArrayOfSequences,1), ...
        size(normalizedArrayOfSequences,2), size(normalizedArrayOfSequences,3));

    % Loop over the different sequences
    for i = 1:size(normalizedArrayOfSequences, 2)
        % Normalize each sequence
        [denormalizedArrayOfSequences(:,i,:)] = denormalizeSequence( ...
            squeeze(normalizedArrayOfSequences(:,i,:)), realMins, realMaxs);
    end

end