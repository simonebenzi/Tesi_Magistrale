
% This function calculates the overall transition matrix.
% CASE 1: giving the vocabulary structure as input.
% If the input is a structure, it brings back to
% 'CalculateTransitionMatrixGivenVocabulary'.
% It will give as output the network again, with the added transition 
% matrix field.
% CASE 2: giving the cluster assignments array as input.
% If the input is not a structure, it brings back to 
% 'CalculateTransitionMatrixGivenClusterAssignments'.
% It will give as output the transition matrix.
function [output] = CalculateTransitionMatrix (input)

if class(input) == "struct"
    output = CalculateTransitionMatrixGivenVocabulary(input);
else
    output = CalculateTransitionMatrixGivenClusterAssignments(input);
end

end