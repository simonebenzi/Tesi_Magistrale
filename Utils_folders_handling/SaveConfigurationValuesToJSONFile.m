
function [] = SaveConfigurationValuesToJSONFile(configurationFilePath, ...
    parametersNames, parametersValues)

% Read the configuration file
jsonText = fileread(configurationFilePath);
jsonData = jsondecode(jsonText);
% Write the parameters in the configuration file
for indexOfParameter = 1:size(parametersNames,1)
    nameOfParameter = parametersNames(indexOfParameter,:);
    nameOfParameter = erase(nameOfParameter, ' ');
    if ischar(parametersValues)
        valueOfParameter = parametersValues(indexOfParameter,:);
        valueOfParameter = erase(valueOfParameter, ' ');
    else
        valueOfParameter = parametersValues(indexOfParameter);
    end
    jsonData = setfield(jsonData,nameOfParameter,valueOfParameter);
end
% Save the json file
% Convert to JSON text
jsonTextToSave = jsonencode(jsonData, "PrettyPrint", true);
% Write to a json file
fid = fopen(configurationFilePath, 'w');
fprintf(fid, '%s', jsonTextToSave);
fclose(fid);

end