
function [baseFolderPath] = LoadBaseDataFolder()

baseFolderPath = fileread('ConfigurationFiles/BaseDataFolder.txt');
baseFolderPath = strtrim(baseFolderPath); % To eliminate newline in Ubuntu

end