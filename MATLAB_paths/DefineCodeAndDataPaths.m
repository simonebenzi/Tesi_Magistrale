
function [paths] = DefineCodeAndDataPaths ()

AddAdditionalNecessaryPaths();
baseFolderPath = LoadBaseDataFolder();
paths = DefinePathsToData(baseFolderPath);

end