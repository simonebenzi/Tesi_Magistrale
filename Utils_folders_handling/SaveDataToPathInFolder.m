
function [] = SaveDataToPathInFolder(folderWhereToSave, pathToFile, data)

CheckWhetherFolderExistsOrCreateIt(folderWhereToSave)
save(pathToFile,'data')

return