
function [] = CheckWhetherFolderExistsOrCreateIt(folder)

if ~exist(folder, 'dir')
   mkdir(folder)
end

end