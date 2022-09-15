function [] = CheckWhetherFolderExistsAndDeleteIt(folder)

    if exist(folder, 'dir')
       rmdir(folder, 's')
    end

end