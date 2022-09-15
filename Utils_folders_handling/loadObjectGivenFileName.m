
function [objectLoaded, isLoaded] = loadObjectGivenFileName(fileName)

    if exist(fileName, 'file')
        objectLoaded = load(fileName);
        objectLoaded = struct2cell(objectLoaded);
        objectLoaded = objectLoaded{1,1};
        isLoaded = true;
    else
        objectLoaded = false;
        isLoaded     = false;
    end

end