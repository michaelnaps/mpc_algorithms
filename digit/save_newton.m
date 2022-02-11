function [TF] = save_newton(filename, data)
    fileID = fopen(filename, 'a+');

    if (~fileID)
        TF = 0;
        return;
    end

    for i = 1:length(data)-1
        fprintf(fileID, "%.10f, ", data(i));
    end
    fprintf(fileID, "%.10f\n", data(end));

    fclose(fileID);
    TF = 1;
    return;
end