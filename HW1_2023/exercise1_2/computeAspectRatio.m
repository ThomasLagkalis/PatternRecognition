function aRatio = computeAspectRatio(image)
    [num_rows, num_cols] = size(image);
    nonzeros_rows = find(any(image, 2)); %find all the indicies of rows containing nonzero integers
    nonzeros_columns = find(any(image,1)); %same thing for columns
    height = size(nonzeros_columns, 2);
    width = size(nonzeros_rows, 1);
    aRatio=width/height;
end

