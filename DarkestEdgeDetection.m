function max_points = darkest_edge_detection(img, n)
    if nargin < 2
        n = 6;
    end

    [rows, cols] = size(img);
    max_points = []; % 存储每列中高斯拟合的最大值点坐标

    % 对每一列进行处理
    for col = 1:cols
        column_values = img(:, col);
        [~, min_val_index] = min(column_values);

        start_index = max(min_val_index - floor(n / 2), 1);
        end_index = min(min_val_index + floor(n / 2), rows);

        y_values = 255 - column_values(start_index:end_index);
        x_values = start_index:end_index;

        % 高斯拟合
        try
            fit_func = @(b,x) b(1) * exp(-((x - b(2)).^2) / (2 * b(3)^2));
            initial_guess = [255, min_val_index, 1];
            options = optimset('Display','off');
            [b, ~] = lsqcurvefit(fit_func, initial_guess, x_values, double(y_values), [], [], options);
            max_points = [max_points; col, b(2)]; % 存储亚像素精度的y坐标
        catch
            % 如果拟合失败，继续处理下一列
            continue
        end
    end
end
