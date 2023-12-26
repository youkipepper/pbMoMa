import numpy as np
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import os
from numpy.polynomial.legendre import Legendre


def gauss_fit(x_values, y_values):
    def gauss(x, a, b, c):
        return a * np.exp(-((x - b) ** 2) / (2 * c**2))

    try:
        params, _ = curve_fit(
            gauss, x_values, y_values, p0=[max(y_values), np.mean(x_values), 1]
        )
        return gauss, params
    except RuntimeError:
        return None, None


def poly_fit(x_values, y_values, degree=6):
    try:
        params = np.polyfit(x_values, y_values, degree)
        return np.poly1d(params)
    except RuntimeError:
        return None


def orthogonal_poly_fit(x_values, y_values, degree=6):
    try:
        # 标准化 x 值到 [-1, 1] 区间
        x_norm = 2 * (x_values - min(x_values)) / (max(x_values) - min(x_values)) - 1
        leg = Legendre.fit(x_norm, y_values, degree)
        return leg
    except RuntimeError:
        return None


def edge_plot(
    column_values,
    start_index,
    end_index,
    min_val_index,
    col,
    fit_type="gauss",
    figure_size=(5, 5),
    dpi=100,
):
    x_values = np.arange(start_index, end_index)
    y_values = 255 - column_values[start_index:end_index]

    # 根据拟合类型选择拟合函数
    if fit_type == "gauss":
        fit_func, params = gauss_fit(x_values, y_values)
        if params is None:
            return None, False
        fitted_func = lambda x: fit_func(x, *params)
        b = params[1]
        fitted_point_y = fit_func(b, *params)

    elif fit_type == "poly":
        fit_func = poly_fit(x_values, y_values)
        if fit_func is None:
            return None, False
        fitted_func = fit_func
        # b = None
        # fitted_point_y = None
        # 计算拟合曲线上的最高点
        x_values_dense = np.linspace(start_index, end_index, 300)
        fitted_points = fitted_func(x_values_dense)
        max_point_index = np.argmax(fitted_points)
        # b = x_values_dense[max_point_index]
        # fitted_point_y = fitted_points[max_point_index]
        # 计算拟合曲线的导数并找到零点（极值点）
        derivative = np.polyder(fit_func)
        roots = np.roots(derivative)
        # 过滤非实数根和不在范围内的根
        real_roots = roots[
            np.isreal(roots) & (roots >= start_index) & (roots <= end_index)
        ].real
        if real_roots.size == 0:
            return None, False

        # 选择最大的极值点
        extreme_values = fitted_func(real_roots)
        max_index = np.argmax(extreme_values)
        b = real_roots[max_index]
        fitted_point_y = extreme_values[max_index]

    elif fit_type == "orthogonal_poly":
        leg = orthogonal_poly_fit(x_values, y_values)
        if leg is None:
            return None, False

        # 使用 Legendre 对象的 deriv 方法计算导数
        derivative = leg.deriv()
        roots = derivative.roots()

        # 转换 roots 到原始 x 值区间的函数
        def convert_to_original_x(x_normalized):
            return (
                x_normalized * (max(x_values) - min(x_values)) / 2
                + (max(x_values) + min(x_values)) / 2
            )

        # 定义 fitted_func 以接受原始 x 值
        def fitted_func(x):
            x_norm = 2 * (x - min(x_values)) / (max(x_values) - min(x_values)) - 1
            return leg(x_norm)

        # 转换 roots 到原始 x 值区间
        roots_original = convert_to_original_x(roots)

        # 过滤非实数根和不在范围内的根
        real_roots = roots_original[
            np.isreal(roots_original)
            & (roots_original >= min(x_values))
            & (roots_original <= max(x_values))
        ].real
        if real_roots.size == 0:
            # 没有找到实数根，选择曲线的最高点作为备选
            x_values_dense = np.linspace(min(x_values), max(x_values), 300)
            fitted_points = [
                leg(2 * (x - min(x_values)) / (max(x_values) - min(x_values)) - 1)
                for x in x_values_dense
            ]
            max_point_index = np.argmax(fitted_points)
            b = x_values_dense[max_point_index]
            fitted_point_y = fitted_points[max_point_index]
        else:
            # 选择最大的极值点
            extreme_values = [
                leg(2 * (x - min(x_values)) / (max(x_values) - min(x_values)) - 1)
                for x in real_roots
            ]
            max_index = np.argmax(extreme_values)
            b = real_roots[max_index]
            fitted_point_y = extreme_values[max_index]

    else:
        raise ValueError("Invalid fit_type. Choose 'gauss' or 'poly'.")

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(x_values, y_values, "bo", label="Original Data")

    x_values_dense = np.linspace(start_index, end_index, 300)
    fitted_curve_dense = fitted_func(x_values_dense)
    ax.plot(x_values_dense, fitted_curve_dense, "r-", label="Fitted Curve")

    ax.axvline(
        x=min_val_index, color="g", linestyle="--", label="Original Darkest Point"
    )

    is_valid = False
    if b is not None:
        is_valid = fitted_point_y >= max(y_values)
        point_color = "g" if is_valid else "r"
        ax.scatter(b, fitted_point_y, color=point_color, zorder=5)
        ax.annotate(
            f"({b:.2f}, {fitted_point_y:.2f})",
            (b, fitted_point_y),
            textcoords="offset points",
            xytext=(10, -10),
            ha="center",
        )

    # ax.set_title(f'Fit for Column {col} ({fit_type})')
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Inverted Pixel Intensity")
    ax.legend()

    plt.close(fig)
    return fig, is_valid


def create_video(
    img, fit_type="gauss", fps=25, preview=False, start_indices=None, end_indices=None
):
    cols = img.shape[1]
    rows, _ = img.shape

    figure_size = (5, 5)
    dpi = 100
    video_size = (int(figure_size[0] * dpi), int(figure_size[1] * dpi))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(f"{fit_type}_fit_video.mp4", fourcc, fps, video_size)

    valid_points = 0  # 初始化有效点的数量
    total_points = 0  # 初始化总点数

    for col in range(cols):
        column_values = img[:, col]
        min_val_index = np.argmin(column_values)
        start_index = (
            start_indices[col]
            if start_indices is not None
            else max(min_val_index - 3, 0)
        )
        end_index = (
            end_indices[col]
            if end_indices is not None
            else min(min_val_index + 4, rows)
        )

        fig, is_valid = edge_plot(
            column_values,
            start_index,
            end_index,
            min_val_index,
            col,
            fit_type,
            figure_size,
            dpi,
        )
        if fig is not None:
            total_points += 1
            if is_valid:
                valid_points += 1
            if preview:
                plt.show(block=False)
                plt.pause(0.1)
                plt.close(fig)

            temp_img_filename = "temp_img.jpg"
            fig.savefig(temp_img_filename, dpi=dpi)
            plt.close(fig)
            frame = cv2.imread(temp_img_filename)
            video.write(frame)
            os.remove(temp_img_filename)

    video.release()

    # 计算并打印有效预测的比例
    if total_points > 0:
        valid_ratio = valid_points / total_points
        print(f"有效预测的比例（绿色点）: {valid_ratio:.2f}")
    else:
        print("没有进行预测。")


def darkest_edge_detection(img, fit_type="gauss", degree=6, label=None):
    rows, cols = img.shape
    max_points = []  # 存储每列中拟合的最大值点坐标

    # 对每一列进行处理
    for col in range(cols):
        column_values = img[:, col]
        min_val_index = np.argmin(column_values)

        start_index = max(min_val_index - 3, 0)
        end_index = min(min_val_index + 4, rows)
        y_values = 255 - column_values[start_index:end_index]
        x_values = np.arange(start_index, end_index)

        # 根据拟合类型进行拟合
        if fit_type == "gauss":
            fit_func, params = gauss_fit(x_values, y_values)
            if params is not None:
                a, b, _ = params
                max_points.append((col, b))
        elif fit_type == "poly":
            fit_func = poly_fit(x_values, y_values, degree)
            if fit_func is not None:
                # 寻找多项式拟合曲线的极值点
                derivative = np.polyder(fit_func)
                roots = np.roots(derivative)
                real_roots = roots[np.isreal(roots)].real
                if real_roots.size > 0:
                    extreme_values = fit_func(real_roots)
                    max_index = np.argmax(extreme_values)
                    b = real_roots[max_index]
                    max_points.append((col, b))

        elif fit_type == "orthogonal_poly":
            fit_func = orthogonal_poly_fit(x_values, y_values, degree)
            if fit_func is not None:
                # 使用 Legendre 对象的 deriv 方法计算导数
                derivative = fit_func.deriv()
                roots = derivative.roots()

                # 转换 roots 到原始 x 值区间的函数
                def convert_to_original_x(x_normalized):
                    return (
                        x_normalized * (max(x_values) - min(x_values)) / 2
                        + (max(x_values) + min(x_values)) / 2
                    )

                # 转换 roots 到原始 x 值区间
                roots_original = convert_to_original_x(roots)

                # 过滤非实数根和不在原始 x 值范围内的根
                real_roots = roots_original[
                    np.isreal(roots_original)
                    & (roots_original >= min(x_values))
                    & (roots_original <= max(x_values))
                ].real
                if real_roots.size > 0:
                    # 选择最大的极值点
                    extreme_values = [
                        fit_func(
                            2 * (x - min(x_values)) / (max(x_values) - min(x_values))
                            - 1
                        )
                        for x in real_roots
                    ]
                    max_index = np.argmax(extreme_values)
                    b = real_roots[max_index]
                    max_points.append((col, b))

    return max_points


def main(fit_type="gauss", degree=6):
    img_path = "/Users/youkipepper/Desktop/pbMoMa/test.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    # 计算每列的 start_index 和 end_index
    start_indices = [max(np.argmin(img[:, col]) - 3, 0) for col in range(cols)]
    end_indices = [min(np.argmin(img[:, col]) + 4, rows) for col in range(cols)]

    # create_video(img, fit_type=fit_type, preview=True, start_indices=start_indices, end_indices=end_indices)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    edge_points = darkest_edge_detection(img, degree=degree)
    for point in edge_points:
        # 在彩色图像上用红色点标记
        cv2.circle(img_color, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
        # 显示带有红色标记的彩色图像
    cv2.imshow(f"edge({fit_type}_degree={degree})", img_color)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

    output_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{fit_type}_degree={degree}.jpg"


    # 保存带有红色标记的彩色图像
    cv2.imwrite(output_filename, img_color)

    # 加载图像并转换为灰度图
    img = cv2.imread("./test.png", cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    fit_type = "orthogonal_poly"  # 'gauss' or 'poly' or 'orthogonal_poly'
    main(fit_type)
