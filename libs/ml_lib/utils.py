from . import *
import time

# # 将输入图像中每个局部区域的像素按列展开，形成一个列向量
def im2col(input_data, filter_h, filter_w, out_h, out_w, stride=1, pad=0):
    s = time.perf_counter()
    N, C, _, _ = input_data.shape
    img = mypy.pad(input_data, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)
    e1 = time.perf_counter()
    # 使用 mypy.lib.stride_tricks.as_strided 实现矢量化
    col = mypy.lib.stride_tricks.as_strided(
        img,
        shape=(N, out_h, out_w, C, filter_h, filter_w),
        strides=(img.strides[0], stride*img.strides[2], stride*img.strides[3], img.strides[1], img.strides[2], img.strides[3])
    ).reshape(N*out_h*out_w, -1)

    e = time.perf_counter()
    # print(e-e1, e1-s, img.shape)
    return col

# 将输入列向量中对应位置的增量加到原矩阵上输出
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # from (N, out_h, out_w, C, filter_h, filter_w) to (N, C, filter_h, filter_w, out_h, out_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = mypy.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]