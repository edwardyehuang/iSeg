"""TensorFlow utility functions and constants for SOD metrics.

This module provides TensorFlow equivalents of NumPy utility functions used in SOD metrics evaluation.
"""

import math

import tensorflow as tf

# The different implementation of epsilon between numpy and matlab
# np.spacing(1) ≈ 2.220446049250313e-16
TYPE = tf.float32
EPS = tf.constant(2.220446049250313e-16, dtype=TYPE)
PI = tf.constant(math.pi, dtype=TYPE)


def validate_and_normalize_input(
    pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
) -> tuple:
    """Validate and optionally normalize prediction and ground truth inputs.

    This function ensures that prediction and ground truth tensors have compatible shapes
    and appropriate data types. When normalization is enabled, it converts inputs to the
    standard format required by the predefined metrics (pred in [0, 1] as float, gt as boolean).

    Args:
        pred (tf.Tensor): Prediction tensor. If `normalize=True`, should be uint8 grayscale
            image (0-255). If `normalize=False`, should be float32/float64 in range [0, 1].
        gt (tf.Tensor): Ground truth tensor. If `normalize=True`, should be uint8 grayscale
            image (0-255). If `normalize=False`, should be boolean tensor.
        normalize (bool, optional): Whether to normalize the input data. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pred (tf.Tensor): Normalized prediction as float64 in range [0, 1].
            - gt (tf.Tensor): Normalized ground truth as boolean tensor.

    Raises:
        tf.errors.InvalidArgumentError: If shapes don't match or validation fails.
    """
    # Validate input shapes
    tf.debugging.assert_equal(
        tf.shape(pred),
        tf.shape(gt),
        message="Shape mismatch between prediction and ground truth",
    )

    if normalize:
        pred, gt = prepare_data(pred, gt)
    else:
        # Ensure proper dtypes
        pred = tf.cast(pred, dtype=TYPE)
        gt = tf.cast(gt, dtype=tf.bool)

    return pred, gt


def get_one_tensor (ref_tensor):

    one = tf.ones_like(ref_tensor)
    one = tf.cast(one, dtype=TYPE)
    one = tf.reduce_mean(one)

    return one


def prepare_data(pred: tf.Tensor, gt: tf.Tensor) -> tuple:
    """Convert and normalize prediction and ground truth data.

    - For predictions, mimics MATLAB's `mapminmax(im2double(...))`.
    - For ground truth, applies binary thresholding at 128.

    Args:
        pred (tf.Tensor): Prediction grayscale image, uint8 type with values in [0, 255].
        gt (tf.Tensor): Ground truth grayscale image, uint8 type with values in [0, 255].

    Returns:
        tuple: A tuple containing:
            - pred (tf.Tensor): Normalized prediction as float64 in range [0, 1].
            - gt (tf.Tensor): Binary ground truth as boolean tensor.
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = tf.cast(pred, dtype=TYPE) / 255.0

    pred_max = tf.reduce_max(pred)
    pred_min = tf.reduce_min(pred)

    # Normalize if max != min
    pred = tf.cond(
        pred_max != pred_min,
        lambda: (pred - pred_min) / (pred_max - pred_min),
        lambda: pred,
    )
    return pred, gt


def get_adaptive_threshold(matrix: tf.Tensor, max_value: float = 1.0) -> tf.Tensor:
    """Return an adaptive threshold, which is equal to twice the mean of `matrix`.

    Args:
        matrix (tf.Tensor): A data tensor.
        max_value (float, optional): The upper limit of the threshold. Defaults to 1.0.

    Returns:
        tf.Tensor: `min(2 * matrix.mean(), max_value)` as a scalar tensor.
    """
    threshold = 2.0 * tf.reduce_mean(matrix)
    return tf.minimum(threshold, tf.constant(max_value, dtype=TYPE))


def tf_histogram(values: tf.Tensor, bins: int, min_val: float, max_val: float) -> tf.Tensor:
    """Compute histogram of tensor values.

    Args:
        values (tf.Tensor): Input values to histogram.
        bins (int): Number of bins.
        min_val (float): Minimum value for histogram range.
        max_val (float): Maximum value for histogram range.

    Returns:
        tf.Tensor: Histogram counts for each bin.
    """
    values = tf.reshape(values, [-1])
    values = tf.cast(values, dtype=TYPE)

    # Compute bin indices
    bin_width = (max_val - min_val) / tf.cast(bins, TYPE)
    indices = tf.cast((values - min_val) / bin_width, tf.int32)
    indices = tf.clip_by_value(indices, 0, bins - 1)

    # Count occurrences in each bin
    hist = tf.math.bincount(indices, minlength=bins, maxlength=bins, dtype=tf.int32)
    return tf.cast(hist, TYPE)


@tf.autograph.experimental.do_not_convert
def safe_divide(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    """Safe division that handles zero denominators.

    Args:
        numerator (tf.Tensor): Numerator values.
        denominator (tf.Tensor): Denominator values.

    Returns:
        tf.Tensor: Result of division with zero handling.
    """
    denominator = tf.cast(denominator, dtype=tf.float32)
    numerator = tf.cast(numerator, dtype=tf.float32)
    return tf.where(
        tf.equal(denominator, 0.0), tf.zeros_like(numerator), numerator / denominator
    )


def tf_convolve2d(
    image: tf.Tensor, kernel: tf.Tensor, mode: str = "constant", cval: float = 0.0
) -> tf.Tensor:
    """2D convolution similar to scipy.ndimage.convolve.

    Args:
        image (tf.Tensor): 2D input tensor.
        kernel (tf.Tensor): 2D convolution kernel.
        mode (str): Padding mode. Currently only 'constant' is supported.
        cval (float): Value to use for constant padding.

    Returns:
        tf.Tensor: Convolved image.
    """
    # Ensure proper shapes for tf.nn.conv2d
    # Input shape: [batch, height, width, channels]
    # Kernel shape: [height, width, in_channels, out_channels]
    image = tf.cast(image, TYPE)
    kernel = tf.cast(kernel, TYPE)

    image_4d = tf.reshape(image, [1, tf.shape(image)[0], tf.shape(image)[1], 1])
    kernel_4d = tf.reshape(kernel, [tf.shape(kernel)[0], tf.shape(kernel)[1], 1, 1])

    # Compute padding using dynamic shape
    kernel_shape = tf.shape(kernel)
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad with constant value
    if mode == "constant":
        padded = tf.pad(
            image_4d,
            [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]],
            mode="CONSTANT",
            constant_values=cval,
        )
    else:
        padded = image_4d

    # Perform convolution
    result = tf.nn.conv2d(padded, kernel_4d, strides=[1, 1, 1, 1], padding="VALID")
    return tf.squeeze(result)


def tf_gaussian_kernel(shape: tuple = (7, 7), sigma: float = 5.0) -> tf.Tensor:
    """Generate a 2D Gaussian kernel compatible with MATLAB's fspecial.

    Args:
        shape (tuple, optional): Kernel size as (height, width). Defaults to (7, 7).
        sigma (float, optional): Standard deviation of the Gaussian. Defaults to 5.0.

    Returns:
        tf.Tensor: Normalized 2D Gaussian kernel.
    """
    m, n = (shape[0] - 1) / 2, (shape[1] - 1) / 2

    y = tf.range(-m, m + 1, dtype=tf.float32)
    x = tf.range(-n, n + 1, dtype=tf.float32)
    y, x = tf.meshgrid(y, x, indexing="ij")

    y = tf.cast(y, dtype=TYPE)
    x = tf.cast(x, dtype=TYPE)

    h = tf.exp(-(x * x + y * y) / (2 * sigma * sigma))

    # Zero out very small values (use module-level EPS for stability)
    h = tf.where(h < EPS * tf.reduce_max(h), tf.zeros_like(h), h)

    # Normalize
    sum_h = tf.reduce_sum(h)
    h = tf.cond(sum_h != 0, lambda: h / sum_h, lambda: h)

    return h


def tf_sobel_edge(mask: tf.Tensor) -> tf.Tensor:
    """Compute edge detection using Sobel operator.

    Args:
        mask (tf.Tensor): Binary mask input.

    Returns:
        tf.Tensor: Edge detection result (binary mask of edges).
    """
    mask = tf.cast(mask, TYPE)

    # Sobel kernels
    sobel_x = tf.constant(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=TYPE
    )
    sobel_y = tf.constant(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=TYPE
    )

    # Reshape for conv2d
    mask_4d = tf.reshape(mask, [1, tf.shape(mask)[0], tf.shape(mask)[1], 1])
    sobel_x_4d = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_4d = tf.reshape(sobel_y, [3, 3, 1, 1])

    # Pad input
    padded = tf.pad(mask_4d, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

    # Compute gradients
    sx = tf.nn.conv2d(padded, sobel_x_4d, strides=[1, 1, 1, 1], padding="VALID")
    sy = tf.nn.conv2d(padded, sobel_y_4d, strides=[1, 1, 1, 1], padding="VALID")

    # Compute magnitude
    sob = tf.sqrt(tf.square(sx) + tf.square(sy))
    sob = tf.squeeze(sob)

    return tf.cast(sob > 0, TYPE)


def tf_distance_transform_edt(
    mask: tf.Tensor, return_indices: bool = False
) -> tf.Tensor:
    """Compute Euclidean distance transform.

    This is a TensorFlow approximation of scipy.ndimage.distance_transform_edt.

    Args:
        mask (tf.Tensor): Binary mask where True/1 values are foreground.
        return_indices (bool): If True, also return indices of nearest background.

    Returns:
        tf.Tensor: Distance transform.
        If return_indices is True, also returns (indices_y, indices_x).
    """
    mask = tf.cast(mask, tf.bool)
    shape = tf.shape(mask)
    h, w = shape[0], shape[1]

    # Create coordinate grids
    y_coords = tf.range(h, dtype=tf.int32)
    x_coords = tf.range(w, dtype=tf.int32)
    yy, xx = tf.meshgrid(y_coords, x_coords, indexing="ij")

    yy = tf.cast(yy, dtype=TYPE)
    xx = tf.cast(xx, dtype=TYPE)

    # Find background pixel coordinates (where mask is False)
    bg_mask = tf.logical_not(mask)
    bg_indices = tf.where(bg_mask)  # Nx2 tensor of [y, x] coordinates

    # If no background pixels, return zeros/max distance
    def no_background():
        dist = tf.fill(tf.shape(mask), tf.cast(tf.reduce_max([h, w]), TYPE))
        if return_indices:
            idx_y = tf.zeros_like(mask, dtype=tf.int32)
            idx_x = tf.zeros_like(mask, dtype=tf.int32)
            return dist, (idx_y, idx_x)
        return dist

    def has_background():
        bg_y = tf.cast(bg_indices[:, 0], TYPE)
        bg_x = tf.cast(bg_indices[:, 1], TYPE)

        # For each pixel, compute distance to all background pixels
        # This is O(n*m) but works for smaller images
        yy_flat = tf.reshape(yy, [-1])
        xx_flat = tf.reshape(xx, [-1])

        # Compute squared distances to each background pixel
        dy = tf.expand_dims(yy_flat, 1) - tf.expand_dims(bg_y, 0)
        dx = tf.expand_dims(xx_flat, 1) - tf.expand_dims(bg_x, 0)
        sq_dist = dy * dy + dx * dx

        # Find minimum distance and its index
        min_idx = tf.argmin(sq_dist, axis=1)
        min_dist = tf.sqrt(tf.reduce_min(sq_dist, axis=1))

        dist = tf.reshape(min_dist, [h, w])

        if return_indices:
            nearest_bg_idx = tf.gather(bg_indices, min_idx)
            idx_y = tf.reshape(nearest_bg_idx[:, 0], [h, w])
            idx_x = tf.reshape(nearest_bg_idx[:, 1], [h, w])
            return dist, (tf.cast(idx_y, tf.int32), tf.cast(idx_x, tf.int32))
        return dist

    num_bg = tf.shape(bg_indices)[0]
    if return_indices:
        return tf.cond(
            num_bg == 0,
            no_background,
            has_background,
        )
    return tf.cond(num_bg == 0, no_background, has_background)


def tf_label_connected_components(mask: tf.Tensor) -> tf.Tensor:
    """Label connected components in a binary mask.

    This is a TensorFlow implementation of scipy.ndimage.label with 4-connectivity.

    Args:
        mask (tf.Tensor): Binary mask input.

    Returns:
        tf.Tensor: Labeled image where each connected component has a unique integer.
    """
    mask = tf.cast(mask, tf.bool)
    # Use TensorFlow's connected components
    labeled = tf.cast(
        tf.image.connected_components(tf.cast(mask, tf.int32)[tf.newaxis, ..., tf.newaxis]),
        tf.int32,
    )
    return tf.squeeze(labeled)


def tf_morphology_dilate(mask: tf.Tensor, kernel_size: int = 3, iterations: int = 1) -> tf.Tensor:
    """Morphological dilation operation.

    Args:
        mask (tf.Tensor): Binary mask input.
        kernel_size (int): Size of the dilation kernel.
        iterations (int): Number of dilation iterations.

    Returns:
        tf.Tensor: Dilated mask.
    """
    mask = tf.cast(mask, TYPE)
    mask_4d = tf.reshape(mask, [1, tf.shape(mask)[0], tf.shape(mask)[1], 1])

    # Create dilation kernel
    kernel = tf.ones([kernel_size, kernel_size, 1], dtype=TYPE)

    for _ in range(iterations):
        mask_4d = tf.nn.dilation2d(
            mask_4d,
            kernel,
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
        )

    return tf.squeeze(mask_4d)


def tf_morphology_erode(mask: tf.Tensor, kernel_size: int = 3, iterations: int = 1) -> tf.Tensor:
    """Morphological erosion operation.

    Args:
        mask (tf.Tensor): Binary mask input.
        kernel_size (int): Size of the erosion kernel.
        iterations (int): Number of erosion iterations.

    Returns:
        tf.Tensor: Eroded mask.
    """
    mask = tf.cast(mask, TYPE)
    mask_4d = tf.reshape(mask, [1, tf.shape(mask)[0], tf.shape(mask)[1], 1])

    # Create erosion kernel
    kernel = tf.ones([kernel_size, kernel_size, 1], dtype=TYPE)

    for _ in range(iterations):
        mask_4d = tf.nn.erosion2d(
            mask_4d,
            kernel,
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
        )

    return tf.squeeze(mask_4d)


def tf_skeletonize(mask: tf.Tensor) -> tf.Tensor:
    """Compute skeleton of a binary mask using morphological operations.

    This is an approximation of skimage.morphology.skeletonize.

    Args:
        mask (tf.Tensor): Binary mask input.

    Returns:
        tf.Tensor: Skeletonized mask.
    """
    mask = tf.cast(mask, tf.bool)
    result = tf.cast(mask, TYPE)

    # Iterative skeletonization using morphological operations
    prev = tf.zeros_like(result)

    def cond(prev, current):
        return tf.reduce_any(tf.not_equal(prev, current))

    def body(prev, current):
        # Erosion
        eroded = tf_morphology_erode(current, kernel_size=3, iterations=1)
        # Opening (erosion followed by dilation)
        opened = tf_morphology_dilate(eroded, kernel_size=3, iterations=1)
        # Subtract opening from current
        subset = current - opened
        subset = tf.maximum(subset, 0.0)
        # Update
        return current, eroded + subset

    _, skeleton = tf.while_loop(
        cond,
        body,
        [prev, result],
        maximum_iterations=100,
    )

    return tf.cast(skeleton > 0, tf.bool)


def tf_count_nonzero(tensor: tf.Tensor) -> tf.Tensor:
    """Count non-zero elements in a tensor.

    Args:
        tensor (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Scalar count of non-zero elements.
    """
    return tf.math.count_nonzero(tensor)


def tf_filter2d(image: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """Apply a 2D filter to an image, similar to cv2.filter2D.

    Args:
        image (tf.Tensor): 2D input image.
        kernel (tf.Tensor): 2D filter kernel.

    Returns:
        tf.Tensor: Filtered image.
    """
    image = tf.cast(image, TYPE)
    kernel = tf.cast(kernel, TYPE)

    # Add batch and channel dimensions
    image_4d = image[tf.newaxis, ..., tf.newaxis]
    kernel_4d = kernel[..., tf.newaxis, tf.newaxis]

    # Compute padding size
    pad_h = tf.shape(kernel)[0] // 2
    pad_w = tf.shape(kernel)[1] // 2

    # Pad the image
    padded = tf.pad(
        image_4d, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT"
    )

    # Apply convolution
    result = tf.nn.conv2d(padded, kernel_4d, strides=[1, 1, 1, 1], padding="VALID")

    return tf.squeeze(result)


def tf_rgb_to_lab(image: tf.Tensor) -> tf.Tensor:
    """Convert RGB image to CIE Lab color space.

    Args:
        image (tf.Tensor): RGB image tensor with values in [0, 255], shape (H, W, 3).

    Returns:
        tf.Tensor: Lab image tensor, shape (H, W, 3).
    """
    # Normalize to [0, 1]
    rgb = tf.cast(image, TYPE) / 255.0

    # sRGB to linear RGB
    rgb = tf.where(
        rgb > 0.04045,
        tf.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )

    # Linear RGB to XYZ (D65 illuminant)
    # Transformation matrix
    rgb_to_xyz = tf.constant([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=TYPE)

    xyz = tf.tensordot(rgb, rgb_to_xyz, axes=[[-1], [1]])

    # Normalize for D65 illuminant
    xyz_ref = tf.constant([0.95047, 1.0, 1.08883], dtype=TYPE)
    xyz = xyz / xyz_ref

    # XYZ to Lab
    epsilon = 0.008856
    kappa = 903.3

    xyz = tf.where(
        xyz > epsilon,
        tf.pow(xyz, 1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0
    )

    L = 116.0 * xyz[..., 1] - 16.0
    a = 500.0 * (xyz[..., 0] - xyz[..., 1])
    b = 200.0 * (xyz[..., 1] - xyz[..., 2])

    return tf.stack([L, a, b], axis=-1)


def tf_delta_e2000(lab1: tf.Tensor, lab2: tf.Tensor) -> tf.Tensor:
    """Compute CIEDE2000 color difference between two Lab images.

    Args:
        lab1 (tf.Tensor): First Lab image, shape (H, W, 3).
        lab2 (tf.Tensor): Second Lab image, shape (H, W, 3).

    Returns:
        tf.Tensor: Delta E 2000 values, shape (H, W).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Constants
    kL = tf.constant(1.0, dtype=TYPE)
    kC = tf.constant(1.0, dtype=TYPE)
    kH = tf.constant(1.0, dtype=TYPE)

    # Calculate C1, C2
    C1 = tf.sqrt(a1 * a1 + b1 * b1)
    C2 = tf.sqrt(a2 * a2 + b2 * b2)

    # Calculate C_bar
    C_bar = (C1 + C2) / 2.0

    # Calculate G
    C_bar_7 = tf.pow(C_bar, 7.0)
    G = 0.5 * (1.0 - tf.sqrt(C_bar_7 / (C_bar_7 + tf.pow(25.0, 7.0))))

    # Calculate a1_prime, a2_prime
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    # Calculate C1_prime, C2_prime
    C1_prime = tf.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = tf.sqrt(a2_prime * a2_prime + b2 * b2)

    # Calculate h1_prime, h2_prime
    h1_prime = tf.atan2(b1, a1_prime)
    h1_prime = tf.where(h1_prime < 0, h1_prime + 2.0 * PI, h1_prime)
    h2_prime = tf.atan2(b2, a2_prime)
    h2_prime = tf.where(h2_prime < 0, h2_prime + 2.0 * PI, h2_prime)

    # Convert to degrees
    h1_prime_deg = h1_prime * 180.0 / PI
    h2_prime_deg = h2_prime * 180.0 / PI

    # Calculate delta_L_prime, delta_C_prime, delta_h_prime
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Calculate delta_h_prime
    h_diff = h2_prime_deg - h1_prime_deg
    C_product = C1_prime * C2_prime

    delta_h_prime_deg = tf.where(
        C_product == 0,
        tf.zeros_like(h_diff),
        tf.where(
            tf.abs(h_diff) <= 180.0,
            h_diff,
            tf.where(
                h_diff > 180.0,
                h_diff - 360.0,
                h_diff + 360.0
            )
        )
    )

    delta_H_prime = 2.0 * tf.sqrt(C_product) * tf.sin(delta_h_prime_deg * PI / 360.0)

    # Calculate L_bar_prime, C_bar_prime, h_bar_prime
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h_sum = h1_prime_deg + h2_prime_deg
    h_bar_prime_deg = tf.where(
        C_product == 0,
        h_sum,
        tf.where(
            tf.abs(h_diff) <= 180.0,
            h_sum / 2.0,
            tf.where(
                h_sum < 360.0,
                (h_sum + 360.0) / 2.0,
                (h_sum - 360.0) / 2.0
            )
        )
    )

    # Calculate T
    T = (1.0 - 0.17 * tf.cos((h_bar_prime_deg - 30.0) * PI / 180.0) +
         0.24 * tf.cos(2.0 * h_bar_prime_deg * PI / 180.0) +
         0.32 * tf.cos((3.0 * h_bar_prime_deg + 6.0) * PI / 180.0) -
         0.20 * tf.cos((4.0 * h_bar_prime_deg - 63.0) * PI / 180.0))

    # Calculate delta_theta
    delta_theta = 30.0 * tf.exp(-tf.square((h_bar_prime_deg - 275.0) / 25.0))

    # Calculate RC
    C_bar_prime_7 = tf.pow(C_bar_prime, 7.0)
    RC = 2.0 * tf.sqrt(C_bar_prime_7 / (C_bar_prime_7 + tf.pow(25.0, 7.0)))

    # Calculate SL, SC, SH
    L_bar_prime_minus_50_sq = tf.square(L_bar_prime - 50.0)
    SL = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / tf.sqrt(20.0 + L_bar_prime_minus_50_sq)
    SC = 1.0 + 0.045 * C_bar_prime
    SH = 1.0 + 0.015 * C_bar_prime * T

    # Calculate RT
    RT = -tf.sin(2.0 * delta_theta * PI / 180.0) * RC

    # Calculate final delta E
    delta_E = tf.sqrt(
        tf.square(delta_L_prime / (kL * SL)) +
        tf.square(delta_C_prime / (kC * SC)) +
        tf.square(delta_H_prime / (kH * SH)) +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )

    return delta_E


def tf_extract_patches(
    image: tf.Tensor,
    mask: tf.Tensor,
    patch_size: int,
    stride: int
) -> tuple:
    """Extract patches from an image based on a mask.

    Args:
        image (tf.Tensor): Input image tensor, shape (H, W, C).
        mask (tf.Tensor): Binary mask tensor, shape (H, W).
        patch_size (int): Size of patches to extract.
        stride (int): Stride between patches.

    Returns:
        tuple: (valid_indices, valid_patches)
            - valid_indices: Tensor of (y, x) coordinates for valid patches
            - valid_patches: Tensor of flattened patch values
    """
    image = tf.cast(image, TYPE)
    mask = tf.cast(mask, TYPE)

    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # Compute padding
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride

    # Pad image and mask
    image_padded = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], mode="REFLECT")
    mask_padded = tf.pad(mask, [[0, pad_h], [0, pad_w]], mode="CONSTANT")

    new_shape = tf.shape(image_padded)
    new_h, new_w = new_shape[0], new_shape[1]

    # Extract patches using tf.image.extract_patches
    image_4d = image_padded[tf.newaxis, ...]
    patches = tf.image.extract_patches(
        image_4d,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patches = tf.squeeze(patches, axis=0)

    # Similarly for mask
    mask_4d = mask_padded[tf.newaxis, ..., tf.newaxis]
    mask_patches = tf.image.extract_patches(
        mask_4d,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    mask_patches = tf.squeeze(mask_patches, axis=0)

    # Reshape patches
    patch_grid_shape = tf.shape(patches)
    gh, gw = patch_grid_shape[0], patch_grid_shape[1]

    patches_flat = tf.reshape(patches, [gh * gw, -1])
    mask_patches_flat = tf.reshape(mask_patches, [gh * gw, -1])

    # Create indices
    grid_y, grid_x = tf.meshgrid(
        tf.range(0, new_h - patch_size + 1, stride),
        tf.range(0, new_w - patch_size + 1, stride),
        indexing="ij"
    )
    all_indices = tf.stack([tf.reshape(grid_y, [-1]), tf.reshape(grid_x, [-1])], axis=1)

    # Find valid patches (all mask values > 0)
    valid_mask = tf.reduce_all(mask_patches_flat > 0, axis=1)
    valid_indices = tf.boolean_mask(all_indices, valid_mask)
    valid_patches = tf.boolean_mask(patches_flat, valid_mask)

    return valid_indices, valid_patches


def tf_nearest_neighbors_with_spatial(
    reference_features: tf.Tensor,
    query_features: tf.Tensor,
    reference_coords: tf.Tensor,
    query_coords: tf.Tensor,
    lambda_spatial: float = 20.0
) -> tf.Tensor:
    """Find nearest neighbors with spatial constraints.

    Args:
        reference_features (tf.Tensor): Reference feature vectors, shape (N, D).
        query_features (tf.Tensor): Query feature vectors, shape (M, D).
        reference_coords (tf.Tensor): Reference coordinates, shape (N, 2).
        query_coords (tf.Tensor): Query coordinates, shape (M, 2).
        lambda_spatial (float): Weight for spatial distance.

    Returns:
        tf.Tensor: Indices of nearest neighbors in reference for each query, shape (M, 1).
    """
    reference_features = tf.cast(reference_features, TYPE)
    query_features = tf.cast(query_features, TYPE)
    reference_coords = tf.cast(reference_coords, TYPE)
    query_coords = tf.cast(query_coords, TYPE)

    # Standardize coordinates
    all_coords = tf.concat([reference_coords, query_coords], axis=0)
    coord_mean = tf.reduce_mean(all_coords, axis=0)
    coord_std = tf.math.reduce_std(all_coords, axis=0) + EPS

    ref_coords_scaled = (reference_coords - coord_mean) / coord_std
    query_coords_scaled = (query_coords - coord_mean) / coord_std

    # Augment features with spatial coordinates
    ref_augmented = tf.concat([reference_features, lambda_spatial * ref_coords_scaled], axis=1)
    query_augmented = tf.concat([query_features, lambda_spatial * query_coords_scaled], axis=1)

    # Compute pairwise distances
    # Using efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    ref_sq_sum = tf.reduce_sum(tf.square(ref_augmented), axis=1, keepdims=True)
    query_sq_sum = tf.reduce_sum(tf.square(query_augmented), axis=1, keepdims=True)
    dot_product = tf.matmul(query_augmented, ref_augmented, transpose_b=True)

    distances = query_sq_sum + tf.transpose(ref_sq_sum) - 2.0 * dot_product

    # Find nearest neighbor indices
    indices = tf.argmin(distances, axis=1)
    return tf.expand_dims(indices, axis=1)


def tf_find_contours(mask: tf.Tensor) -> tf.Tensor:
    """Find contour points in a binary mask.

    This is a TensorFlow approximation of cv2.findContours.
    Returns boundary points rather than hierarchical contours.

    Args:
        mask (tf.Tensor): Binary mask tensor, shape (H, W).

    Returns:
        tf.Tensor: Contour points as (N, 2) tensor of (row, col) coordinates.
    """
    mask = tf.cast(mask, TYPE)

    # Use Sobel edge detection to find boundaries
    mask_4d = mask[tf.newaxis, ..., tf.newaxis]

    # Sobel filters for edge detection
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=TYPE)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=TYPE)

    sobel_x_4d = sobel_x[..., tf.newaxis, tf.newaxis]
    sobel_y_4d = sobel_y[..., tf.newaxis, tf.newaxis]

    # Pad and convolve
    padded = tf.pad(mask_4d, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    gx = tf.nn.conv2d(padded, sobel_x_4d, strides=[1, 1, 1, 1], padding="VALID")
    gy = tf.nn.conv2d(padded, sobel_y_4d, strides=[1, 1, 1, 1], padding="VALID")

    edge_magnitude = tf.sqrt(tf.square(gx) + tf.square(gy))
    edge_mask = tf.squeeze(edge_magnitude) > 0

    # Get contour point coordinates
    contour_points = tf.where(edge_mask)
    return contour_points


def tf_approx_poly_dp(points: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Approximate a polygon using Ramer-Douglas-Peucker algorithm.

    This is a TensorFlow implementation of cv2.approxPolyDP.

    Args:
        points (tf.Tensor): Polygon points, shape (N, 2).
        epsilon (float): Approximation accuracy (max distance from original curve).

    Returns:
        tf.Tensor: Simplified polygon points.
    """
    points = tf.cast(points, TYPE)
    n_points = tf.shape(points)[0]

    def rdp_recursive(pts, eps):
        """Recursive RDP implementation."""
        n = tf.shape(pts)[0]

        def base_case():
            return pts

        def recursive_case():
            # Find point with maximum distance from line
            start = pts[0]
            end = pts[-1]
            line_vec = end - start
            line_len_sq = tf.reduce_sum(tf.square(line_vec)) + EPS

            # Compute perpendicular distances
            pts_vec = pts - start
            t = tf.reduce_sum(pts_vec * line_vec, axis=1) / line_len_sq
            t = tf.clip_by_value(t, 0.0, 1.0)
            projections = start + tf.expand_dims(t, 1) * line_vec
            distances = tf.sqrt(tf.reduce_sum(tf.square(pts - projections), axis=1))

            max_dist = tf.reduce_max(distances)
            max_idx = tf.argmax(distances)

            def simplify():
                # Recursively simplify both halves
                left = rdp_recursive(pts[:max_idx + 1], eps)
                right = rdp_recursive(pts[max_idx:], eps)
                # Combine (excluding duplicate point)
                return tf.concat([left[:-1], right], axis=0)

            def keep_endpoints():
                return tf.stack([start, end], axis=0)

            return tf.cond(max_dist > eps, simplify, keep_endpoints)

        return tf.cond(n <= 2, base_case, recursive_case)

    # Handle edge cases
    def empty_case():
        return points

    def normal_case():
        return rdp_recursive(points, epsilon)

    return tf.cond(n_points <= 2, empty_case, normal_case)


def tf_filter_conditional_boundary(
    contour_points: tf.Tensor,
    mask: tf.Tensor,
    condition: tf.Tensor
) -> tuple:
    """Filter boundary segments based on condition mask.

    Args:
        contour_points (tf.Tensor): Contour points, shape (N, 2).
        mask (tf.Tensor): Binary mask of the region.
        condition (tf.Tensor): Condition mask for filtering.

    Returns:
        tuple: (filtered_boundary_points, independent_count)
    """
    # Dilate condition mask
    condition = tf.cast(condition, TYPE)
    condition_dilated = tf_morphology_dilate(condition, kernel_size=3, iterations=1)
    condition_dilated = condition_dilated > 0

    # Label connected components in mask
    labeled = tf_label_connected_components(mask)

    # Get number of contour points
    n_points = tf.shape(contour_points)[0]

    # Filter points based on condition
    def check_point(i):
        row = contour_points[i, 0]
        col = contour_points[i, 1]
        shape = tf.shape(condition_dilated)
        in_bounds = tf.logical_and(
            tf.logical_and(row >= 0, row < shape[0]),
            tf.logical_and(col >= 0, col < shape[1])
        )
        cond_val = tf.cond(
            in_bounds,
            lambda: condition_dilated[row, col],
            lambda: False
        )
        return cond_val

    # Get valid point mask
    valid_mask = tf.map_fn(check_point, tf.range(n_points), fn_output_signature=tf.bool)
    filtered_points = tf.boolean_mask(contour_points, valid_mask)

    # Count independent regions (simplified: count unique labels at filtered points)
    n_filtered = tf.shape(filtered_points)[0]

    def count_regions():
        point_labels = tf.map_fn(
            lambda i: labeled[filtered_points[i, 0], filtered_points[i, 1]],
            tf.range(n_filtered),
            fn_output_signature=tf.int32
        )
        unique_labels, _ = tf.unique(point_labels)
        # Filter out background (label 0)
        non_bg = unique_labels > 0
        return tf.reduce_sum(tf.cast(non_bg, tf.int32))

    independent_count = tf.cond(n_filtered > 0, count_regions, lambda: tf.constant(0))

    return filtered_points, independent_count


def tf_count_polygon_control_points(
    boundary_points: tf.Tensor,
    epsilon: float
) -> tf.Tensor:
    """Count polygon control points using RDP approximation.

    Args:
        boundary_points (tf.Tensor): Boundary points, shape (N, 2).
        epsilon (float): RDP approximation tolerance.

    Returns:
        tf.Tensor: Number of control points.
    """
    n_points = tf.shape(boundary_points)[0]

    def has_points():
        approx = tf_approx_poly_dp(boundary_points, epsilon)
        return tf.shape(approx)[0]

    return tf.cond(n_points > 0, has_points, lambda: tf.constant(0))