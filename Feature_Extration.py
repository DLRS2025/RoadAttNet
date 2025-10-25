import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize, remove_small_objects
import os

def display_all_features(image_path):
    """
    Extract and display all features from the given image path using the original methods.
    
    Parameters:
        image_path: Path to the input image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Resize for consistency
    img = cv2.resize(img, (1500, 1500))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Feature 1 & 2: From color_filter.py
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    _, sat_mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)  # 低饱和度
    val_range = cv2.inRange(v, 80, 220)  # 中等亮度
    non_green = cv2.bitwise_not(cv2.inRange(h, 35, 90))
    color_filter = cv2.bitwise_and(cv2.bitwise_and(sat_mask, val_range), non_green)

    
    # Feature 3 & 4: From feature89.py
    # 1. 噪声去除
    denoised = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    
    # 2. 颜色空间转换
    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    
    # 3. 道路检测掩码生成
    # A. 基于亮度和饱和度
    _, bright_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, low_sat_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    road_mask1 = cv2.bitwise_and(low_sat_mask, bright_mask)
    
    # B. 基于颜色特征
    gray_road_lower = np.array([0, 0, 80])
    gray_road_upper = np.array([180, 40, 220])
    gray_road_mask = cv2.inRange(hsv, gray_road_lower, gray_road_upper)
    dark_road_lower = np.array([0, 0, 20])
    dark_road_upper = np.array([180, 30, 100])
    dark_road_mask = cv2.inRange(hsv, dark_road_lower, dark_road_upper)
    color_road_mask = cv2.bitwise_or(gray_road_mask, dark_road_mask)
    
    # C. 线性结构增强
    line_kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19)),
        np.eye(13, dtype=np.uint8),
        np.flip(np.eye(13, dtype=np.uint8), 0)
    ]
    line_features = np.zeros_like(gray)
    for kernel in line_kernels:
        temp = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        line_features = cv2.max(line_features, temp)
    road_line_diff = cv2.absdiff(gray, line_features)
    road_line_mask = cv2.adaptiveThreshold(
        road_line_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # D. 多方向梯度
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_dir = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    road_dir_mask = np.zeros_like(gray)
    for i in range(0, gray.shape[0]-10, 10):
        for j in range(0, gray.shape[1]-10, 10):
            window = grad_dir[i:i+10, j:j+10]
            if np.std(window) < 30:
                road_dir_mask[i:i+10, j:j+10] = 255
    
    # 4. 纹理特征
    local_var = cv2.GaussianBlur(gray, (0, 0), 2) - cv2.GaussianBlur(gray, (0, 0), 5)
    local_var = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(local_var, 30, 255, cv2.THRESH_BINARY_INV)
    
    # 5. 组合掩码
    primary_mask = cv2.bitwise_or(road_mask1, color_road_mask)
    linear_mask = cv2.bitwise_or(road_line_mask, road_dir_mask)
    grad_mask = (grad_mag > 40).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(
        primary_mask,
        cv2.bitwise_or(linear_mask, cv2.bitwise_or(grad_mask, texture_mask))
    )
    
    # 6. 形态学处理
    kernel_close = np.ones((5, 5), np.uint8)
    connected_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, kernel_open)
    skeleton = morphology.skeletonize(cleaned_mask > 0)
    skeleton = skeleton.astype(np.uint8) * 255
    kernel_dilate = np.ones((3, 3), np.uint8)
    road_rebuilt = cv2.dilate(skeleton, kernel_dilate, iterations=2)
    combined_mask = cv2.bitwise_or(cleaned_mask, road_rebuilt)
    
    # 7. 高级道路增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    gaussian = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    usm = cv2.addWeighted(enhanced_gray, 1.8, gaussian, -0.8, 0)
    
    # 8. 创建最终增强图像
    road_enhanced = cv2.bitwise_and(usm, usm, mask=combined_mask)
    road_bg = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(combined_mask))
    result_gray = cv2.add(road_enhanced, road_bg)
    
    # 创建伪彩色图像
    result_pseudo = cv2.applyColorMap(result_gray, cv2.COLORMAP_JET)
    result_pseudo = cv2.cvtColor(result_pseudo, cv2.COLOR_BGR2RGB)
    
    # 创建道路突出显示图像
    road_highlight_overlay = np.zeros_like(img_rgb)
    road_highlight_overlay[combined_mask > 0] = [0, 255, 0]  # 绿色表示道路
    result_highlight = cv2.addWeighted(img_rgb, 0.7, road_highlight_overlay, 0.3, 0)
    
    # Feature 5: From road_highlight.py
    # Apply bilateral filter to preserve edges while removing noise
    denoised_rh = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    gray_rh = cv2.cvtColor(denoised_rh, cv2.COLOR_RGB2GRAY)
    
    # Convert to HSV color space for better feature extraction
    hsv_rh = cv2.cvtColor(denoised_rh, cv2.COLOR_RGB2HSV)
    h_rh, s_rh, v_rh = cv2.split(hsv_rh)
    
    # STEP 1: Basic road features
    bright_mask = cv2.adaptiveThreshold(v_rh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 101, -15)
    
    # Saturation is low for roads
    _, sat_mask = cv2.threshold(s_rh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # STEP 2: Add vegetation filtering
    r, g, b = cv2.split(denoised_rh)
    veg_index = np.zeros_like(r, dtype=np.float32)
    non_zero_mask = (r + g) > 0
    veg_index[non_zero_mask] = (g[non_zero_mask].astype(float) - r[non_zero_mask].astype(float)) / (g[non_zero_mask] + r[non_zero_mask])
    _, veg_mask = cv2.threshold(((veg_index + 1) * 127.5).astype(np.uint8), 165, 255, cv2.THRESH_BINARY_INV)
    
    # STEP 3: Combine primary features
    road_mask_preliminary = cv2.bitwise_and(bright_mask, sat_mask)
    road_mask_preliminary = cv2.bitwise_and(road_mask_preliminary, veg_mask)
    
    # STEP 4: Add linear features
    edges = cv2.Canny(gray_rh, 50, 150)
    lines_mask = np.zeros_like(gray_rh)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Dilate lines slightly to connect nearby segments
    kernel_line = np.ones((5, 5), np.uint8)
    lines_mask = cv2.dilate(lines_mask, kernel_line, iterations=1)
    
    # STEP 5: Combine primary mask with line features
    road_prob = (road_mask_preliminary.astype(np.float32) * 0.6 +
                    lines_mask.astype(np.float32) * 0.4)
    road_prob_norm = (road_prob / 255.0 * 255).astype(np.uint8)
    _, binary_road_mask = cv2.threshold(road_prob_norm, 75, 255, cv2.THRESH_BINARY)
    
    # STEP 6: Morphological operations
    kernel = np.ones((7, 7), np.uint8)
    road_mask = cv2.morphologyEx(binary_road_mask, cv2.MORPH_CLOSE, kernel)
    
    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, 8)
    min_area = 300
    refined_mask = np.zeros_like(road_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            refined_mask[labels == i] = 255
    
    # STEP 7: Final cleanup
    final_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Create visual results
    road_highlight = np.zeros_like(img_rgb)
    road_highlight[final_mask > 0] = [0, 255, 0]
    result_highlight_rh = cv2.addWeighted(img_rgb, 0.7, road_highlight, 0.3, 0)
    
    # Feature 6 & 7: From road_probability.py
    gray_rp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_rp = cv2.GaussianBlur(gray_rp, (5, 5), 2.0)
    
    clahe_rp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe_rp.apply(denoised_rp)
    
    edges_rp = cv2.Canny(enhanced, 50, 150)
    
    kernel_rp = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges_rp, kernel_rp, iterations=1)
    
    lines_mask_rp = np.zeros_like(gray_rp)
    
    major_lines = cv2.HoughLinesP(
        dilated_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=60,
        minLineLength=80,
        maxLineGap=20
    )
    
    if major_lines is not None:
        for line in major_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask_rp, (x1, y1), (x2, y2), 255, 3)
    
    hsv_rp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_rp = hsv_rp[:, :, 0]
    s_rp = hsv_rp[:, :, 1]
    
    _, sat_mask_rp = cv2.threshold(s_rp, 40, 255, cv2.THRESH_BINARY_INV)
    
    green_mask = cv2.inRange(h_rp, 35, 85)
    veg_mask_rp = cv2.bitwise_not(green_mask)
    
    color_filter_rp = cv2.bitwise_and(sat_mask_rp, veg_mask_rp)
    
    filtered_lines = cv2.bitwise_and(lines_mask_rp, color_filter_rp)
    
    road_prob_rp = cv2.GaussianBlur(filtered_lines.astype(float), (15, 15), 0)
    max_val = np.max(road_prob_rp)
    if max_val > 0:
        road_prob_rp = (road_prob_rp / max_val * 255).astype(np.uint8)
    else:
        road_prob_rp = np.zeros_like(road_prob_rp, dtype=np.uint8)
    

    features = {
        'color_filter': color_filter,
        'road_mask1': road_mask1,
        'color_road_mask': color_road_mask,
        'combined_mask': combined_mask,
        'road_enhanced': road_enhanced,
        'result_gray': result_gray,
        'result_pseudo': result_pseudo,
        'result_highlight': result_highlight,
        'road_highlight': road_highlight,
        'road_prob_rp': road_prob_rp,
        'filtered_lines': filtered_lines,
    }
    

    return features


def feature_extraction(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    features = display_all_features(image_path)
    # Combine all features with specific weights
    combined_features = np.zeros_like(features['color_filter'], dtype=np.float32)
    
    # Define weights for each feature
    weights = {
        'color_filter': 8,
        'road_mask1': 1,
        'color_road_mask': 0.1,
        'combined_mask': 0.1,
        'road_enhanced': 1,
        'result_gray': 9,
        'result_pseudo': 0.5,
        'result_highlight': 2,
        'road_highlight': 3,
        'road_prob_rp': 8,
        'filtered_lines': 9
    }
    
    for feature_name, feature_img in features.items():
        # Ensure all features are in the same format (grayscale)
        if len(feature_img.shape) == 3:
            # Convert RGB to grayscale if needed
            feature_gray = cv2.cvtColor(feature_img, cv2.COLOR_RGB2GRAY)
        else:
            feature_gray = feature_img
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        feature_equalized = clahe.apply(feature_gray)
            
        # Add to the combined features with appropriate weight
        weight = weights.get(feature_name, 1.0)  # Default weight is 1.0
        combined_features += weight * feature_equalized.astype(np.float32)
    
    # Normalize the combined features to 0-255 range
    min_val = np.min(combined_features)
    max_val = np.max(combined_features)
    if max_val > min_val:
        combined_features = 255 * (combined_features - min_val) / (max_val - min_val)
    
    # Add the combined features to the features dictionary
    features['combined_all'] = combined_features.astype(np.float32)
    # 使用滤波器强化线状特征
    # 创建一个Gabor滤波器组来增强不同方向的线状特征
    combined_gabor = np.zeros_like(combined_features)
    
    # 定义Gabor滤波器参数 - 针对3-20像素宽度的线状地物
    ksize = 80  # 滤波器大小，适合中等宽度的线状特征
    sigma = 50   # 标准差，控制滤波器的有效范围
    lambd = 70  # 波长，设置为线状地物宽度的上限，捕获不同宽度
    gamma = 0.6 # 空间纵横比，略微增加以更好地捕获窄线条
    psi = 0     # 相位偏移，使用0以获得对称响应
    
    # 应用不同方向的Gabor滤波器
    for theta in np.arange(0, np.pi, np.pi/8):  # 8个不同方向
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(combined_features.astype(np.uint8), cv2.CV_8UC3, gabor_kernel)
        combined_gabor = np.maximum(combined_gabor, filtered)
    
    # 应用形态学操作进一步增强线状特征
    kernel_line = np.ones((3, 3), np.uint8)
    enhanced_lines = cv2.morphologyEx(combined_gabor.astype(np.uint8), cv2.MORPH_CLOSE, kernel_line)
    
    # 使用自适应阈值处理增强对比度
    enhanced_lines_thresh = cv2.adaptiveThreshold(
        enhanced_lines,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # 细化线条
    skeleton = morphology.skeletonize(enhanced_lines_thresh > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    # 将增强后的线状特征添加到特征字典中
    features['enhanced_lines'] = enhanced_lines
    features['enhanced_lines_thresh'] = enhanced_lines_thresh
    features['skeleton_lines'] = skeleton_img
    
    # 将线状特征与原始组合特征融合
    features['combined_with_lines'] = cv2.addWeighted(
        combined_features.astype(np.uint8), 
        0.7, 
        enhanced_lines, 
        0.3, 
        0
    )

    # 获取输入图像的基本文件名
    base_filename = os.path.basename(image_path)
    base_name = os.path.splitext(base_filename)[0]
    
    # 保存结果时包含基本文件名
    # 将combined_with_lines映射到0-255
    combined_with_lines = (features['combined_with_lines'] - np.min(features['combined_with_lines'])) / (np.max(features['combined_with_lines']) - np.min(features['combined_with_lines'])) * 255
    combined_with_lines = combined_with_lines.astype(np.uint8)
    # Create the subdirectory if it doesn't exist
    cwl_dir = os.path.join(output_dir, "cwl")
    if not os.path.exists(cwl_dir):
        os.makedirs(cwl_dir)
    cv2.imwrite(os.path.join(output_dir, "cwl", f"{base_name}.tiff"), combined_with_lines)
    # 展示combined_all
    # plt.figure(figsize=(15, 10))
    # plt.imshow(features['combined_with_lines'], cmap='gray')
    # plt.title('Combined All Features (Weighted with CLAHE Enhancement)')
    # plt.colorbar(label='Intensity')
    # plt.axis('on')
    # plt.show()

    # # 展示所有feature总共11个
    # plt.figure(figsize=(15, 10))
    
    # feature_names = list(features.keys())
    # num_features = len(feature_names)
    # rows = (num_features + 2) // 3  # 计算需要的行数
    
    # # 展示所有特征
    # for i, (name, feature) in enumerate(features.items()):
    #     plt.subplot(rows, 3, i+1)
        
    #     # 根据特征类型选择合适的显示方式
    #     if len(feature.shape) == 3:  # RGB图像
    #         plt.imshow(feature)
    #     else:  # 灰度图或掩码
    #         plt.imshow(feature, cmap='gray')
            
    #     plt.title(name)
    #     plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()

    # 2. 图像增强——使用CLAHE提高局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(features['combined_with_lines'])

    # 3. 降噪处理——高斯滤波（也可以尝试中值滤波）
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    

    # 4. 线状特征增强——使用 Gabor 滤波器
    #    注：可尝试多角度滤波后融合，这里仅以一个角度为例
    ksize = 31         # 滤波器尺寸
    sigma = 4.0        # 高斯核标准差
    theta = np.pi / 4  # 滤波方向（45°），可循环多个方向后加权融合
    lamda = 10.0       # 波长
    gamma = 0.5        # 长宽比
    psi = 0            # 相位偏移
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(blurred, cv2.CV_8UC3, gabor_kernel)

    # 5. 融合原图与 Gabor 滤波结果（加权平均，可调整权重）
    combined = cv2.addWeighted(blurred, 0.5, gabor_filtered, 0.5, 0)

    # 6. 阈值分割——采用 Otsu 自适应阈值分割
    ret, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. 形态学处理——闭运算填补小空洞，去除孤立噪点
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph, iterations=2)

    # 8. 去除小对象——利用 skimage 的 remove_small_objects（注意输入需为布尔类型图像）
    closed_bool = closed.astype(bool)
    cleaned = remove_small_objects(closed_bool, min_size=150, connectivity=2)
    cleaned = (cleaned.astype(np.uint8)) * 255

    # 9. 骨架提取——得到道路中心线
    skeleton = skeletonize(cleaned // 255)  # skeletonize 期望输入为二值（0/1）图像
    skeleton = (skeleton.astype(np.uint8)) * 255


    # 将blurred映射到0-255
    blurred = (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred)) * 255
    blurred = blurred.astype(np.uint8)
    # Create the subdirectory if it doesn't exist
    blurred_dir = os.path.join(output_dir, "blurred")
    if not os.path.exists(blurred_dir):
        os.makedirs(blurred_dir)
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_skeleton.tiff"), skeleton)
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_combined.tiff"), combined)
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.tiff"), enhanced)
    cv2.imwrite(os.path.join(output_dir, "blurred", f"{base_name}.tiff"), blurred)
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_gabor_filtered.tiff"), gabor_filtered)

    # # 10. 保存各步骤结果（可选）
    # cv2.imwrite("enhanced.png", enhanced)
    # cv2.imwrite("blurred.png", blurred)
    # cv2.imwrite("gabor_filtered.png", gabor_filtered)
    # cv2.imwrite("combined.png", combined)
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("closed.png", closed)
    # cv2.imwrite("cleaned.png", cleaned)
    # cv2.imwrite("skeleton.png", skeleton)

    # # 11. 显示结果
    # titles = ['Original Grayscale', 'CLAHE Enhancement', 'Gaussian Blur', 'Gabor Filter', 'Combined Image', 'Binary Segmentation', 'Closing Operation', 'Small Object Removal', 'Skeleton Extraction']
    # images = [features['combined_with_lines'], enhanced, blurred, gabor_filtered, combined, binary, closed, cleaned, skeleton]

    # plt.figure(figsize=(15, 8))
    # for i in range(len(images)):
    #     plt.subplot(3, 3, i+1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # Replace with your actual image path
    image_path = r"D:\YWWY\Projects\ZQproject\Road-extraction\archive\tiff\test\10378780_15.tiff"
    output_dir = r"D:\YWWY\Projects\ZQproject\Road-extraction\archive\tiff"

    feature_extraction(image_path, output_dir)

