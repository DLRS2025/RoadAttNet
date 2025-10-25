import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from glob import glob
import cv2

# 设置GPU内存增长
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 数据路径
OPTICAL_DIR = "D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img1_shuffle"
FEATURE1_DIR = "D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img2_shuffle"
FEATURE2_DIR = "D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img3_shuffle"
MASK_DIR = "D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/mask_shuffle"

# 模型参数
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 4
EPOCHS = 2
IMG_CHANNELS = 3

# 创新点1: 注意力模块
def attention_module(x, ratio=8):
    """通道注意力和空间注意力的组合"""
    # 通道注意力
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    filters = x.shape[channel_axis]
    
    # Squeeze操作
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    # Excitation操作
    avg_pool = tf.keras.layers.Reshape((1, 1, filters))(avg_pool)
    max_pool = tf.keras.layers.Reshape((1, 1, filters))(max_pool)
    
    avg_pool = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(avg_pool)
    avg_pool = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(avg_pool)
    
    max_pool = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(max_pool)
    max_pool = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(max_pool)
    
    cbam_feature = avg_pool + max_pool
    
    # 空间注意力
    avg_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(x)
    max_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(x)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])
    
    spatial_feature = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    # 合并通道和空间注意力
    refined = x * cbam_feature * spatial_feature
    
    return refined

# 创新点2: 多尺度特征提取模块
def multi_scale_module(input_tensor, filters):
    # 1x1 卷积
    conv1x1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    
    # 3x3 卷积
    conv3x3 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    
    # 膨胀卷积，用于增大感受野
    dilated_conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(2, 2), activation='relu')(input_tensor)
    dilated_conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(4, 4), activation='relu')(input_tensor)
    
    # 特征融合
    concat = tf.keras.layers.Concatenate()([conv1x1, conv3x3, dilated_conv1, dilated_conv2])
    output = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(concat)
    
    return output

# 创新点3: 边界增强模块
def boundary_refinement_module(input_tensor):
    # 提取边缘特征
    edge_detector1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    edge_detector1 = tf.keras.layers.BatchNormalization()(edge_detector1)
    
    edge_detector2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(edge_detector1)
    edge_detector2 = tf.keras.layers.BatchNormalization()(edge_detector2)
    
    # 增强边缘特征
    enhanced_edges = tf.keras.layers.add([edge_detector1, edge_detector2])
    
    # 将边缘特征与原始特征融合
    output = tf.keras.layers.Concatenate()([input_tensor, enhanced_edges])
    output = tf.keras.layers.Conv2D(input_tensor.shape[-1], (1, 1), padding='same')(output)
    
    return output

# 创新点4: 多特征融合模块
def feature_fusion_module(optical, feature1, feature2):
    # 确保所有特征具有相同的通道数
    optical_processed = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(optical)
    feature1_processed = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(feature1)
    feature2_processed = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(feature2)
    
    # 使用注意力机制进行特征融合
    optical_att = attention_module(optical_processed)
    feature1_att = attention_module(feature1_processed)
    feature2_att = attention_module(feature2_processed)
    
    # 融合特征
    fused_features = tf.keras.layers.add([optical_att, feature1_att, feature2_att])
    
    return fused_features

# 构建道路提取模型
def build_road_extraction_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    # 多输入
    optical_input = tf.keras.Input(shape=input_shape)
    feature1_input = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    feature2_input = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # 编码器部分 - 光学图像
    optical_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(optical_input)
    optical_conv1 = tf.keras.layers.BatchNormalization()(optical_conv1)
    optical_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(optical_conv1)
    optical_conv1 = tf.keras.layers.BatchNormalization()(optical_conv1)
    optical_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(optical_conv1)
    
    optical_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(optical_pool1)
    optical_conv2 = tf.keras.layers.BatchNormalization()(optical_conv2)
    optical_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(optical_conv2)
    optical_conv2 = tf.keras.layers.BatchNormalization()(optical_conv2)
    optical_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(optical_conv2)
    
    optical_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(optical_pool2)
    optical_conv3 = tf.keras.layers.BatchNormalization()(optical_conv3)
    optical_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(optical_conv3)
    optical_conv3 = tf.keras.layers.BatchNormalization()(optical_conv3)
    optical_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(optical_conv3)
    
    # 编码器部分 - 特征1
    feature1_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(feature1_input)
    feature1_conv1 = tf.keras.layers.BatchNormalization()(feature1_conv1)
    feature1_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature1_conv1)
    
    feature1_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(feature1_pool1)
    feature1_conv2 = tf.keras.layers.BatchNormalization()(feature1_conv2)
    feature1_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature1_conv2)
    
    feature1_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(feature1_pool2)
    feature1_conv3 = tf.keras.layers.BatchNormalization()(feature1_conv3)
    feature1_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature1_conv3)
    
    # 编码器部分 - 特征2
    feature2_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(feature2_input)
    feature2_conv1 = tf.keras.layers.BatchNormalization()(feature2_conv1)
    feature2_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature2_conv1)
    
    feature2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(feature2_pool1)
    feature2_conv2 = tf.keras.layers.BatchNormalization()(feature2_conv2)
    feature2_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature2_conv2)
    
    feature2_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(feature2_pool2)
    feature2_conv3 = tf.keras.layers.BatchNormalization()(feature2_conv3)
    feature2_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature2_conv3)
    
    # 特征融合 - 中层
    fused_features = feature_fusion_module(optical_pool3, feature1_pool3, feature2_pool3)
    
    # 中间层 - 瓶颈
    bottleneck = multi_scale_module(fused_features, 512)
    bottleneck = attention_module(bottleneck)
    bottleneck = boundary_refinement_module(bottleneck)
    
    # 解码器部分
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(bottleneck)
    up3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(up3)
    concat3 = tf.keras.layers.Concatenate()([up3, optical_conv3, feature1_conv3, feature2_conv3])
    concat3 = attention_module(concat3)
    decoder_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(concat3)
    decoder_conv3 = tf.keras.layers.BatchNormalization()(decoder_conv3)
    decoder_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(decoder_conv3)
    decoder_conv3 = tf.keras.layers.BatchNormalization()(decoder_conv3)
    
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(decoder_conv3)
    up2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(up2)
    concat2 = tf.keras.layers.Concatenate()([up2, optical_conv2, feature1_conv2, feature2_conv2])
    concat2 = attention_module(concat2)
    decoder_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(concat2)
    decoder_conv2 = tf.keras.layers.BatchNormalization()(decoder_conv2)
    decoder_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(decoder_conv2)
    decoder_conv2 = tf.keras.layers.BatchNormalization()(decoder_conv2)
    
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(decoder_conv2)
    up1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(up1)
    concat1 = tf.keras.layers.Concatenate()([up1, optical_conv1, feature1_conv1, feature2_conv1])
    concat1 = attention_module(concat1)
    decoder_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(concat1)
    decoder_conv1 = tf.keras.layers.BatchNormalization()(decoder_conv1)
    decoder_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(decoder_conv1)
    decoder_conv1 = tf.keras.layers.BatchNormalization()(decoder_conv1)
    
    # 边界细化
    refined_features = boundary_refinement_module(decoder_conv1)
    
    # 输出层
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(refined_features)
    
    # 创建模型
    model = tf.keras.Model(inputs=[optical_input, feature1_input, feature2_input], outputs=outputs)
    
    return model

# 创新点5: 混合损失函数
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + ((1 - y_true) * (1 - alpha))
    modulating_factor = tf.pow((1.0 - p_t), gamma)
    
    # 计算交叉熵
    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    loss = alpha_factor * modulating_factor * ce
    
    return tf.reduce_mean(loss)

def hybrid_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    return bce + dice + 0.5 * focal

# 数据加载函数
def load_data(optical_dir, feature1_dir, feature2_dir, mask_dir):
    optical_files = sorted(glob(os.path.join(optical_dir, "*.tiff")))
    feature1_files = sorted(glob(os.path.join(feature1_dir, "*.tiff")))
    feature2_files = sorted(glob(os.path.join(feature2_dir, "*.tiff")))
    mask_files = sorted(glob(os.path.join(mask_dir, "*.tiff")))
    
    return optical_files, feature1_files, feature2_files, mask_files

# 数据生成器
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, optical_files, feature1_files, feature2_files, mask_files, batch_size=8, shuffle=True):
        self.optical_files = optical_files
        self.feature1_files = feature1_files
        self.feature2_files = feature2_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.optical_files) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.optical_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        # 生成批次索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # 批次文件列表
        optical_batch = [self.optical_files[i] for i in indexes]
        feature1_batch = [self.feature1_files[i] for i in indexes]
        feature2_batch = [self.feature2_files[i] for i in indexes]
        mask_batch = [self.mask_files[i] for i in indexes]
        
        # 生成数据
        X, y = self.__data_generation(optical_batch, feature1_batch, feature2_batch, mask_batch)
        
        return X, y
    
    def __data_generation(self, optical_batch, feature1_batch, feature2_batch, mask_batch):
        # 初始化
        X_optical = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        X_feature1 = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        X_feature2 = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        y = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        
        # 生成数据
        for i, (optical_file, feature1_file, feature2_file, mask_file) in enumerate(zip(optical_batch, feature1_batch, feature2_batch, mask_batch)):
            # 加载图像
            optical = cv2.imread(optical_file)
            optical = cv2.cvtColor(optical, cv2.COLOR_BGR2RGB)
            optical = optical.astype(np.float32) / 255.0
            
            feature1 = cv2.imread(feature1_file, cv2.IMREAD_GRAYSCALE)
            feature1 = np.expand_dims(feature1, axis=-1)
            feature1 = feature1.astype(np.float32) / 255.0
            
            feature2 = cv2.imread(feature2_file, cv2.IMREAD_GRAYSCALE)
            feature2 = np.expand_dims(feature2, axis=-1)
            feature2 = feature2.astype(np.float32) / 255.0
            
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(np.float32) / 255.0
            
            # 存储样本
            X_optical[i,] = optical
            X_feature1[i,] = feature1
            X_feature2[i,] = feature2
            y[i,] = mask
        
        return [X_optical, X_feature1, X_feature2], y

# 创新点6: 自适应学习率调度器
class CosineAnnealingLR(tf.keras.callbacks.Callback):
    def __init__(self, base_lr, max_lr, cycle_length, warmup_epochs=5):
        super(CosineAnnealingLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.warmup_epochs = warmup_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # 线性预热
            lr = self.base_lr + (self.max_lr - self.base_lr) * epoch / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) % self.cycle_length / self.cycle_length
            lr = self.base_lr + 0.5 * (self.max_lr - self.base_lr) * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.6f}")

# 创新点7: 数据增强函数
def apply_augmentation(optical, feature1, feature2, mask):
    # 随机旋转
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)  # 随机旋转90°，180°或270°
        optical = np.rot90(optical, k)
        feature1 = np.rot90(feature1, k)
        feature2 = np.rot90(feature2, k)
        mask = np.rot90(mask, k)
    
    # 随机翻转
    if np.random.rand() < 0.5:
        optical = np.fliplr(optical)
        feature1 = np.fliplr(feature1)
        feature2 = np.fliplr(feature2)
        mask = np.fliplr(mask)
    
    if np.random.rand() < 0.5:
        optical = np.flipud(optical)
        feature1 = np.flipud(feature1)
        feature2 = np.flipud(feature2)
        mask = np.flipud(mask)
    
    # 随机亮度和对比度调整（仅适用于光学图像）
    if np.random.rand() < 0.5:
        alpha = 0.8 + np.random.rand() * 0.4  # 对比度因子：0.8-1.2
        beta = -0.1 + np.random.rand() * 0.2  # 亮度调整：-0.1-0.1
        optical = np.clip(alpha * optical + beta, 0, 1)
    
    return optical, feature1, feature2, mask

# 主函数
def main():
    # 加载数据
    optical_files, feature1_files, feature2_files, mask_files = load_data(OPTICAL_DIR, FEATURE1_DIR, FEATURE2_DIR, MASK_DIR)
    
    # 划分训练集、验证集
    indices = np.arange(len(optical_files))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_optical = [optical_files[i] for i in train_idx]
    train_feature1 = [feature1_files[i] for i in train_idx]
    train_feature2 = [feature2_files[i] for i in train_idx]
    train_mask = [mask_files[i] for i in train_idx]
    
    val_optical = [optical_files[i] for i in val_idx]
    val_feature1 = [feature1_files[i] for i in val_idx]
    val_feature2 = [feature2_files[i] for i in val_idx]
    val_mask = [mask_files[i] for i in val_idx]
    
    # 创建数据生成器
    train_generator = DataGenerator(train_optical, train_feature1, train_feature2, train_mask, batch_size=BATCH_SIZE)
    val_generator = DataGenerator(val_optical, val_feature1, val_feature2, val_mask, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    model = build_road_extraction_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=hybrid_loss,
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'), tf.keras.metrics.Recall()]
    )
    
    # 回调函数
    model_checkpoint = ModelCheckpoint(
        'road_extraction_best_model.h5',
        monitor='val_iou',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_iou',
        patience=15,
        mode='max',
        verbose=1
    )
    
    lr_scheduler = CosineAnnealingLR(
        base_lr=1e-5,
        max_lr=1e-3,
        cycle_length=10,
        warmup_epochs=5
    )
    
    # 训练模型
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping, lr_scheduler]
    )
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['iou'], label='Train')
    plt.plot(history.history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # 加载最佳模型并进行预测
    model = tf.keras.models.load_model('road_extraction_best_model.h5', compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=hybrid_loss,
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'), tf.keras.metrics.Recall()]
    )
    
    # 评估模型
    eval_results = model.evaluate(val_generator)
    print(f"Evaluation results: Loss: {eval_results[0]:.4f}, Accuracy: {eval_results[1]:.4f}, IoU: {eval_results[2]:.4f}, Recall: {eval_results[3]:.4f}")
    
    # 可视化一些预测结果
    X_test, y_test = val_generator.__getitem__(0)
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(20, 10))
    for i in range(min(5, BATCH_SIZE)):
        # 原始光学图像
        plt.subplot(3, 5, i + 1)
        plt.imshow(X_test[0][i])
        plt.title('Optical Image')
        plt.axis('off')
        
        # 真实掩码
        plt.subplot(3, 5, i + 6)
        plt.imshow(y_test[i, :, :, 0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # 预测掩码
        plt.subplot(3, 5, i + 11)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

if __name__ == "__main__":
    main()