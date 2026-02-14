import tensorflow as tf

EPS = 1e-6


def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * inter + EPS) / (denom + EPS)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), EPS, 1.0 - EPS)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(loss)


def boundary_aware_loss(y_true, y_pred, w=5.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), EPS, 1.0 - EPS)
    edges = tf.image.sobel_edges(y_true)
    mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    e = tf.clip_by_value(mag / 4.0, 0.0, 1.0)
    weights = 1.0 + w * e
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(weights * bce)


def composite_loss(y_true, y_pred, lam_d=0.4, lam_f=0.4, lam_b=0.2):
    ld = dice_loss(y_true, y_pred)
    lf = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    lb = boundary_aware_loss(y_true, y_pred, w=5.0)
    return lam_d * ld + lam_f * lf + lam_b * lb
