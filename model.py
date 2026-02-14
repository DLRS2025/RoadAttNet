import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from losses import composite_loss


def _bilinear_sample(img, coords_y, coords_x):
    img = tf.convert_to_tensor(img)
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    y = coords_y
    x = coords_x
    y0 = tf.floor(y)
    x0 = tf.floor(x)
    y1 = y0 + 1.0
    x1 = x0 + 1.0
    y0i = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
    x0i = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
    y1i = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)
    x1i = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)

    b = tf.range(B, dtype=tf.int32)[:, None, None]
    b = tf.tile(b, [1, H, W])

    def gather(y_idx, x_idx):
        idx = tf.stack([b, y_idx, x_idx], axis=-1)
        return tf.gather_nd(img, idx)

    Ia = gather(y0i, x0i)
    Ib = gather(y1i, x0i)
    Ic = gather(y0i, x1i)
    Id = gather(y1i, x1i)

    y0f = tf.cast(y0i, tf.float32)
    x0f = tf.cast(x0i, tf.float32)
    y1f = tf.cast(y1i, tf.float32)
    x1f = tf.cast(x1i, tf.float32)

    wa = (x1f - x) * (y1f - y)
    wb = (x1f - x) * (y - y0f)
    wc = (x - x0f) * (y1f - y)
    wd = (x - x0f) * (y - y0f)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


class OrientedCoordinateAttention(layers.Layer):
    def __init__(self, length=9, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.reduction = int(reduction)

    def build(self, input_shape):
        C = int(input_shape[-1])
        hidden = max(8, C // self.reduction)
        self.theta_conv3 = layers.Conv2D(hidden, 3, padding="same", activation="relu")
        self.theta_conv1 = layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        self.attn_reduce = layers.Conv2D(hidden, 1, padding="same", activation="relu")
        self.attn_expand = layers.Conv2D(2 * C, 1, padding="same", activation="sigmoid")
        super().build(input_shape)

    def _oriented_pool(self, x, vx, vy):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        yy, xx = tf.meshgrid(tf.range(H, dtype=tf.float32), tf.range(W, dtype=tf.float32), indexing="ij")
        yy = tf.reshape(yy, [1, H, W, 1])
        xx = tf.reshape(xx, [1, H, W, 1])
        yy = tf.tile(yy, [B, 1, 1, 1])
        xx = tf.tile(xx, [B, 1, 1, 1])
        half = self.length // 2
        offsets = tf.range(-half, half + 1, dtype=tf.float32)
        acc = 0.0
        for t in tf.unstack(offsets):
            coords_y = yy + t * vy
            coords_x = xx + t * vx
            sample = _bilinear_sample(x, tf.squeeze(coords_y, -1), tf.squeeze(coords_x, -1))
            acc = acc + sample
        return acc / tf.cast(self.length, tf.float32)

    def call(self, x):
        theta = np.pi * self.theta_conv1(self.theta_conv3(x))
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        vtan_x, vtan_y = cos_t, sin_t
        vnor_x, vnor_y = -sin_t, cos_t
        tan_feat = self._oriented_pool(x, vtan_x, vtan_y)
        nor_feat = self._oriented_pool(x, vnor_x, vnor_y)
        context = tf.concat([tan_feat, nor_feat], axis=-1)
        w = self.attn_expand(self.attn_reduce(context))
        alpha_tan, alpha_norm = tf.split(w, num_or_size_splits=2, axis=-1)
        return (alpha_tan + alpha_norm) * x


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def multiscale_rgb_branch(rgb, base_filters=32):
    f0 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(rgb)
    d2 = layers.AveragePooling2D(pool_size=2)(rgb)
    d2 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(d2)
    u2 = layers.UpSampling2D(size=2, interpolation="bilinear")(d2)
    d4 = layers.AveragePooling2D(pool_size=4)(rgb)
    d4 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(d4)
    u4 = layers.UpSampling2D(size=4, interpolation="bilinear")(d4)
    ms = layers.Concatenate()([f0, u2, u4])
    ms = layers.Conv2D(base_filters * 2, 1, padding="same", activation="relu")(ms)
    return ms


def multidim_prior_branch(prior, out_filters=64):
    x = layers.Conv2D(out_filters, 3, padding="same", activation="relu")(prior)
    x = layers.Conv2D(out_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(out_filters, 1, padding="same", activation="relu")(x)
    return x


def build_roadattnet_core(input_shape=(512, 512, 4), base_filters=64, oca_length=9):
    inp = layers.Input(shape=input_shape)
    rgb = inp[..., :3]
    prior = inp[..., 3:4]

    rgb_ms = multiscale_rgb_branch(rgb, base_filters=base_filters // 2)
    prior_f = multidim_prior_branch(prior, out_filters=base_filters)

    x0 = layers.Concatenate()([rgb_ms, prior_f])
    e1 = residual_block(x0, base_filters)
    p1 = layers.MaxPooling2D(pool_size=2)(e1)

    e2 = residual_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=2)(e2)

    e3 = residual_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=2)(e3)

    e4 = residual_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=2)(e4)

    bott = residual_block(p4, base_filters * 16)

    oca1 = OrientedCoordinateAttention(length=oca_length, name="oca_d4")
    oca2 = OrientedCoordinateAttention(length=oca_length, name="oca_d3")
    oca3 = OrientedCoordinateAttention(length=oca_length, name="oca_d2")
    oca4 = OrientedCoordinateAttention(length=oca_length, name="oca_d1")

    d4 = layers.UpSampling2D(size=2, interpolation="bilinear")(bott)
    d4 = layers.Concatenate()([d4, e4])
    d4 = oca1(d4)
    d4 = residual_block(d4, base_filters * 8)
    aux1 = layers.Conv2D(1, 1, activation="sigmoid", name="aux1_raw")(d4)
    aux1_up = layers.UpSampling2D(size=8, interpolation="bilinear", name="aux1")(aux1)

    d3 = layers.UpSampling2D(size=2, interpolation="bilinear")(d4)
    d3 = layers.Concatenate()([d3, e3])
    d3 = oca2(d3)
    d3 = residual_block(d3, base_filters * 4)
    aux2 = layers.Conv2D(1, 1, activation="sigmoid", name="aux2_raw")(d3)
    aux2_up = layers.UpSampling2D(size=4, interpolation="bilinear", name="aux2")(aux2)

    d2 = layers.UpSampling2D(size=2, interpolation="bilinear")(d3)
    d2 = layers.Concatenate()([d2, e2])
    d2 = oca3(d2)
    d2 = residual_block(d2, base_filters * 2)
    aux3 = layers.Conv2D(1, 1, activation="sigmoid", name="aux3_raw")(d2)
    aux3_up = layers.UpSampling2D(size=2, interpolation="bilinear", name="aux3")(aux3)

    d1 = layers.UpSampling2D(size=2, interpolation="bilinear")(d2)
    d1 = layers.Concatenate()([d1, e1])
    d1 = oca4(d1)
    d1 = residual_block(d1, base_filters)
    main = layers.Conv2D(1, 1, activation="sigmoid", name="main")(d1)

    return tf.keras.Model(inputs=inp, outputs=[main, aux1_up, aux2_up, aux3_up], name="RoadAttNetCore")


class RoadAttNet(tf.keras.Model):
    def __init__(self, core: tf.keras.Model, grad_accum_steps: int = 1, grad_clip_norm: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.core = core
        self.grad_accum_steps = int(max(1, grad_accum_steps))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))

        self.s1 = self.add_weight(name="s1", shape=(), initializer="zeros", trainable=True)
        self.s2 = self.add_weight(name="s2", shape=(), initializer="zeros", trainable=True)
        self.s3 = self.add_weight(name="s3", shape=(), initializer="zeros", trainable=True)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.main_loss_tracker = tf.keras.metrics.Mean(name="main_loss")
        self.aux_loss_tracker = tf.keras.metrics.Mean(name="aux_loss")
        self.acc = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.iou = tf.keras.metrics.MeanIoU(num_classes=2, name="iou")

        self._accum_step = None
        self._accum_grads = None
        self._var_index = None

    @property
    def metrics(self):
        return [self.loss_tracker, self.main_loss_tracker, self.aux_loss_tracker, self.acc, self.recall, self.iou]

    def build(self, input_shape):
        self.core.build(input_shape)
        super().build(input_shape)
        self._var_index = {v.ref(): i for i, v in enumerate(self.trainable_variables)}
        if self.grad_accum_steps > 1 and self._accum_grads is None:
            self._accum_step = self.add_weight(name="accum_step", shape=(), dtype=tf.int32, initializer="zeros", trainable=False)
            self._accum_grads = []
            for v in self.trainable_variables:
                self._accum_grads.append(
                    self.add_weight(
                        name=("accum_" + v.name.replace(":", "_")),
                        shape=v.shape,
                        dtype=tf.float32,
                        initializer="zeros",
                        trainable=False,
                    )
                )

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def _update_iou(self, y_true, y_pred):
        y_true_i = tf.cast(y_true > 0.5, tf.int32)
        y_pred_i = tf.cast(y_pred > 0.5, tf.int32)
        self.iou.update_state(y_true_i, y_pred_i)

    def _compute_total_loss(self, y, main, aux1, aux2, aux3):
        L_main = composite_loss(y, main)
        L1 = composite_loss(y, aux1)
        L2 = composite_loss(y, aux2)
        L3 = composite_loss(y, aux3)
        aux_term = (
            0.5 * tf.exp(-2.0 * self.s1) * L1 + self.s1
            + 0.5 * tf.exp(-2.0 * self.s2) * L2 + self.s2
            + 0.5 * tf.exp(-2.0 * self.s3) * L3 + self.s3
        )
        loss = L_main + aux_term
        loss += tf.add_n(self.losses) if self.losses else 0.0
        return loss, L_main, (L1 + L2 + L3) / 3.0

    def _maybe_scale_loss(self, loss):
        if hasattr(self.optimizer, "get_scaled_loss"):
            return self.optimizer.get_scaled_loss(loss), True
        return loss, False

    def _maybe_unscale_grads(self, grads, scaled: bool):
        if scaled and hasattr(self.optimizer, "get_unscaled_gradients"):
            return self.optimizer.get_unscaled_gradients(grads)
        return grads

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            main, aux1, aux2, aux3 = self(x, training=True)
            loss, L_main, L_aux = self._compute_total_loss(y, main, aux1, aux2, aux3)
            scaled_loss, scaled = self._maybe_scale_loss(loss)

        grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self._maybe_unscale_grads(grads, scaled)

        if self.grad_clip_norm > 0:
            g_non = [g for g in grads if g is not None]
            if g_non:
                g_non, _ = tf.clip_by_global_norm(g_non, self.grad_clip_norm)
                it = iter(g_non)
                grads = [next(it) if g is not None else None for g in grads]

        if self.grad_accum_steps == 1:
            self.optimizer.apply_gradients([(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None])
        else:
            if self._accum_grads is None:
                raise RuntimeError("Accum buffers not built")
            for i, g in enumerate(grads):
                if g is not None:
                    self._accum_grads[i].assign_add(tf.cast(g, tf.float32))
            self._accum_step.assign_add(1)

            def _apply():
                mean_grads = []
                mean_vars = []
                for i, v in enumerate(self.trainable_variables):
                    g = self._accum_grads[i] / float(self.grad_accum_steps)
                    mean_grads.append(tf.cast(g, v.dtype))
                    mean_vars.append(v)
                self.optimizer.apply_gradients(list(zip(mean_grads, mean_vars)))
                for gb in self._accum_grads:
                    gb.assign(tf.zeros_like(gb))
                self._accum_step.assign(0)
                return 0

            tf.cond(tf.equal(self._accum_step, self.grad_accum_steps), _apply, lambda: 0)

        self.loss_tracker.update_state(loss)
        self.main_loss_tracker.update_state(L_main)
        self.aux_loss_tracker.update_state(L_aux)

        self.acc.update_state(y, main)
        self.recall.update_state(y, main)
        self._update_iou(y, main)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        main, aux1, aux2, aux3 = self(x, training=False)
        loss, L_main, L_aux = self._compute_total_loss(y, main, aux1, aux2, aux3)

        self.loss_tracker.update_state(loss)
        self.main_loss_tracker.update_state(L_main)
        self.aux_loss_tracker.update_state(L_aux)

        self.acc.update_state(y, main)
        self.recall.update_state(y, main)
        self._update_iou(y, main)

        return {m.name: m.result() for m in self.metrics}
