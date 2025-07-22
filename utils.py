# utils.py

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================================
# === CUSTOM KERAS COMPONENTS (FULLY VERIFIED - FINAL) ====================
# =========================================================================

@keras.utils.register_keras_serializable()
class ProbabilisticHead(keras.Model):
    def __init__(self, kl_weight, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        kl_divergence_function = lambda q, p, ignore: self.kl_weight * tfp.distributions.kl_divergence(q, p)
        self.dense_reparam = tfp.layers.DenseReparameterization(
            units=tfp.layers.IndependentNormal.params_size(1),
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=kl_divergence_function,
        )
        self.output_dist = tfp.layers.IndependentNormal(1)
    def call(self, inputs): return self.output_dist(self.dense_reparam(inputs))
    def get_config(self):
        config = super().get_config()
        config.update({'kl_weight': self.kl_weight})
        return config

def nll(y_true, y_pred_dist): return -y_pred_dist.log_prob(y_true)

@keras.utils.register_keras_serializable()
class AutoCorrelation(layers.Layer):
    def __init__(self, d_model, factor=1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.d_model = d_model
    def time_delay_agg_training(self, values, corr):
        head = tf.cast(tf.shape(values)[1], tf.float32)
        top_k = tf.cast(self.factor * tf.math.log(head), tf.int32)
        mean_value = tf.reduce_mean(tf.math.top_k(corr, top_k, sorted=False).values, axis=-1)
        mean_value = tf.expand_dims(mean_value, axis=-1)
        d_model = tf.shape(corr)[-1]
        mean_value = tf.tile(mean_value, [1, 1, d_model])
        corr = tf.stack([corr, mean_value], axis=-1)
        weights = tf.nn.softmax(corr, axis=-1)
        tmp_corr = tf.transpose(tf.roll(tf.transpose(values, perm=[0, 2, 1]), shift=-1, axis=2), perm=[0, 2, 1])
        all_values = tf.stack([values, tmp_corr], axis=-1)
        aggregated_values = tf.reduce_sum(all_values * weights, axis=-1)
        return aggregated_values
    def call(self, query, key, value):
        q_fft = tf.signal.rfft(query, fft_length=[self.d_model])
        k_fft = tf.signal.rfft(key, fft_length=[self.d_model])
        corr = tf.signal.irfft(q_fft * tf.math.conj(k_fft), fft_length=[self.d_model])
        return self.time_delay_agg_training(value, corr)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "factor": self.factor})
        return config

@keras.utils.register_keras_serializable()
class AutoCorrelationLayer(layers.Layer):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, **kwargs):
        super().__init__(**kwargs)
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model, self.n_heads = d_model, n_heads
        self.query_projection = layers.Dense(d_keys * n_heads)
        self.key_projection = layers.Dense(d_keys * n_heads)
        self.value_projection = layers.Dense(d_values * n_heads)
        self.out_projection = layers.Dense(d_model)
        # --- THE FIX ---
        # The d_model from the constructor MUST be passed to the inner layer.
        self.inner_correlation = AutoCorrelation(d_model=self.d_model)
        # --- END FIX ---
    def call(self, query, key, value):
        queries, keys, values = self.query_projection(query), self.key_projection(key), self.value_projection(value)
        out = self.inner_correlation(queries, keys, values)
        return self.out_projection(out)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads})
        return config

@keras.utils.register_keras_serializable()
class SeasonalTrendDecompositionBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, moving_avg_kernel_size=25, seq_len=96, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.dropout, self.moving_avg_kernel_size, self.seq_len = dropout, moving_avg_kernel_size, seq_len
        self.correlation = AutoCorrelationLayer(d_model=d_model, n_heads=n_heads)
        self.conv1, self.conv2 = layers.Conv1D(filters=d_ff, kernel_size=1, activation='relu'), layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-5), layers.LayerNormalization(epsilon=1e-5)
        self.dropout_layer, self.moving_avg = layers.Dropout(dropout), layers.AveragePooling1D(pool_size=moving_avg_kernel_size, strides=1, padding='same')
    def call(self, x, training=False):
        trend_init = self.moving_avg(x)
        seasonal_init = x - trend_init
        seasonal_output = self.correlation(seasonal_init, seasonal_init, seasonal_init)
        seasonal_output = self.dropout_layer(seasonal_output, training=training)
        seasonal_part = self.norm1(seasonal_init + seasonal_output)
        trend_output = self.conv2(self.dropout_layer(self.conv1(trend_init), training=training))
        trend_output = self.dropout_layer(trend_output, training=training)
        trend_part = self.norm2(trend_init + trend_output)
        return trend_part + seasonal_part
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff,
            "dropout": self.dropout, "moving_avg_kernel_size": self.moving_avg_kernel_size,
            "seq_len": self.seq_len
        })
        return config

# =========================================================================
# === DATA HANDLING & GENERATOR FUNCTIONS =================================
# =========================================================================

def create_sequences(data, seq_length, target_idx, is_df=False):
    if is_df: data = data.to_numpy()
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, target_idx])
    return np.array(xs), np.array(ys)

def autoformer_generator(data, seq_len, pred_len, target_idx, batch_size):
    n_features = data.shape[-1]
    start_index = 0
    while True:
        x_encoder_batch, x_decoder_batch, y_batch = [], [], []
        for _ in range(batch_size):
            if start_index + seq_len + pred_len > len(data): start_index = 0
            enc_start, enc_end = start_index, start_index + seq_len
            dec_start, dec_end = enc_end - (seq_len // 2), enc_end
            y_end = enc_end + pred_len
            x_encoder_batch.append(data[enc_start:enc_end])
            decoder_known_part = data[dec_start:dec_end]
            decoder_padding = np.zeros((pred_len, n_features))
            x_decoder_batch.append(np.concatenate([decoder_known_part, decoder_padding], axis=0))
            y_batch.append(data[enc_end:y_end, target_idx])
            start_index += 1
        yield (np.array(x_encoder_batch), np.array(x_decoder_batch)), np.array(y_batch).reshape(batch_size, pred_len, 1)

def inverse_transform(data, scaler, target_idx, n_features):
    data = np.array(data).reshape(-1, 1)
    dummy = np.zeros(shape=(len(data), n_features))
    dummy[:, target_idx] = data[:, 0]
    return scaler.inverse_transform(dummy)[:, target_idx]