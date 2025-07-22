# autoformer_layers.py
# Contains the Keras layer implementations for the Autoformer model.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import config # <<< NEW: Import config to access the regularization factor

class AutoCorrelation(layers.Layer):
    """
    AutoCorrelation mechanism, computes dependencies based on series periodicity.
    """
    def __init__(self, factor=1, seq_len=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        if seq_len is None:
            raise ValueError("seq_len must be provided to AutoCorrelation layer.")
        self.k = int(self.factor * math.log(seq_len))
        self.k = max(1, min(self.k, seq_len))

    def call(self, queries, keys, values, B):
        L = tf.shape(queries)[1]
        
        queries_fft = tf.signal.rfft(queries, fft_length=[L])
        keys_fft = tf.signal.rfft(keys, fft_length=[L])
        x_corr_fft = tf.math.conj(queries_fft) * keys_fft
        x_corr = tf.signal.irfft(x_corr_fft, fft_length=[L])
        
        x_corr_mean = tf.reduce_mean(x_corr, axis=(2, 3))
        top_k_values, top_k_indices = tf.math.top_k(x_corr_mean, k=self.k)
        
        weights = tf.nn.softmax(top_k_values, axis=-1)
        
        batch_indices = tf.tile(tf.range(B)[:, tf.newaxis], [1, self.k])
        gather_indices = tf.stack([batch_indices, top_k_indices], axis=-1)
        
        gathered_values = tf.gather_nd(values, gather_indices)
        
        weights_reshaped = weights[:, :, tf.newaxis, tf.newaxis]
        
        aggregated_values = tf.reduce_sum(weights_reshaped * gathered_values, axis=1)
        
        return aggregated_values, None

class AutoCorrelationLayer(layers.Layer):
    """Wrapper layer for AutoCorrelation. Fully static and graph-compatible."""
    def __init__(self, d_model, n_heads, seq_len, d_keys=None, d_values=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_values = d_values
        self.n_heads = n_heads

        self.inner_correlation = AutoCorrelation(seq_len=seq_len)
        
        # <<< MODIFIED: Applied L2 regularization to all projection layers.
        l2_reg = keras.regularizers.l2(config.L2_REG_FACTOR)
        self.query_projection = layers.Dense(d_keys * n_heads, kernel_regularizer=l2_reg)
        self.key_projection = layers.Dense(d_keys * n_heads, kernel_regularizer=l2_reg)
        self.value_projection = layers.Dense(d_values * n_heads, kernel_regularizer=l2_reg)
        self.out_projection = layers.Dense(d_model, kernel_regularizer=l2_reg)
        
        self.projection_layer = layers.Dense(seq_len * d_model)
        self.reshape_layer = layers.Reshape((seq_len, d_model))

    def call(self, queries, keys, values):
        B = tf.shape(queries)[0]
        L = tf.shape(queries)[1]
        S = tf.shape(keys)[1]
        H = self.n_heads

        queries_reshaped = tf.reshape(self.query_projection(queries), [B, L, H, -1])
        keys_reshaped = tf.reshape(self.key_projection(keys), [B, S, H, -1])
        values_reshaped = tf.reshape(self.value_projection(values), [B, S, H, -1])
        
        out, attn = self.inner_correlation(queries_reshaped, keys_reshaped, values_reshaped, B)
        
        out_reshaped = tf.reshape(out, [B, H * self.d_values])
        
        out_projected = self.projection_layer(out_reshaped)
        final_out = self.reshape_layer(out_projected)
        
        return final_out

class SeasonalTrendDecompositionBlock(layers.Layer):
    """The core block of Autoformer, performing seasonal-trend decomposition."""
    def __init__(self, d_model, n_heads, d_ff, dropout, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seasonal_part = AutoCorrelationLayer(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        # <<< MODIFIED: Applied L2 regularization to the feed-forward trend layers.
        l2_reg = keras.regularizers.l2(config.L2_REG_FACTOR)
        self.trend_part1 = layers.Dense(d_ff, activation='relu', kernel_regularizer=l2_reg)
        self.trend_part2 = layers.Dense(d_model, kernel_regularizer=l2_reg)
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.moving_avg_res = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')

    def call(self, x):
        trend_init = self.moving_avg_res(x)
        seasonal_init = x - trend_init
        
        seasonal_output = self.seasonal_part(seasonal_init, seasonal_init, seasonal_init)
        seasonal_output = self.norm1(seasonal_init + self.dropout1(seasonal_output))
        
        trend_output = self.trend_part1(seasonal_output)
        trend_output = self.trend_part2(trend_output)
        trend_output = self.norm2(seasonal_output + self.dropout2(trend_output))
        
        final_output = trend_init + trend_output
        return final_output