import math
import numpy as np
import keras
import keras as ks
import tensorflow as tf
import transformers as tff
from keras import Input, Model
from keras.src.layers import Dense
from keras.src.ops import shape
from numpy.random import normal
from transformers import AutoModel


class MixtureOfExpertsLayer(tf.keras.layers.Layer):
    def __init__(self, numOfExperts, parametersSize):
        super(MixtureOfExpertsLayer, self).__init__()
        self.numOfExperts = tf.Variable(numOfExperts, True, False, None, "NumberOfExperts")
        self.parameterSize = parametersSize
        self.experts = []
        self.gatingNetwork = Dense(self.numOfExperts, activation='softmax')
    def build(self, input_shape):
        self.experts = [Dense(self.parameterSize) for _ in range(self.numOfExperts)]
    def call(self, inputs):
        gate_values = self.gatingNetwork(inputs)
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=-1)
        output = tf.reduce_sum(expert_outputs * tf.expand_dims(gate_values, axis=-2), axis=-1)
        return output

class NoiseCancellingAttentionMechanism(tf.keras.layers.Layer):
    def __init__(self, model, num_heads, parameter):
        super(NoiseCancellingAttentionMechanism, self).__init__()
        self.heads = num_heads
        self.model = model
        self.depth = model // num_heads
        self.weightedQuery = Dense(model)
        self.weightedKey = Dense(model)
        self.weightedValue = Dense(model)
        parameter = tf.Tensor(parameter)
        self.parameter = tf.Variable(parameter, True, True, None, "stride")
        self.parameterMask = tf.boolean_mask(self.parameter, ([True, True, False]))
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def __call__(self, v, k, q1, q2):
        batch_size = tf.shape(q1)[0]
        q1 = self.split_heads(self.weightedQuery(q1), batch_size)
        q2 = self.split_heads(self.weightedQuery(q2), batch_size)
        k = self.split_heads(self.weightedKey(k), batch_size)
        v = self.split_heads(self.weightedValue(v), batch_size)
        matmul_q1k = tf.matmul(q1, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, tf.float32))
        matmul_q2k = tf.matmul(q2, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, tf.float32))
        attention_q1 = tf.nn.softmax(matmul_q1k, axis=-1)
        attention_q2 = tf.nn.softmax(matmul_q2k, axis=-1)
        diff_attention = attention_q1 - (self.parameter * attention_q2)
        output = tf.matmul(diff_attention, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        final_output = Dense(self.d_model)(concat_attention)
        return final_output


def define_block(expertNumber: int, parameterSize: int, num_heads, model_dim) -> []:
    numOfExperts = tf.Tensor(expertNumber)
    parameterSizeArray = tf.Tensor(parameterSize)
    query = keras.Input(shape=(None, 512))
    value = keras.Input(shape=(None, 512))
    query_mask = tf.constant([True, True, False])
    value_mask = tf.constant([True, False, True])
    layers = []
    att_layer = keras.layers.GroupQueryAttention(model_dim, num_heads)
    layers.append(att_layer(query, value, attention_mask=query_mask))
    attention_output = att_layer(query, value)
    normal_layer = keras.layers.LayerNormalization(axis=-1)
    normalized_out = normal_layer(attention_output)
    layers.append(normal_layer(query))
    noise_cancelling_att = NoiseCancellingAttentionMechanism(model_dim, num_heads, parameter=tf.zeros([64, 64]))
    cancelling_output = noise_cancelling_att(query, value, query, query)
    layers.append(noise_cancelling_att)
    forward_pass = MixtureOfExpertsLayer(expertNumber, parameterSize)
    layers.append(forward_pass)
    forward_pass_output = forward_pass(query)
    output = tf.concat([normalized_out, cancelling_output, forward_pass_output], -1, "concat")
    return Model(inputs=[query, value], outputs=output)
def build_model(num_blocks, num_experts, expert_param_size, model_dim, num_heads):
    inputs = Input(shape=(None, model_dim))
    x = inputs
    for _ in range(num_blocks):
        block = define_block(num_experts, expert_param_size, model_dim, num_heads)
        x = block(x)
    model = Model(inputs=inputs, outputs=x)
    return model



