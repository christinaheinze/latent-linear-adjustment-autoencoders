import tensorflow as tf


def penalty(model, predictions, inputs, training):
    error = inputs - model.decode(predictions, training=training)["output"]
    penalty = tf.reduce_mean(tf.reduce_sum(tf.square(error), 
        axis=[1, 2, 3]))
       
    return penalty
