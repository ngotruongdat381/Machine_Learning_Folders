import numpy as np
import tensorflow as tf

def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)

x = convert(np.array([
    [2, 2, 1, 3],
    [4, 5, 6, -1],
    [0, 1, 1, -2],
    [6, 2, 3, 0]
]))

y = convert(np.array([1, 2, 5, 3, 7]))
z = convert(np.array([1, 1, 4, 6, 2]))

arg_min = tf.argmin(x, 1)
arg_max = tf.argmax(x, 1)
unique = tf.unique(y)
diff = tf.setdiff1d(y, z)

with tf.Session() as session:
    print ("Argmin = ", session.run(arg_min))
    print ("Argmax = ", session.run(arg_max))

    print ("Unique_values = ", session.run(unique)[0])
    print ("Unique_idx = ", session.run(unique)[1])

    print ("Setdiff_values = ", session.run(diff)[0])
    print ("Setdiff_idx = ", session.run(diff)[1])

    print (session.run(diff)[1])