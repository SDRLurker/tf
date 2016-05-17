import tensorflow as tf

# Start tf session
sess = tf.Session()

# Basic constant operations
# The value returned by the construct represents the output of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

c = a+b

# Print out operation
print c

# print out the result of operation
print sess.run(c)
