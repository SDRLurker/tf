# -*- coding: utf-8 -*-
import tensorflow as tf
import time

# 학습 데이터
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b that compute y_data = W * x_data + b
# y_data = W * x_data + b를 만족하는 W와 b를 찾습니다.
# ( We know that W should be 1 and b 0, but Tensorflow will
# figure out that out for us. )
# ( 우리는 W가 1이고 b가 0이 되어야 하는 걸 알지만, Tensorflow가
# 이 값을 학습 알고리즘을 통해 찾을 것입니다. ) 
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
# 우리의 가설함수 h(x)
hypothesis = W * X + b

# Simplified cost function
# cost 함수를 다음처럼 표현합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
# cost를 최소화하기 위해 경사 하강법 알고리즘을 사용합니다.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables, We will 'run' this first.
# 시작하기 전에 모든 변수들을 초기화 합니다. 
init = tf.initialize_all_variables()

# Launch the graph.
# 데이터 흐름 그래프를 계산하기 위해 세션을 시작합니다.
sess = tf.Session()
sess.run(init)

# Fit the line.
print "경사 하강법 알고리즘으로 학습을 시작합니다..."
time.sleep(3)
for step in xrange(2001):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if step % 20 == 0:
		print "%5s %10s %10s %10s" % ("step","cost","W","b")
		print "%5d %10.4f %10.4f %10.4f" % (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

# Learns best fit is W: [1], b: [0]
print "학습 알고리즘 h(x)를 바탕으로 다음 평수의 집 가격을 예측합니다..."
print "X(집 평수)가 5일 때 Y(집 가격)은 %10.4f입니다." % sess.run(hypothesis, feed_dict={X:5})
print "X(집 평수)가 2.5일 때 X(집 가격)은 %10.4f입니다." % sess.run(hypothesis, feed_dict={X:2.5})
