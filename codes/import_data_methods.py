import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()

# 从内存中构建数据集

# 从 array构建
dataset0 = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])
print(dataset0.output_types)
print(dataset0.output_shapes)
iterator = dataset0.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

print("="*10)

# tf.Tensor构建
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

iterator = dataset1.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    next_element = iterator.get_next()
    try:
        while True:
            print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
        print("end!")


print("="*100)


dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 10], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"
iterator = dataset2.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    next_element = iterator.get_next()
    try:
        while True:
            print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
        print("end!")



dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

iterator = dataset3.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    next_element = iterator.get_next()
    try:
        while True:
            print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
        print("end!")

