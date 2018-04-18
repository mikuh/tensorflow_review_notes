# TensorFlow导入数据新方式

以前玩TensorFlow的时候，导入数据都是自己构建迭代器的。现在新的版本有了很多导数据的API，使得导数据更加方便灵活高效,我们来一个一个看。



### 1.  tf.data.Dataset.from_tensor_slices

这个方法能够从内存构建数据集，它的作用就是将传入的数据从第一个维度切割。举两个简单的例子：

```python
# 从 np.array 或者是 list 构建
dataset0 = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [5, 6]]))
print(dataset0.output_types)
print(dataset0.output_shapes)
# 构建迭代器的方式下面再细讲
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
```

当然，传入数据的形式多种多样，还可以是tuple，items或者dict的方式（**第一维的尺寸必须要一致**，不然没法切割）举几个例子：

```python
# tf.TensorFlow tuple
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
        
# tf.Tensor items
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

# tf.Tensor dict
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```