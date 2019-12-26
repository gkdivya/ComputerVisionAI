<h1> Eager Execution </h1>
TensorFlow 1 used lazy execution. It is called lazy because operations are not run by the framework until asked specifically to do so.

Tensorflow 2, Eager Execution is introduced where operations are executed eagerly rather than lazy. When we enable eager execution, instead of creating a graph for tensorflow operation, execution happens immediately and the results are available. 

For example
'''
import tensorflow as tf
a = tf.constant([1,2,3])
b = tf.constant([0, 0, 1])
c = tf.add(a, b)

print(c)
'''

Lazy Execution will return ''' Tensor("Add:0", shape=(3,), dtype=int32) '''
Eager execution will return '''tf.Tensor([1 2 4], shape=(3,), dtype=int32)'''

In both cases, the output is a Tensor. In the second case, the operation has been run eagerly and we can observe directly that the Tensor contains the result ([1 2 4]). In the first case, the Tensor contains information about the addition operation (Add:0), but not the result of the operation.

In lazy execution, more code would be needed to compute the result, making the development process more complex. Eager execution makes code easier to debug and to develop.

In lazy execution, the purpose of havig a computational graph and executing in a session was to make code scalable. In eager execution, we have to use @tf.function decorator to make the code execute in distributed mode.

