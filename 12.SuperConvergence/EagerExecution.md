<h1> Eager Execution </h1>
TensorFlow 1.X uses lazy execution, where operations are not run by the framework until asked specifically to do so.
In Tensorflow 2, Eager Execution is introduced where operations are executed eagerly rather than lazy. When we enable eager execution, instead of creating a graph for tensorflow operation, execution happens immediately and the results are available. 

For example

```
import tensorflow as tf
a = tf.constant([1,2,3])
b = tf.constant([0, 0, 1])
c = tf.add(a, b)

print(c)

```

Lazy Execution will return ``` Tensor("Add:0", shape=(3,), dtype=int32) ```

Eager execution will return ```tf.Tensor([1 2 4], shape=(3,), dtype=int32)```

In both cases, the output is a Tensor. In the second case, the operation has been run eagerly and we can observe directly that the Tensor contains the result ([1 2 4]). In the first case, the Tensor contains information about the addition operation (Add:0), but not the result of the operation.

Eager execution makes code easier to debug and to develop. 

### TensorFlow 2 Graphs
```
def compute(a, b, c):    
  d = a * b + c    
  e = a * b * c    
  return d, e
```

Assuming a, b, and c are Tensor matrices, this code computes two new values: d and e. Using eager execution, TensorFlow would compute the value for d and then compute the value for e.

Using lazy execution, TensorFlow would create a graph of operations. Before running the graph to get the result, a graph optimizer would be run. To avoid computing a * b twice, the optimizer would cache the result and reuse it when necessary. For more complex operations, the optimizer could enable parallelism to make computation faster. Both techniques are important when running large and complex models.

As we saw, running in eager mode implies that every operation is run when defined. Therefore, such optimizations cannot be applied. Thankfully, TensorFlow includes a module to work around this—TensorFlow AutoGraph.

TensorFlow 2 and Keras in detail
We have introduced the general architecture of TensorFlow and trained our first model using Keras. Let's now walk through the main concepts of TensorFlow 2. We will explain several core concepts of TensorFlow that feature in this book, followed by some advanced notions. While we may not employ all of them in the remainder of the book, you might find it useful to understand some open source models that are available on GitHub or to get a deeper understanding of the library.

Core concepts
Released in spring 2019, the new version of the framework is focused on simplicity and ease of use. In this section, we will introduce the concepts that TensorFlow relies on and cover how they evolved from version 1 to version 2.

Introducing tensors
TensorFlow takes its name from a mathematical object called a tensor. You can imagine tensors as N-dimensional arrays. A tensor could be a scalar, a vector, a 3D matrix, or an N-dimensional matrix.

A fundamental component of TensorFlow, the Tensor object is used to store mathematical values. It can contain fixed values (created using tf.constant) or changing values (created using tf.Variable).

In this book, tensor denotes the mathematical concept, while Tensor (with a capital T) corresponds to the TensorFlow object.
Each Tensor object has the following:

Type: string, float32, float16, or int8, among others.
Shape: The dimensions of the data. For instance, the shape would be () for a scalar, (n) for a vector of size n, and (n, m) for a 2D matrix of size n × m.
Rank: The number of dimensions, 0 for a scalar, 1 for a vector, and 2 for a 2D matrix.
Some tensors can have partially unknown shapes. For instance, a model accepting images of variable sizes could have an input shape of (None, None, 3). Since the height and the width of the images are not known in advance, the first two dimensions are set to None. However, the number of channels (3, corresponding to red, blue, and green) is known and is therefore set.

TensorFlow graphs
TensorFlow uses tensors as inputs as well as outputs. A component that transforms input into output is called an operation. A computer vision model is therefore composed of multiple operations.

TensorFlow represents these operations using a directed acyclic graph (DAC), also referred to as a graph. In TensorFlow 2, graph operations have disappeared under the hood to make the framework easier to use. Nevertheless, the graph concept remains important to understand how TensorFlow really works.

When building the previous example using Keras, TensorFlow actually built a graph:



Figure 2.3: A simplified graph corresponding to our model. In practice, each node is composed of smaller operations (such as matrix multiplications and additions)
While very simple, this graph represents the different layers of our model in the form of operations. Relying on graphs has many advantages, allowing TensorFlow to do the following:

Run part of the operations on the CPU and another part on the GPU
Run different parts of the graph on different machines in the case of a distributed model
Optimize the graph to avoid unnecessary operations, leading to better computational performance
Moreover, the graph concept allows TensorFlow models to be portable. A single graph definition can be run on any kind of device.

In TensorFlow 2, graph creation is no longer handled by the user. While managing graphs used to be a complex task in TensorFlow 1, the new version greatly improves usability while still maintaining performance. In the next section, we will peek into the inner workings of TensorFlow and briefly explore how graphs are created.

Comparing lazy execution to eager execution
The main change in TensorFlow 2 is eager execution. Historically, TensorFlow 1 always used lazy execution by default. It is called lazy because operations are not run by the framework until asked specifically to do so.

Let's start with a very simple example to illustrate the difference between lazy and eager execution, summing the values of two vectors:

Copy
import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([0, 0, 1])
c = tf.add(a, b)

print(c)
Note that tf.add(a, b) could be replaced by a + b since TensorFlow overloads many Python operators.

The output of the previous code depends on the TensorFlow version. With TensorFlow 1 (where lazy execution is the default mode), the output would be this:

Copy
Tensor("Add:0", shape=(3,), dtype=int32)
However, with TensorFlow 2 (where eager execution is the default mode), you would get the following output:

Copy
tf.Tensor([1 2 4], shape=(3,), dtype=int32)
In both cases, the output is a Tensor. In the second case, the operation has been run eagerly and we can observe directly that the Tensor contains the result ([1 2 4]). In the first case, the Tensor contains information about the addition operation (Add:0), but not the result of the operation.

In eager mode, you can access the value of a Tensor by calling the .numpy() method. In our example, calling c.numpy() returns [1 2 4] (as a NumPy array).

In TensorFlow 1, more code would be needed to compute the result, making the development process more complex. Eager execution makes code easier to debug (as developers can peak at the value of a Tensor at any time) and easier to develop. In the next section, we will detail the inner workings of TensorFlow and look at how it builds graphs.

Creating graphs in TensorFlow 2
We'll start with a simple example to illustrate graph creation and optimization:

Copy
def compute(a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e
Assuming a, b, and c are Tensor matrices, this code computes two new values: d and e. Using eager execution, TensorFlow would compute the value for d and then compute the value for e.

Using lazy execution, TensorFlow would create a graph of operations. Before running the graph to get the result, a graph optimizer would be run. To avoid computing a * b twice, the optimizer would cache the result and reuse it when necessary. For more complex operations, the optimizer could enable parallelism to make computation faster. Both techniques are important when running large and complex models.

As we saw, running in eager mode implies that every operation is run when defined. Therefore, such optimizations cannot be applied. Thankfully, TensorFlow includes a module to work around this—TensorFlow AutoGraph.

### TensorFlow AutoGraph and tf.function
The TensorFlow AutoGraph module makes it easy to turn eager code into a graph, allowing automatic optimization. To do so, the easiest way is to add the tf.function decorator on top of your function:

When we call the compute function for the first time, TensorFlow will transparently create the following graph:

![Alt text](/path/to/image.jpg)

The graph automatically generated by TensorFlow when calling the compute function for the first time
TensorFlow AutoGraph can convert most Python statements, such as for loops, while loops, if statements, and iterations. Thanks to graph optimizations, graph execution can sometimes be faster than eager code. More generally, AutoGraph should be used in the following scenarios:

  > When the model needs to be exported to other devices
  > When performance is paramount and graph optimizations can lead to speed improvements
  
Another advantage of graphs is their automatic differentiation. Knowing the full list of operations, TensorFlow can easily compute the gradient for each variable.

However, since, in eager mode, each operation is independent from one another, automatic differentiation is not possible by default. Thankfully, TensorFlow 2 provides a way to perform automatic differentiation while still using eager mode—the gradient tape.

### Backpropagating errors using the gradient tape
The gradient tape allows easy backpropagation in eager mode. To illustrate this, we will use a simple example. Let's assume that we want to solve the equation A × X = B, where A and B are constants. We want to find the value of X to solve the equation. To do so, we will try to minimize a simple loss, abs(A × X - B).

In code, this translates to the following:

```
A, B = tf.constant(3.0), tf.constant(6.0)
X = tf.Variable(20.0) # In practice, we would start with a random value
loss = tf.math.abs(A * X - B)
```

Now, to update the value of X, we would like to compute the gradient of the loss with respect to X. However, when printing the content of the loss, we obtain the following:

<tf.Tensor: id=18525, shape=(), dtype=float32, numpy=54.0>

In eager mode, TensorFlow computed the result of the operation instead of storing the operation! With no information on the operation and its inputs, it would be impossible to automatically differentiate the loss operation.

That is where the gradient tape comes in handy. By running our loss computation in the context of tf.GradientTape, TensorFlow will automatically record all operations and allow us to replay them backward afterward:

def train_step():    with tf.GradientTape() as tape:        loss = tf.math.abs(A * X - B)    dX = tape.gradient(loss, X)        print('X = {:.2f}, dX = {:2f}'.format(X.numpy(), dX))    X.assign(X - dX)for i in range(7):    train_step()

The previous code defines a single training step. Every time train_step is called, the loss is computed in the context of the gradient tape. The context is then used to compute the gradient. The X variable is then updated. Indeed, we can see X converging toward the value that solves the equation:

You will notice that in the very first example of this chapter, we did not make use of the gradient tape. This is because Keras models encapsulate training inside the .fit() function—there's no need to update the variables manually. Nevertheless, for innovative models or when experimenting, the gradient tape is a powerful tool that allows automatic differentiation without much effort.
