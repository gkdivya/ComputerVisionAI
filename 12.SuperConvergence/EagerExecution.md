<h1> Eager Execution </h1>
TensorFlow 1.X uses lazy execution, where operations are not run by the framework until asked specifically.
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

As we saw, running in eager mode implies that every operation is run when defined. Therefore, such optimizations cannot be applied. Thankfully, TensorFlow 2 includes a module to work around this: TensorFlow AutoGraph.

### TensorFlow AutoGraph
The TensorFlow AutoGraph module makes it easy to turn eager code into a graph, allowing automatic optimization. To do so, the easiest way is to add the tf.function decorator on top of the function:

```
@tf.function
def compute(a, b, c):    
  d = a * b + c    
  e = a * b * c    
  return d, e
```
When we call the compute function for the first time, TensorFlow will transparently create the following graph:

![Compute Graph](/12.SuperConvergence/assets/Graph.png)


TensorFlow AutoGraph can convert most Python statements, such as for loops, while loops, if statements, and iterations. Thanks to graph optimizations, graph execution can sometimes be faster than eager code. More generally, AutoGraph should be used in the following scenarios:

  > * When the model needs to be exported to other devices
  > * When performance is paramount and graph optimizations can lead to speed improvements
  
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
```
def train_step():    
  with tf.GradientTape() as tape:        
    loss = tf.math.abs(A * X - B)    
  dX = tape.gradient(loss, X)        
    
  print('X = {:.2f}, dX = {:2f}'.format(X.numpy(), dX))    
  X.assign(X - dX)
    
for i in range(7):    
  train_step()
```
The previous code defines a single training step. Every time train_step is called, the loss is computed in the context of the gradient tape. The context is then used to compute the gradient. The X variable is then updated. 
 >  X = 20.00, dX = 3.000000  
 >  X = 17.00, dX = 3.000000   
 >  X = 14.00, dX = 3.000000   
 >  X = 11.00, dX = 3.000000   
 >  X = 8.00, dX = 3.000000   
 >  X = 5.00, dX = 3.000000   
 >  X = 2.00, dX = 0.000000  

Indeed, we can see X converging toward the value that solves the equation. Gradient tape is a powerful tool that allows automatic differentiation without much effort.


