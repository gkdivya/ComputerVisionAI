<h1> Eager Execution </h1>
Eager Execution runtime in tensorflow, operations are executed eagerly rather than lazy. When we enable eager execution, instead of creating a graph for tensorflow operation, execution happens immediately and the results are available. 

In lazy execution, the purpose of havig a computational graph and executing in a session was to make code scalable. In eager execution, we have to use @tf.function decorator to make the code execute in distributed mode.

