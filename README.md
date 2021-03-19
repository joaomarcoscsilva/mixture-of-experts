# Mixture of Experts
A replication of the paper "Adaptive Mixtures of Local Experts" applied to the CIFAR-10 image classification dataset.

Here are the results on the validation set, in function of the number of experts used:

|![[Section-3-Panel-0-ixvacjhdn.png]]| ![[Section-3-Panel-1-tqll7dzmy.png]]|
| ---------------------------------- |:----------------------------------- |

Here, it is clear that the Mixture of Experts model is capable of increasing generalization performance. However, the gains eventually saturate and then decrease when the number of experts increase, possibly due to difficulties in the optimization procedure. This suggests that there is an optimum amount of experts for the task.

One important caveat is that the experts in this experiment have a very small capacity (only one narrow hidden layer), meaning that these graphs are all mostly in the underfitting regime.

We can also visualize the training dynamics of a few of those networks:

|![[Section-6-Panel-0-6qwqk9ntv.png]]| ![[Section-6-Panel-1-98rd5i0d3.png]]|
| ---------------------------------- |:----------------------------------- |

Interestingly, in this graph, both the 1 and the 4 experts networks converge really fast. The 10 expert network, however, initially performs just as bad as the single expert, and only after a while starts to gradually improve, approaching the 4 expert network. Notice that the 10 expert network has not converged yet after the 30 epochs.

This exemplifies the possible optimization challenges in networks with more experts.
