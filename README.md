## Updated Mar 07th.

In this version, the codes are modularized. It remains to include local data loading as a defined function.

---

The codes by default turn on cuda (GPU) to speed up the training and testing evaluations.

In the code, I call the proposed algorithm as ``AMFED`` (Adversarial Memory Federated). This is for ease of exposition only. 

*Note that the capitalization of the arguments letters in 'Environmental Parameters' should match exactly the ones provided in this doc.*

### Pre-installation
The code is based on Python. In the first run, it will download MNIST or CIFAR10 datasets from torch vision. 
#### ML Part
**pytorch with cuda.**

I did not consider the ``cpu`` version, it is recommended to run on a cluster.

#### Visulization
**pandas**
**tensorboard**
**matplotlib**

The results, including train loss, test accuracy, are recorded through tensorboard SummaryWriter.

Although the results are printed out in each iteration, it requires extra efforts to read out the results from ``./runs`` folder.


### Code setup

**Environmental Parameters**

*Learning*

* --model_name : the neural network to be evaluated. We have the following options: 
  * Logistic: multinomial logistic classification as a convex function
  * lenet : LeNet-5
  * The other optional arguments are: 
    * mlp: Multilayer Perceptron
    * VGG11BN: VGG11
    * ResNet18
* --algorithm : the proposed and baseline algorithms
  * AMFED: the proposed algorithm with FedAvg type updates;
  * MIFA: the algorithm from Longbo Huang's paper;
  * FEDAVG: the biased FedAvg Updates;
  * AMFED_Prox: the proposed algorithm with FedProx type updates;
  * FEDAVG_Prox: the regular FedProx algorithm.
* --weight_decay: 1e-3 by default.
* --step_decay: linear for '1/T' and sqrt for '1/sqrt(T)' the indicator of a decaying step size or a constant step size;
* --dataset_name: MNIST or CIFAR10
* --batch_size: mini-batch size of the mini-batch stochastic gradients

*Algorithm Specific Parameters*

* --mu: the Proximal parameter $\mu$ in FedProx and AMFED_Prox
* --speedup: the indicator of $\beta$ parameter in the paper. $0$ for $\beta=1$, and $1$ for $\beta=1/\sqrt{p}$, where $p$ is the sample fraction.

*Federated System*

* --size : the number of clients
* --sample_frac : the fraction to be sampled. It can be calculated by ``the number of sampled clients/ size``
* --sample_strategy: 'uar' for uniformly at random and 'withp' for with probability p.
* --communication_round
* --local_steps: the argument to override the default settings and set a constant local steps
* --local_epochs: 2 by default. The multiplicative factor to the minimum nuber of the batches.
* --imbalanced: clients with balanced dataset or not.
* --iid: 1 for IID, 2 for non-IID with each client holding images from the given number of classes
* --class_per_client: 2 by default, the given number of classes that each client holds;

*Adversarial Dropout*

* --alpha: the parameter to control the dropout fraction in *Scheme 2*.
* --dropout: alternate for dropout *Scheme 1*, and forever for dropout *Scheme 2*.

*Coding*

* --seed: random seed
* --n_swap: the indicator to load historical best performance model;
* --gpu: the argument to switch on or off the gpu. However, this work is by default with cuda, so this argument is merely a placeholder.



**Dropout Schemes**

We consider two adversarial dropout schemes, where the dropout fraction is not known to the PS or the clients in each iteration:

* Alternating dropout

  In this scheme, we have two parameters, which we call as "repetition" and "interval".

  * Repetition defines the number of slices. For example, we have 100 clients and 4 repetitions, then the clients will be grouped into four slices of each 25 clients.
  * Interval is a parameter to control the the unavailable communication rounds. For instance, if '--interval 100', for every 100 communication rounds, the given group will become unavailable and thus drop out whenever a connection request from the PS raises.

* Forever Dropout

  In this scheme, the only related parameter is $\alpha$, which defines the fraction of clients to be randomly selected and thus dropped.
  


### How to read from the running results?

Use the ``read.ipynb`` Jupyter notebook. 
However, be mindful of the decimal format.

### Sample Bash Script
``python -m main --size 100 --dataset_name CIFAR10 --model_name lenet --algorithm AMFED --sample_frac 0.1 --alpha 0.1 -imbalanced 1 --seed 123 --iid 2 --dropout 2 --class_per_client 2 --learning_rate 0.1 --weight_decay 1e-3 --batch_size 5 ``

``python -m main --size 100 --dataset_name MNIST --model_name Logistic --algorithm AMFED --sample_frac 0.1 --alpha 0.1 -imbalanced 1 --seed 123 --iid 2 --dropout 2 --class_per_client 2 --learning_rate 0.1 --weight_decay 1e-3 --batch_size 10 ``