## Node Dependent Local Smoothing for Scalable Graph Learning



### Requirements

Environments: Xeon Gold 5120 (CPU), 384GB(RAM), TITAN RTX (GPU), Ubuntu 16.04 (OS)

The PyTorch version we use is torch 1.7.1+cu110. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

To install other requirements:

```setup
pip install -r requirements.txt
```



### Training

To train the model(s) in the paper, run this command:

```train
cd src; python train.py --dataset cora/citeseer/pubmed
```

Please refer to the Appendix for the detailed hyperparameters.



### Node Classification Results

1. Transductive Setting:

   ![transductive](./transductive_results.png)

2. Inductive Setting: 

   ![inductive](./inductive_results.png)

3. Efficiency Comparison: 

   ![efficiency](./efficiency.png)
   
