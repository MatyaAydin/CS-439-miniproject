# CS-439-miniproject


### Project description

This project is realized for the CS-439 Optimization for machine learning class taught at EPFL and extends the paper [First-ish Order Methods: Hessian-aware Scalings of Gradient Descent](https://arxiv.org/abs/2502.03701).


### Authors


* Arthur Pollet | SCIPER: 325074
* Matya Aydin | SCIPER: 388895
* Miki Vanoušek | SCIPER: 394827


### Repo structure

```text
CS-439-miniproject/
├── src/
│   ├── optimizers/                  # Optimizers implementations
│   │   ├── gd.py                    # Gradient descent (fixed step)
│   │   ├── adam.py                  # Adam optimizer
│   │   ├── HB.py                    # Heavy-Ball momentum
│   │   ├── mrcg.py                  # Scaled gradient method (MRCG)
│   │   └── LS_GD.py                 # Line-search gradient descent (Armijo)
│   │
│   └── models/                      
│       └── california_model.py      # Linear regression model on California housing
│
├── cifar_graphs_.ipynb              # CIFAR-10 logistic regression – scaled methods
├── cifar_graphs_all_optims.ipynb    # CIFAR-10 logistic regression – all optimizers
├── california_regression.ipynb      # California housing – stochastic optimizer comparison
```

### Reproducing results

All plots used in the report can be obtained running the cifar_graphs.ipynb, cifar_graphs_all_optims.ipynb and stochastic.ipynb notebooks. 


### Dependencies

We recommend using a virtual environment:
```bash

python3 -m venv .venv
.venv\Scripts\activate
```

To get all necessary packages, run the following command:

```bash

pip install -r requirements.txt
```