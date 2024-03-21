# Import necessary libraries
import argparse
import os.path
import warnings

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchmetrics.classification import Accuracy

from block import Block
from custom.callback import moo_callback
from custom.display import moo_display
from gfo import GradientFreeOptimization
from model import model_loader


# Define the dataset preprocessing function
def preprocess_data(dataset, seed=42, **kwargs):
    # Your preprocessing code here
    rng = np.random.default_rng(seed)
    forget_indices = rng.choice(np.arange(start=dataset.__len__()-10000, stop=dataset.__len__()), 500)
    # Create a subset of the original dataset using the balanced indices
    forget_dataset = Subset(dataset, forget_indices)
    # Create DataLoaders for the balanced datasets
    forget_data_loader = DataLoader(
        forget_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    # Create an empty list to store the balanced dataset
    retain_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(10):
        class_indices = np.where(np.array(dataset.targets[:dataset.__len__()-10000]) == i)[0]
        selected_indices = rng.choice(class_indices, 1000, replace=False)
        retain_indices.extend(selected_indices)
    np.savez("data/cifar-10-selected-data-indices.npz", forget=forget_indices, retain=retain_indices)
    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(dataset, retain_indices)
    # Create DataLoaders for the balanced datasets
    retain_data_loader = DataLoader(
        balanced_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    return forget_data_loader, retain_data_loader


# Define the ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18()
        self.fc = nn.Linear(1000, 10)  # Modify output for CIFAR-10

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


# Define the unlearning problem
class UnlearningProblem(Problem):
    def __init__(self, n_var=1, model=None, gfo=None, block=None, forget_data=None, retain_data=None, num_classes=10):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=-1, xu=1)
        self.model = model
        self.block = block
        self.gfo = gfo
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.num_classes = num_classes

    def _calc_pareto_front(self, n_pareto_points=100):

        x = np.linspace(0, 1, n_pareto_points)
        # return np.array([np.ones(n_pareto_points), x]).T

        return np.array([x, 1 - np.sqrt(x)]).T
        # x = np.linspace(1, 0, n_pareto_points)
        # return np.array([x, np.sqrt(x)]).T
        # return np.ones((n_pareto_points, 2))

    def _evaluate(self, X, out, *args, **kwargs):
        # x is the unlearning parameter
        # Train the ResNet-18 model on the modified dataset with unlearning parameter x
        NP = X.shape[0]
        obj1 = np.zeros(NP)
        obj2 = np.zeros(NP)
        for i in range(X.shape[0]):
            xi = X[i]
            uxi = xi.copy()
            if len(xi) != self.block.dims:
                uxi, = self.block.unblocker(np.array([xi]))
            self.model.load_state_dict(self.gfo.set_model_state(self.model.state_dict(), uxi))
            self.model.to(self.gfo.DEVICE)
            self.model.eval()
            metric = Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(self.gfo.DEVICE)
            forget_accuracies = []
            retain_accuracies = []
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(self.forget_data):
                    data, label = data.to(self.gfo.DEVICE), label.to(self.gfo.DEVICE)
                    preds = self.model(data).to(self.gfo.DEVICE)

                    acc = metric(preds, label).cpu().detach().numpy()
                    forget_accuracies.append(acc)
                for batch_idx, (data, label) in enumerate(self.retain_data):
                    data, label = data.to(self.gfo.DEVICE), label.to(self.gfo.DEVICE)
                    preds = self.model(data).to(self.gfo.DEVICE)

                    acc = metric(preds, label).cpu().detach().numpy()
                    retain_accuracies.append(acc)
            # Calculate objectives
            obj1[i] = np.mean(forget_accuracies)  # Objective 1: Accuracy on forget data
            obj2[i] = np.mean(retain_accuracies)  # Objective 2: Retained accuracy on retain data
            # print(obj1[i], obj2[i])
        out["F"] = np.column_stack([obj1, 1 - obj2])


def main(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ]
    )
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Preprocess the CIFAR-10 dataset
    forget_loader, retain_loader = preprocess_data(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    network, w = model_loader("resnet", 'CIFAR10')
    gfo = GradientFreeOptimization(dataset='CIFAR10', model_name="resnet", model_save_path=args.model_path,
                                   network=network, weights=w, num_classes=10)
    model_params = gfo.get_parameters(gfo.model)
    dims = len(model_params)
    block = Block(gfo=gfo, scheme='1bin', block_file=args.block_path, dims=dims)
    dims = block.blocked_dims

    # Define the unlearning problem
    problem = UnlearningProblem(n_var=dims, model=gfo.model, gfo=gfo, block=block, forget_data=forget_loader,
                                retain_data=retain_loader)

    out = {"F": None}
    problem._evaluate(np.array([model_params]), out)
    F = out["F"]
    gb_f1, gb_f2 = F[0, 0], F[0, 1]
    gb_test_f1score = gfo.evaluate_params(model_params, data_loader=test_loader)

    (block_solution,) = block.blocker(np.array([model_params]))
    out = {"F": None}
    problem._evaluate(np.array([block_solution]), out)
    F = out["F"]
    block_gb_f1, block_gb_f2 = F[0, 0], F[0, 1]
    block_gb_test_f1score = gfo.evaluate_params(block.unblocker(np.array([block_solution]))[0], data_loader=test_loader)

    print(f"Baseline test F1-score: {gb_test_f1score}, Block Baseline test F1-score: {block_gb_test_f1score}")

    # Define the NSGA-II algorithm
    init_pop = gfo.block_population_init(pop_size=100, block=block)
    algorithm = NSGA2(pop_size=100, sampling=init_pop)

    if not os.path.exists('output/figures'):
        os.makedirs('output/figures')
    # Optimize the unlearning problem
    warnings.filterwarnings("ignore")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   output=moo_display(evaluator=gfo.evaluate_params, unblocker=block.unblocker, test_loader=test_loader),
                   callback=moo_callback(gb_f1=[gb_f1, block_gb_f1], gb_f2=[gb_f2, block_gb_f2], plot_path='output/figures/plot'),
                   verbose=True)

    if not os.path.exists('output/history'):
        os.makedirs('output/history')
    np.savez('output/history/500forget_1000retain_100gens_100np/last_pf_sols.npz', X=res.X, F=res.F)
    np.savez('output/history/500forget_1000retain_100gens_100np/last_pop_sols.npz', X=res.pop.get("X"), F=res.pop.get("F"))


    # Get the optimal solution
    optimal_solution = res.X[np.argmin(res.F[:, 0])]

    # Perform unlearning based on the optimal solution
    # Modify the ResNet-18 model weights based on the optimal_solution
    # Re-train the model on the modified dataset


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup variables")
    parser.add_argument('--block_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    main(args)
