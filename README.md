<!-- GETTING STARTED -->
## Getting Started
A novel approach to tackle machine unlearning problems in fine-trained DNN models by creating multiple conflicting objectives to optimize. 
In order to mitigate the huge-scale problem in the EA algorithms, the histogram-based blocking approach is used to reduce the search space from **11M** to **31** with a 5.34 compression rate in memory capacity.
In this project, the CIFAR10 dataset is studied on the ResNet-18 model.

### Installation
Let's create virtual environment (venv) in the project and install packages using ```pip```.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage
Simply run the main.py to unlearn 500 data in CIFAR10 data.
```
python main.py --model_path ".\input\models\resnet_cifar10_epochs10_state_dict" --block_path ".\input\blocks\solution_197bins_remerged.pickle"
```

<!-- ROADMAP -->
## Roadmap

- [ ] Fine-trained DNN
- [ ] Multiple Conflicted Objectives
- [ ] Unlearning using MOO algorithms
- [ ] Pareto Frontier
    - [ ] Selection
     
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Rasa Khosrowshahli - rkhosrowshahli@brocku.ca
