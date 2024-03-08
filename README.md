# Siamese Network for ASL Alphabet Classification
Implementation of a simple Siamese Network using Apple's MLX Framework. Trained and Tested models for K-Shot Image Classification on the ASL Alphabet Dataset.
## Getting Started  
- References
  - [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
  - [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
  - [Example Siamese Network](https://github.com/jingpingjiao/siamese_cnn/tree/master)
  - [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
  - [MLX Cifar Example](https://github.com/ml-explore/mlx-examples/tree/main/cifar)
- Environment  
  - Tested on M1 Mac
  - Python: 3.10
  - conda create --name <env> --file requirements.txt
- Dataset
  - Access dataset [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)
  - 87k images with 29 classes (3k/class)
  - Train: 60k, Test: 18k, Val: 9k
  - Test and Validation classes not included in training dataset to observe how well the model can generalize to unseen classes.
  - Run scripts in demos/eda.ipynb to get data setup for training.
- Training
  - In train.py, update DATA_PATH and BASE_OUTPUT_PATH
  - See training args: python train.py --help
  - See utils/model.py to define a custom model
  - See utils/transforms.py to define custom data augmentations.
  - Run: python train.py
    - Note: If you don't have a Apple Silicon ARM chip, you must use the "--cpu" arg.
  - Run python train.py --help for info on optional args.
- Testing
  - 29-ways K-Shot testing done with a batch size of 1.
  - See demos/Test_models.ipynb