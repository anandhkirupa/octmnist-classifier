# OCTMNIST Classifier

A Python package for classifying retinal OCT images into four categories: **CNV**, **DME**, **Drusen**, and **Normal**, based on the [OCTMNIST](https://medmnist.com/) dataset. Built with a custom CNN and includes preprocessing, class balancing strategies, and a CLI for inference.

## 🔧 Installation

```bash
pip install octmnist-classifier
```

> Make sure to have 'torch', 'medmnist', and other dependencies listed in 'setup.py' installed.

## 📦 Features

- Preprocessing and loading of the OCTMNIST dataset
- Class balancing using:

  - SMOTE
  - SMOTE + Tomek Links
  - Undersampling

- CNN model with ReLU, BatchNorm, MaxPooling
- Trainable via script interface ('scripts/train.py')
- CLI interface to run inference on new images

## 🧠 Pretrained Models

Download pretrained models from the **[Releases](https://github.com/anandhkirupa/octmnist-classifier/releases)** page:

- 'model_smote.pt'
- 'model_smote_tomek.pt'

Place them in the 'saved_model/' directory before running predictions.

## 🚀 Usage

### 1. Predict a single image

```bash
octmnist-predict --image retina.png --model saved_model/model_smote.pt
```

Output:

```
Predicted class: NORMAL (Class ID: 3)
```

### 2. Training with balancing

```bash
python scripts/train.py --strategy smote --epochs 15
```

Options for '--strategy': 'smote', 'smote_tomek', 'undersample'

## 🧪 Evaluation (planned)

In future versions, you'll be able to run:

```bash
octmnist-eval --model saved_model/model_smote.pt --test_dir test_images/
```

## 🔍 Model Architecture

- 4 Conv layers with ReLU, BatchNorm, MaxPooling
- Final Linear layer with Dropout
- Xavier/He initialization supported

## 🗂️ Project Structure

```
octmnist-classifier/
├── octmnist_classifier/
│   ├── model.py
│   ├── preprocess.py
│   ├── predict.py
│   └── cli.py
├── saved_model/
│   └── model_smote.pt
├── scripts/
│   └── train.py
└── setup.py
```

## 📄 License

MIT License

---

Maintained by [Kirupanandan Jagadeesan](https://www.linkedin.com/in/kirupanandanjagadeesan/). Contributions and feedback welcome!


@article{yang2021medmnist,
  title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
  author={Yang, Jiancheng and Shi, Yinan and Lin, Yucheng and et al.},
  journal={arXiv preprint arXiv:2012.14591},
  year={2021}
}
