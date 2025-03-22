# AcciNet

A **binary classifier** using **Convolutional Neural Networks (CNN)** to distinguish between accident and non-accident images.

## 📊 Dataset

The dataset is organized into three subsets:

```
AcciNet/
├── dataset/
│    ├── train/   # Training data
│    ├── val/     # Validation data
│    └── test/    # Testing data
└── main.ipynb    # Model training and evaluation
```

### Dataset Details:
- **train/**: This directory contains images used for training the model. The model learns to identify patterns specific to accident and non-accident scenarios.
- **val/**: Images in this directory are used to validate the model's performance during training, helping to fine-tune hyperparameters and prevent overfitting.
- **test/**: This directory holds images for the final evaluation of the model's accuracy and robustness.

## 🧠 Model

The binary classifier is built using a **Convolutional Neural Network (CNN)** architecture to detect and classify accident and non-accident images.

### Model Architecture:
The model follows a standard CNN design:

1. **Input Layer**: Image data is preprocessed and fed into the model.
2. **Convolutional Layers**: Extracts spatial features using multiple filters.
3. **Batch Normalization & Activation**: Improves convergence and stabilizes learning.
4. **Pooling Layers**: Reduces dimensionality while retaining key features.
5. **Fully Connected Layers**: Classifies the image based on extracted features.
6. **Output Layer**: Binary output (Accident/Non-Accident) using a sigmoid activation function.

### Model Features:
- Data Augmentation (flipping, rotation, and scaling) to improve generalization.
- Binary output indicating accident or non-accident status.
- Performance metrics: Accuracy, precision, recall, and F1-score.

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/hanish9193/AcciNet.git
cd AcciNet
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Ensure the dataset is in the correct directory structure as shown above.

2. Run the Jupyter Notebook to train and evaluate the model:

```bash
jupyter notebook main.ipynb
```

### Training the Model:
- Ensure the dataset is properly placed in the `dataset/` directory.
- Open `main.ipynb` in Jupyter Notebook.
- Execute each cell sequentially to preprocess the data, train the model, and evaluate its performance.

### Evaluating the Model:
- The notebook will output accuracy, loss, and visualizations such as:
  - Accuracy and loss curves.
  - Confusion matrix for model performance.
  - Example predictions with images.

## 📊 Model Performance

After training the model, you will receive several evaluation metrics:

- **Accuracy**: Measures the percentage of correctly classified images.
- **Loss**: Evaluates the error during training.
- **Confusion Matrix**: Visual representation of true vs. predicted classifications.
- **Precision & Recall**: Useful for imbalanced datasets.

## 📈 Example Results

After training, the model typically achieves:

- Accuracy: ~95% on the validation set.
- Low false positives and false negatives.

The model can be fine-tuned further by adjusting hyperparameters such as learning rate, batch size, and number of epochs.

## 📌 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/new-feature
```

3. Commit your changes:

```bash
git commit -m "Add new feature"
```

4. Push to your forked repository:

```bash
git push origin feature/new-feature
```

5. Open a pull request on GitHub.

## 🛠️ Troubleshooting

- **Dataset not found**: Ensure the `train`, `val`, and `test` directories are in the correct location.
- **Large file errors**: Use `git lfs` for large datasets.

To install Git LFS:

```bash
git lfs install
```

Add large files using:

```bash
git lfs track "dataset/*"
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📧 Contact

For questions or suggestions, open an issue or reach out to the repository owner.

