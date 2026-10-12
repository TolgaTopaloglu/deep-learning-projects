# 01 – Iris Classification

Classifies iris flowers into three species (setosa, versicolor, virginica) from four measurements: sepal length/width and petal length/width.

## Steps
1. Data exploration with pandas and seaborn
2. Label encoding, stratified 80/20 train/test split
3. MLP in PyTorch: 4 → 10 → 10 → 3 with ReLU
4. Training with `CrossEntropyLoss` and Adam (lr=0.01, 100 epochs)
5. Evaluation: accuracy, loss curves, confusion matrix (torchmetrics)

## Result
Test accuracy is around 97–100%. The test set has only 30 samples, so one sample changes accuracy by about 3.3%.

## Data
`iris.csv` – the classic Iris dataset (150 samples, 50 per class).
