# NeuralRisk
A multi-label neural engine built to predict 15 different medical conditions simultaneously from standard patient vitals and demographic data.

Architecture:
 - Framework: PyTorch
 - Layers: 2-layer MLP (64 -> 32) with ReLU activation.
 - Embeddings: nn.Embedding for categorical features to capture non-linear relationships.
 - Regularization: BatchNorm1d and Dropout (0.2) to prevent the model from memorizing the CSV.
 - The Math: Weighted BCEWithLogitsLoss using a square-root scaling factor to force the model to care about rare conditions.
 - Threshold: lowered activiation threshold to 0.3 in order to increase recall (specifically increase activiation for diseases with lower prevelance)

Results:
 - The model hit a F1-score of 0.91 for chronic keidney disease, 0.81 for hypertension, and 0.76 for obesity
 - I could not get loss function below 0.5--reasonable, since many diseases the MLP tried to predict are hereditary

Files:
 - MrDoc.ipynb: The Jupiter Notebook with everything, including the architecture, the training loop, and a standalone inference script.
 - mistaDocta_v1.pth: The best weights from running the training script stored here
 - patients.csv: The dataset. See below for download

Download patients.csv from the following kraggle dataset: https://www.kaggle.com/datasets/sergionefedov/patient-records-100k-patients-15-conditions?resource=download
