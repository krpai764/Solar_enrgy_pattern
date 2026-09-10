Spatio-Temporal Hybrid Model for Solar Irradiance Forecasting ☀️

This repository contains a PyTorch-based, dual-stream deep learning pipeline designed to forecast Global Horizontal Irradiance (GHI) across a regional network of solar stations.

By fusing the physical mapping capabilities of a Graph Convolutional Recurrent Network (GConvGRU) with the statistical pattern-recognition of a 2D Convolutional Neural Network (CNN), this hybrid architecture captures both the spatial movement of weather fronts and the hidden multivariate relationships between atmospheric variables.

🚀 Key Features

Dual-Stream Architecture:

GConvGRU Stream: Processes 10-hour sliding windows of GHI data alongside a geographic edge_index map to learn the physical, spatio-temporal trajectory of cloud cover and solar intensity.

2D-CNN Stream: Scans custom-engineered statistical "images" (correlation matrices of non-target weather variables like temperature, humidity, and pressure) to extract hidden environmental indicators.

Late-Stage Feature Fusion: Both streams operate independently before their output vectors are flattened and concatenated (torch.cat), allowing the final fully connected layers to balance spatial physics with atmospheric statistics.

High Precision Performance: The model was trained and tested on 5 years of National Solar Radiation Database (NSRDB) data across 5 distinct Southern California stations (Los Angeles & San Diego regions), achieving a Root Mean Squared Error (RMSE) of 9.24 W/m².

🛠️ Technology Stack

Core Deep Learning: torch (PyTorch), torch.nn

Graph Neural Networks: torch_geometric, torch_geometric_temporal (specifically StaticGraphTemporalSignal and GConvGRU)

Data Processing: pandas, numpy, sklearn.preprocessing.MinMaxScaler

Evaluation: sklearn.metrics (MSE, MAE, MAPE, R2)

📊 Dataset Preparation

The pipeline ingests raw CSV data from the NSRDB. To ensure the model receives clean, synchronized data, the preprocessing module performs the following steps:

Variable Isolation: Extraneous variables (e.g., 'Minute', 'Clearsky GHI') are dropped to prevent target leakage.

Nighttime Filtering: Rows where GHI == 0 are removed to prevent the model from artificially reducing its error rate by continuously predicting zero during night hours.

Timestamp Synchronization: A set.intersection operation is performed across the Datetime indices of all 5 station dataframes. Any timestamp where a sensor went offline at any station is dropped from all stations, ensuring the spatial graph remains perfectly temporally aligned.

Min-Max Scaling: All weather variables are normalized between 0 and 1 to ensure numerical stability during gradient descent.

🧠 Model Architecture

1. Graph Convolutional Network (GCN)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GConvGRU(input_dim, hidden_dim, K=2)
        self.conv2 = GConvGRU(hidden_dim, hidden_dim, K=2)
        self.fc = nn.Linear(hidden_dim, output_dim)


Utilizes Chebyshev Spectral Graph Convolutions (K=2) to allow each station to learn from its immediate neighbors and its neighbors' neighbors.

2. Convolutional Neural Network (ConvNet)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128, 10)


Applies 3x3 kernels and max pooling to 2D correlation matrices, extracting high-level atmospheric patterns into a 10-feature vector.

3. The Combined Model

class CombinedModel(nn.Module):
    def __init__(self, g_in, g_hid, g_out):
        super(CombinedModel, self).__init__()
        # ... initializes GCN, ConvNet, and FC layers ...
    def forward(self, x_gcn, edge_index, x_cnn):
        out_gcn = torch.flatten(self.gcn_model(x_gcn, edge_index))
        out_cnn = torch.flatten(self.cnn_model(x_cnn))
        z = F.relu(self.fc(torch.cat((out_gcn, out_cnn), axis=0)))
        return self.fc2(z)


📈 Evaluation & Results

The model is evaluated on a strictly isolated test set to measure its generalization capabilities. Raw predictions are inverse-transformed back to Watts per square meter ($W/m^2$) before metric calculation.

R² Score: 0.999 (Demonstrates near-perfect alignment with structural variance)

MAPE: 13.8% (Mean Absolute Percentage Error)

RMSE: 9.24 $W/m^2$ (Heavily penalizes massive outlier mistakes)

MAE: ~6.3 $W/m^2$ (The baseline average error rate)

🔮 Future Scalability

To scale this pipeline from a 5-node regional graph to a 500-node statewide network, two critical updates should be implemented:

Data Preprocessing: Replace the set.intersection dropout method with a KNNImputer to prevent total dataset loss when individual hardware sensors fail.

Architecture: Replace the "Complete Graph" (edge_index) with a localized KNN Graph (K-Nearest Neighbors) to drastically reduce the number of spatial edges, preventing GPU memory exhaustion during the GConvGRU processing step.
