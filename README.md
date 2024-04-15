# Predictor Model Repository

This repository contains Python files used for predicting various parameters in memory architectures based on different machine learning models.

## Overview

The predictor model in this repository is designed to predict various parameters such as Read Latency, Buffer Latency, Buffer Read Dynamic Energy, IC Latency, IC Read Dynamic Energy, Leakage Energy, Leakage Power, and Read Dynamic Energy. The prediction is made based on the selected machine learning model (DenseNet40 or VGG8) and the specifications of memory cell type, subarray size, and tech node size.

## Files

- `Predictor.ipynb`: Jupyter Notebook containing the Python code for the predictor model.
- `ReadLatencyDenseNet40.py`: Python script containing the code for predicting Read Latency using DenseNet40.
- `ReadLatencyVGG8.py`: Python script containing the code for predicting Read Latency using VGG8.
- `BufferLatencyDenseNet40.py`: Python script containing the code for predicting Buffer Latency using DenseNet40.
- `BufferLatencyVGG8.py`: Python script containing the code for predicting Buffer Latency using VGG8.
- `BufferReadDynamicEnergyDenseNet40.py`: Python script containing the code for predicting Buffer Read Dynamic Energy using DenseNet40.
- `BufferReadDynamicEnergyVGG8.py`: Python script containing the code for predicting Buffer Read Dynamic Energy using VGG8.
- `ICLatencyDenseNet40.py`: Python script containing the code for predicting IC Latency using DenseNet40.
- `ICLatencyVGG8.py`: Python script containing the code for predicting IC Latency using VGG8.
- `ICReadDynamicEnergyDenseNet40.py`: Python script containing the code for predicting IC Read Dynamic Energy using DenseNet40.
- `ICReadDynamicEnergyVGG8.py`: Python script containing the code for predicting IC Read Dynamic Energy using VGG8.
- `LeakageEnergyDenseNet40.py`: Python script containing the code for predicting Leakage Energy using DenseNet40.
- `LeakageEnergyVGG8.py`: Python script containing the code for predicting Leakage Energy using VGG8.
- `LeakagePowerDenseNet40.py`: Python script containing the code for predicting Leakage Power using DenseNet40.
- `LeakagePowerVGG8.py`: Python script containing the code for predicting Leakage Power using VGG8.
- `ReadDyanmicEnergyDenseNet40.py`: Python script containing the code for predicting Read Dynamic Energy using DenseNet40.
- `ReadDyanmicEnergyVGG8.py`: Python script containing the code for predicting Read Dynamic Energy using VGG8.
- `TechnodePackage.py`: Python script containing the code for predicting parameters related to technode size.
## Model Accuracies

| Parameter                            | Accuracy |
|--------------------------------------|----------|
| Read Latency                         | 83.53%   |
| Read Dynamic Energy                  | 88.51%   |
| Leakage Power                        | 90.39%   |
| Leakage Energy                       | 85.03%   |
| Buffer Latency                       | 96.78%   |
| Buffer Read Dynamic Energy           | 98.62%   |
| IC Latency                          | 88.45%   |
| IC Read Dynamic Energy               | 91.29%   |
| Energy Efficiency TOPS/W            | 98.17%   |
| Throughput TOPS                     | 96.51%   |
| Throughput FPS                      | 97.27%   |
| Compute Efficiency TOPS/mm^2        | 95.89%   |
## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/mrkaesy/NeuroSim-Profiling-Data-Analysis.git
```
2. cd predictor-model
3. Run the Google Colab -> Predictor.ipynb to interactively use the predictor model and make predictions.


