<p align="center">
<img width="805" height="310" alt="image" src="https://github.com/user-attachments/assets/6f0ff77f-c83e-45eb-bdf6-b6f2fcbd31ed" />
</p>

MTSA is a research toolkit designed to aggregate machine learning models for anomaly detection, with a strong focus on enhancing reproducibility and explainability in model implementation. It offers a structured environment for developing, testing, and comparing various anomaly detection approaches, prioritizing replicability and ease of use. The toolkit is continuously updated to include both classical and state-of-the-art algorithms for anomaly detection in multivariate time series.

## 🔧 Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/MTSA.git
cd MTSA
pip install -r requirements.txt
```

## 🚀 Usage

MTSA allows you to run anomaly detection models on acoustic data collected from complex systems like industrial machines.

A complete example is available in the following Jupyter notebook:  
👉 [examples/MTSA.ipynb](examples/MTSA.ipynb)

> **Note:** If you encounter issues while running on Google Colab, try upgrading the Colab package:

```bash
pip install --upgrade google-colab
```

## 🧠 Implemented Machine Learning Approaches

MTSA currently integrates the following anomaly detection models:

- **Hitachi**  
  A robust autoencoder model specifically designed for industrial anomaly detection tasks.

- **RANSynCoders**  
  Ensemble of autoencoders with FFT, leveraging bootstrapping to perform robust anomaly inference.
  
- **GANF**  
  A model that combines graph structures, recurrent neural networks (RNNs), and normalizing flows to perform anomaly inference.

- **Isolation Forest**  
  A tree-based ensemble method that isolates anomalies.

- **OSVM (One-Class SVM)**  
  A support vector-based approach for detecting outliers by modeling the distribution of normal data.

And more!


## 🌐 Learn More

For full documentation, examples, and additional resources, visit our [official website](https://iotdataatelier.github.io/mtsa-docs/).

