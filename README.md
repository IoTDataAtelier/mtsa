# MTSA (Multiple Time Series Analysis)

MTSA is a unified framework for anomaly detection in multiple time series, designed to facilitate the replication and development of state-of-the-art anomaly detection approaches. The framework is particularly focused on handling acoustic data from industrial machinery, such as valves, pumps, fans, and slide rails.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Implemented Approaches](#implemented-approaches)

## Introduction

Anomaly detection is a crucial technique for identifying unusual patterns in data, which can indicate malfunctions or other significant events. In industrial settings, detecting anomalies in machinery data can enable predictive maintenance and reduce downtime.

MTSA (Multiple Time Series Analysis) is a framework designed to simplify the implementation and replication of anomaly detection approaches, particularly for acoustic data. The framework provides a structured environment for developing, testing, and comparing different anomaly detection models, with a focus on replicability and ease of use.

## Features

- **Unified Framework:** MTSA provides a modular and extensible structure for developing and testing anomaly detection models on multiple time series.
- **State-of-the-Art Implementations:** The framework includes implementations of three state-of-the-art anomaly detection approaches: Hitachi, RANSynCoders, and GANF.
- **Enhanced Pipelines:** RANSynCoders and GANF pipelines have been enhanced with feature extraction based on Mel-Frequency Cepstral Coefficients (MFCC), improving their performance on acoustic data.
- **Comparative Analysis:** MTSA enables easy comparison of different approaches, with a focus on acoustic data from industrial machinery.

## Installation

To install MTSA, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/MTSA.git
cd MTSA
pip install -r requirements.txt
```

## Usage

MTSA can be used to run anomaly detection models on acoustic data from industrial machines. Find an example [here](examples/MTSA.ipynb).

if you have problems with google colab please try running it:

```
    %pip install --upgrade google-colab
```

## Implemented Approaches

MTSA includes the following anomaly detection approaches:

- **Hitachi**: A robust approach tailored for industrial anomaly detection.
- **RANSynCoders**: A state-of-the-art model enhanced with MFCC for better performance on acoustic data.
- **GANF**: A generative adversarial network approach, also enhanced with MFCC.
- **Isolation Forest**: An approach using ITrees to isolate anomalies, also enhanced with MFCC.