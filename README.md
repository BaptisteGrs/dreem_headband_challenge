# dreem_headband_challenge

Automatic detection of sleep stages using Dreem handband signals and a machine learning approach. 

## Work in progress! 

### Features to add : 
- [X] Distance between the EEGs
- [X] Spectrograms to include frequencies information
- [ ] Decomposition into wavelets
- [ ] Coefficients of fitted auto-regressive models

### Data manipulation
- [X] Remove outliers with huge extremas values
- [ ] Treat the imbalanced dataset problem

### Model selection

- [X] Random Forest
- [ ] XGBoost
- [ ] ExtraTrees

## Build pipeline

- [ ] Different models built into a pipeline : one for awake/drowsy VS sleep and then classification of sleep 

## Current features 

### Basic features : 
- Absolute mean/median/min/max of the signal
- Standard Deviation

### Chaos features
- Petrosian fractal dimension 
- Approximate entropy
- Higuchi fractal dimension

All calculated with entroPy 

### Features from Power Spectral Analysis
- Max frequency (frequency with the maximum power in the spectral decomposition)
- Max power in the spectral decomposition
- Sum of the powers of the spectrum

If the signal is from an EEG, we currently add : 

- the sum of the power from the frequencies corresponding to alpha, beta, theta and delta waves. 
