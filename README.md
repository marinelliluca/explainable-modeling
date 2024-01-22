# Explainable Modeling of Gender-Targeting Practices in Toy Advertising Sound and Music
### Luca Marinelli, Charalampos Saitis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Abstract:
This study examines gender coding in sound and music, in a context where music plays a supportive role to other modalities, such as in toy advertising. We trained a series of binary XGBoost classifiers on handcrafted features extracted from the soundtracks and then performed SAGE and SHAP analyses to identify key audio features in predicting the gender target of the ads. Our analysis reveals that timbral dimensions play a prominent role, and that commercials aimed at girls tend to be more harmonious and rhythmical, with a broader and smoother spectrum, while those targeting boys are characterised by higher loudness, spectral entropy, and roughness. Mixed audience commercials instead appear to be as rhythmical as girls-only ads, although slower, but show intermediate characteristics in terms of harmonicity and roughness.

## Method

- **Dataset Description:**
  - Size: 606 commercials
  - Period covered: 2012-2022
  - Categories: 
    - Feminine audience: 163 commercials
    - Masculine audience: 149 commercials
    - Mixed audience: 200 commercials
    - No actors/presenters: 94 commercials (excluded from analysis)
  - Source: Official YouTube channel of Smyths Toys Superstores
  - Selection criteria: High-quality videos intended for television
  - Preprocessing: Removal of duplicate videos and soundtracks trimmed (excluding last 5 seconds)
  - Commercial classification: Based on gender of actors/presenters and content analysis
  - Reliability: Krippendorff's alpha level of .91 

- **Feature Extraction:**
  - MIRtoolbox (version 1.8.1)

- **Model Training and Evaluation:**
  - Employed XGBoost classifiers
  - Settings: Girls vs boys-only, girls-only vs mixed audience, boys-only vs mixed audience
  - Validation: Leave-one-out cross-validation
  - Tuning: Optuna for hyperparameters optimization
  - Feature selection: Recursive feature elimination with cross-validation (RFECV)
    - for an analysis of the selected features run [this](https://github.com/marinelliluca/explainable-modeling/blob/b111fd6b7882a7701e58ca0c69cfc1753ee248df/data/temp.ipynb) notebook

- **Model interpretation:**
  - SHAP for local explanation
  - SAGE for global importance
