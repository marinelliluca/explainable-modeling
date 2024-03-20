# Explainable Modeling of Gender-Targeting Practices in Toy Advertising Sound and Music

Accepted at the [ICASSP 2024 XAI-SA Workshop](https://xai-sa-workshop.github.io/web/Accepted%20papers.html)

[Link to the paper](https://www.researchgate.net/publication/379085262_Explainable_Modeling_of_Gender-Targeting_Practices_in_Toy_Advertising_Sound_and_Music)

### Luca Marinelli, Charalampos Saitis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Abstract:
This study examines gender coding in sound and music, in a context where music plays a supportive role to other modalities, such as in toy advertising. We trained a series of binary XGBoost classifiers on handcrafted features extracted from the soundtracks and then performed SAGE and SHAP analyses to identify key audio features in predicting the gender target of the ads. Our analysis reveals that timbral dimensions play a prominent role and that commercials aimed at girls tend to be more harmonious and rhythmical, with a broader and smoother spectrum, while those targeting boys are characterised by higher loudness, spectral entropy, and roughness. Mixed audience commercials instead appear to be as rhythmical as girls-only ads, although slower, but show intermediate characteristics in terms of harmonicity and roughness. This study highlights the importance of music in shaping societal norms and the need for greater accountability in its use in marketing and other industries. We provide a public repository containing all code and data used in this study.

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
  - Commercial classification: Based on gender of actors/presenters and content analysis (Krippendorff's alpha level of .91)

Videos available on YouTube
```python
import pandas as pd
fn = f"data/features_mean_with_target.csv"
df = pd.read_csv(fn, index_col="stimulus_id")

youtube_id = df.sample(1).index[0]

# some videos might have been deleted by the owner
print(f"https://www.youtube.com/watch?v={youtube_id}") 
```

- **Feature Extraction:**
  - MIRtoolbox (version 1.8.1)

- **Model Training and Evaluation:**
  - Employed XGBoost classifiers
  - Tasks: Girls vs boys-only, girls-only vs mixed audience, boys-only vs mixed audience
  - Optuna for hyperparameters optimization
  - Recursive feature elimination with cross-validation (RFECV)
    - for the intersection of the selecteted feature across tasks see [here](https://github.com/marinelliluca/explainable-modeling/blob/a36d9265648c7781e937aca1bfcb52df095b3c9e/data/temp.ipynb)

- **Model interpretation:**
  - SHAP for local explanation
  - SAGE for global importance
