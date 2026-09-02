# Third-Party Dataset Licenses

The root MIT license applies to original software and repository-authored
documentation. It does **not** relicense third-party dataset files or their
processed derivatives under `<DATASET>/data/`. Those materials remain subject
to the terms specified by their original providers.

| Dataset | Creator or publisher | Original source | Source-declared license |
| --- | --- | --- | --- |
| Gallstone Status | Irfan Esen, Hilal Arslan, Selin Aktürk, Mervenur Gülşen, Nimet Kültekin, and Oğuzhan Özdemir | [UCI record and citation](https://archive.ics.uci.edu/dataset/1150/gallstone-1) ([associated DOI](https://doi.org/10.1097/md.0000000000037258)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Breast Cancer Wisconsin (Diagnostic) | William Wolberg, Olvi Mangasarian, Nick Street, and W. Street | [UCI record and citation](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) ([dataset DOI](https://doi.org/10.24432/C5DW2B)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Cancer Risk Prediction | Rabie El Kharoua (`rabieelkharoua`) | [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Kidney Stone Risk | Kaggle publisher `omarayman15` | [Kaggle](https://www.kaggle.com/datasets/omarayman15/kidney-stones) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| Lung Cancer Risk Level | Kaggle publisher `himabindumarpini` | [Kaggle](https://www.kaggle.com/datasets/himabindumarpini/lung-cancer-datasets) | [MIT](https://opensource.org/license/mit) |
| BRFSS 2015 Diabetes Health Indicators | Alex Teboul (`alexteboul`) | [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

The repository redistributes provider-sourced dataset files and stores derived
`processed_data.pkl` files. The processed derivatives were created by removing
identifiers where applicable, encoding targets and categorical variables,
performing a stratified 80/20 split, and fitting a `MinMaxScaler` on training
data before transforming both splits. These modifications are identified here
to satisfy attribution requirements and must not be interpreted as provider
endorsement.

The source links are authoritative for current access conditions, attribution
requirements, and any dataset-specific restrictions. Users are responsible for
complying with those terms. No additional rights in third-party data are
granted by this repository.
