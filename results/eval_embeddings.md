
# Eval Report
## Scores
- **ResP_k (Polarity Score):** 0.5623
- **ResMT_k (Town Score):** 0.5826
- **ResT_k (Type Score):** 0.9595
- **Sentiment(k) (Overall Score):** 0.6386
## Classification Reports
### Type Classification Report
                      Hotel    Attractive    Restaurant  accuracy     macro avg  weighted avg
precision      0.969525      0.959689      0.955366  0.961163      0.961527      0.961198
recall         0.968510      0.932484      0.972108  0.961163      0.957701      0.961163
f1-score       0.969017      0.945891      0.963664  0.961163      0.959524      0.961102
support    14322.000000  10442.000000  17747.000000  0.961163  42511.000000  42511.000000
    ### Town Classification Report
               Valle_de_Bravo     Xilitla  San_Cristobal_de_las_Casas       Tulum  Isla_Mujeres  ...  Real_de_Catorce    Coatepec  accuracy     macro avg  weighted avg
precision        0.749482    0.910448                    0.549517    0.969388      0.979798  ...         0.963855    0.859551  0.660253      0.788168      0.694227
recall           0.475066    0.369697                    0.627298    0.368217      0.515957  ...         0.524590    0.454006  0.660253      0.493519      0.660253
f1-score         0.581526    0.525862                    0.585837    0.533708      0.675958  ...         0.679406    0.594175  0.660253      0.582555      0.653213
support        762.000000  330.000000                 2176.000000  258.000000    188.000000  ...       305.000000  337.000000  0.660253  42511.000000  42511.000000

[4 rows x 43 columns]
    ### Polarity Classification Report
                         5            4            1            2             3  accuracy     macro avg  weighted avg
precision     0.635569     0.439216     0.525671     0.547154      0.815680  0.742396      0.592658      0.716678
recall        0.646405     0.318182     0.476577     0.356382      0.932014  0.742396      0.545912      0.742396
f1-score      0.640941     0.369028     0.499921     0.431628      0.869975  0.742396      0.562299      0.722284
support    1349.000000  1408.000000  3330.000000  9198.000000  27226.000000  0.742396  42511.000000  42511.000000
    