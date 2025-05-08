
# Eval Report

## Scores

- **ResP_k (Polarity Score):** 0.6248
- **ResMT_k (Town Score):** 0.6054
- **ResT_k (Type Score):** 0.9601
- **Sentiment(k) (Overall Score):** 0.6710

## Classification Reports

    ### Type Classification Report
                      Hotel    Attractive    Restaurant  accuracy     macro avg  weighted avg
precision      0.965043      0.965394      0.957379  0.961869      0.962605      0.961929
recall         0.969557      0.929707      0.974587  0.961869      0.957951      0.961869
f1-score       0.967295      0.947214      0.965906  0.961869      0.960138      0.961783
support    14322.000000  10442.000000  17747.000000  0.961869  42511.000000  42511.000000

    ### Town Classification Report
               Valle_de_Bravo     Xilitla  San_Cristobal_de_las_Casas       Tulum  Isla_Mujeres  ...  Real_de_Catorce    Coatepec  accuracy     macro avg  weighted avg
precision        0.670750    0.701681                    0.627133    0.708995      0.854962  ...         0.765217    0.673469   0.69415      0.674296      0.693886
recall           0.574803    0.506061                    0.675551    0.519380      0.595745  ...         0.577049    0.489614   0.69415      0.559414      0.694150
f1-score         0.619081    0.588028                    0.650442    0.599553      0.702194  ...         0.657944    0.567010   0.69415      0.605393      0.689893
support        762.000000  330.000000                 2176.000000  258.000000    188.000000  ...       305.000000  337.000000   0.69415  42511.000000  42511.000000

[4 rows x 43 columns]

    ### Polarity Classification Report
                         5            4            1            2             3  accuracy     macro avg  weighted avg
precision     0.698413     0.514754     0.591377     0.576253      0.848209  0.766931      0.645801      0.753451
recall        0.652335     0.446023     0.551952     0.474886      0.914163  0.766931      0.607872      0.766931
f1-score      0.674588     0.477930     0.570985     0.520682      0.879952  0.766931      0.624827      0.758183
support    1349.000000  1408.000000  3330.000000  9198.000000  27226.000000  0.766931  42511.000000  42511.000000

    