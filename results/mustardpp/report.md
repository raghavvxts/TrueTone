# MUStARD vs MUStARD++: local paired comparison



All conditions use identical frozen encoders, LR tuning grid and threshold. No emotion ground-truth annotations are fed as features.

The primary test population is the same 688 shared IDs. Repeated normalized utterances and identical decoded audio are grouped in outer and inner CV.

There are three repetitions of five grouped folds. Added clips may be training examples only when their target group is absent from the test fold.

Size-matched results separate dataset composition from simply having more training examples. Original baseline has its original labels/transcripts.

Shared test features use MUStARD++ transcripts, including six corrected transcripts. Audio for shared IDs is the private original-audio copy.

Block scalers are fitted on the outer training set and reused during inner C selection; inner preprocessing is not fully nested. The outer test set remains excluded.

These are development-set comparisons; confidence intervals do not capture every source of dataset/domain uncertainty. Exact-match grouping cannot eliminate related dialogue, episode or speaker effects. Held-out-show evaluation is reported separately; new speakers and live microphones need a separate labeled benchmark.



## Pooled shared-test results (average each clip’s three out-of-fold probabilities)



| training               | mode             |   n_test |       f1 |   macro_f1 |   accuracy |   precision |   recall |   mean_fold_f1 |   std_fold_f1 |
|:-----------------------|:-----------------|---------:|---------:|-----------:|-----------:|------------:|---------:|---------------:|--------------:|
| MUStARD                | Speech reference |      688 | 0.723647 |   0.717906 |   0.718023 |    0.709497 | 0.738372 |       0.722657 |     0.0365946 |
| MUStARD                | Text + speech    |      688 | 0.743516 |   0.741259 |   0.741279 |    0.737143 | 0.75     |       0.741647 |     0.040639  |
| MUStARD                | Text only        |      688 | 0.66     |   0.653964 |   0.65407  |    0.648876 | 0.671512 |       0.65881  |     0.0326473 |
| MUStARD++              | Speech reference |      688 | 0.723703 |   0.713284 |   0.713663 |    0.699187 | 0.75     |       0.735084 |     0.0459746 |
| MUStARD++              | Text + speech    |      688 | 0.760839 |   0.75107  |   0.751453 |    0.733154 | 0.790698 |       0.758152 |     0.0348947 |
| MUStARD++              | Text only        |      688 | 0.685552 |   0.677105 |   0.677326 |    0.668508 | 0.703488 |       0.692413 |     0.0373456 |
| MUStARD++ size-matched | Speech reference |      688 | 0.732044 |   0.717249 |   0.718023 |    0.697368 | 0.770349 |       0.718279 |     0.044218  |
| MUStARD++ size-matched | Text + speech    |      688 | 0.743484 |   0.727229 |   0.728198 |    0.703896 | 0.787791 |       0.732984 |     0.0379965 |
| MUStARD++ size-matched | Text only        |      688 | 0.706371 |   0.691106 |   0.69186  |    0.674603 | 0.741279 |       0.672922 |     0.0365129 |



## Paired F1 differences (2,000 group bootstrap samples)



| mode             | comparison                           |     delta_f1 |      ci_low |   ci_high |
|:-----------------|:-------------------------------------|-------------:|------------:|----------:|
| Text only        | MUStARD++ minus MUStARD              |  0.0255524   | -0.00386278 | 0.0575538 |
| Text only        | MUStARD++ size-matched minus MUStARD |  0.0463712   |  0.0194993  | 0.0746217 |
| Speech reference | MUStARD++ minus MUStARD              |  5.59411e-05 | -0.0263136  | 0.0262171 |
| Speech reference | MUStARD++ size-matched minus MUStARD |  0.00839748  | -0.0164894  | 0.0341011 |
| Text + speech    | MUStARD++ minus MUStARD              |  0.0173233   | -0.00731456 | 0.0419835 |
| Text + speech    | MUStARD++ size-matched minus MUStARD | -3.16252e-05 | -0.0274331  | 0.025327  |



## Held-out-show evaluation



| show           | training   | mode             |   n_train |   n_test |   n_positive |       f1 |   macro_f1 |   accuracy |   precision |    recall |
|:---------------|:-----------|:-----------------|----------:|---------:|-------------:|---------:|-----------:|-----------:|------------:|----------:|
| BBT            | MUStARD    | Text only        |       405 |      716 |          360 | 0.611675 |  0.56826   |  0.572626  |    0.563084 | 0.669444  |
| BBT            | MUStARD    | Speech reference |       405 |      716 |          360 | 0.683572 |  0.568453  |  0.599162  |    0.566728 | 0.861111  |
| BBT            | MUStARD    | Text + speech    |       405 |      716 |          360 | 0.703495 |  0.610463  |  0.632682  |    0.59203  | 0.866667  |
| BBT            | MUStARD++  | Text only        |       481 |      716 |          360 | 0.626109 |  0.583661  |  0.587989  |    0.575758 | 0.686111  |
| BBT            | MUStARD++  | Speech reference |       481 |      716 |          360 | 0.696471 |  0.626586  |  0.639665  |    0.604082 | 0.822222  |
| BBT            | MUStARD++  | Text + speech    |       481 |      716 |          360 | 0.699401 |  0.639483  |  0.649441  |    0.614737 | 0.811111  |
| FRIENDS        | MUStARD    | Text only        |       333 |      354 |          151 | 0.516667 |  0.508333  |  0.508475  |    0.444976 | 0.615894  |
| FRIENDS        | MUStARD    | Speech reference |       333 |      354 |          151 | 0.649842 |  0.682977  |  0.686441  |    0.620482 | 0.682119  |
| FRIENDS        | MUStARD    | Text + speech    |       333 |      354 |          151 | 0.612903 |  0.655698  |  0.661017  |    0.597484 | 0.629139  |
| FRIENDS        | MUStARD++  | Text only        |       844 |      354 |          151 | 0.620253 |  0.657065  |  0.661017  |    0.593939 | 0.649007  |
| FRIENDS        | MUStARD++  | Speech reference |       844 |      354 |          151 | 0.727848 |  0.75423   |  0.757062  |    0.69697  | 0.761589  |
| FRIENDS        | MUStARD++  | Text + speech    |       844 |      354 |          151 | 0.721854 |  0.757479  |  0.762712  |    0.721854 | 0.721854  |
| GOLDENGIRLS    | MUStARD    | Text only        |       650 |       40 |           39 | 0.888889 |  0.444444  |  0.8       |    0.969697 | 0.820513  |
| GOLDENGIRLS    | MUStARD    | Speech reference |       650 |       40 |           39 | 0.961039 |  0.480519  |  0.925     |    0.973684 | 0.948718  |
| GOLDENGIRLS    | MUStARD    | Text + speech    |       650 |       40 |           39 | 0.947368 |  0.473684  |  0.9       |    0.972973 | 0.923077  |
| GOLDENGIRLS    | MUStARD++  | Text only        |      1162 |       40 |           39 | 0.888889 |  0.444444  |  0.8       |    0.969697 | 0.820513  |
| GOLDENGIRLS    | MUStARD++  | Speech reference |      1162 |       40 |           39 | 0.974359 |  0.487179  |  0.95      |    0.974359 | 0.974359  |
| GOLDENGIRLS    | MUStARD++  | Text + speech    |      1162 |       40 |           39 | 0.961039 |  0.480519  |  0.925     |    0.973684 | 0.948718  |
| SARCASMOHOLICS | MUStARD    | Text only        |       674 |       14 |           14 | 0.526316 |  0.263158  |  0.357143  |    1        | 0.357143  |
| SARCASMOHOLICS | MUStARD    | Speech reference |       674 |       14 |           14 | 0        |  0         |  0         |    0        | 0         |
| SARCASMOHOLICS | MUStARD    | Text + speech    |       674 |       14 |           14 | 0.352941 |  0.176471  |  0.214286  |    1        | 0.214286  |
| SARCASMOHOLICS | MUStARD++  | Text only        |      1186 |       14 |           14 | 0.6      |  0.3       |  0.428571  |    1        | 0.428571  |
| SARCASMOHOLICS | MUStARD++  | Speech reference |      1186 |       14 |           14 | 0.133333 |  0.0666667 |  0.0714286 |    1        | 0.0714286 |
| SARCASMOHOLICS | MUStARD++  | Text + speech    |      1186 |       14 |           14 | 0.352941 |  0.176471  |  0.214286  |    1        | 0.214286  |
| SV             | MUStARD    | Text only        |       686 |       78 |           37 | 0.575342 |  0.600924  |  0.602564  |    0.583333 | 0.567568  |
| SV             | MUStARD    | Speech reference |       686 |       78 |           37 | 0.212766 |  0.436658  |  0.525641  |    0.5      | 0.135135  |
| SV             | MUStARD    | Text + speech    |       686 |       78 |           37 | 0.352941 |  0.519328  |  0.576923  |    0.642857 | 0.243243  |
| SV             | MUStARD++  | Text only        |      1120 |       78 |           37 | 0.60241  |  0.575177  |  0.576923  |    0.543478 | 0.675676  |
| SV             | MUStARD++  | Speech reference |      1120 |       78 |           37 | 0.307692 |  0.480769  |  0.538462  |    0.533333 | 0.216216  |
| SV             | MUStARD++  | Text + speech    |      1120 |       78 |           37 | 0.53125  |  0.602582  |  0.615385  |    0.62963  | 0.459459  |



The app uses the MUStARD++ final refit; MUStARD weights are archived as the baseline. No personal recordings were used in training or evaluation.

A dataset cannot be declared universally better from these scores. Inspect shared-test, size-matched and held-out-show results together.

Total comparison and final-refit runtime: 0.52 minutes.
