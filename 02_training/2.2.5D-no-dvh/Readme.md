# Dose Prediction Evaluation Report

## Evaluation Results for WL (Whole Left) (n=20)
| Metric              | GT (mean ± std)   | Pred (mean ± std)   |   p-value | Method   |
|:--------------------|:------------------|:--------------------|----------:|:---------|
| CTV_Dmax            | 27.42 ± 0.29      | 28.60 ± 0.00        |    0      | Wilcoxon |
| CTV_Dmean           | 26.04 ± 0.29      | 25.98 ± 0.06        |    0.8408 | Wilcoxon |
| CTV_V95%            | 96.94 ± 1.08      | 97.43 ± 0.79        |    0.084  | T-test   |
| Contra_Breast_Dmax  | 6.55 ± 3.40       | 7.36 ± 2.88         |    0.1329 | T-test   |
| Contra_Breast_Dmean | 1.34 ± 0.76       | 1.64 ± 0.48         |    0.0558 | T-test   |
| Contra_Lung_Dmax    | 4.52 ± 1.91       | 6.47 ± 1.77         |    0.003  | T-test   |
| Contra_Lung_Dmean   | 0.71 ± 0.28       | 1.03 ± 0.26         |    0.0002 | T-test   |
| DICE_50%            | 1.00 ± 0.00       | 0.94 ± 0.02         |    0      | Wilcoxon |
| DICE_95%            | 1.00 ± 0.00       | 0.95 ± 0.01         |    0      | T-test   |
| External_Dmax       | 27.42 ± 0.29      | 28.60 ± 0.00        |    0      | Wilcoxon |
| External_Dmean      | 1.61 ± 0.38       | 1.68 ± 0.35         |    0.0624 | T-test   |
| Heart_Dmax          | 3.95 ± 2.54       | 3.41 ± 0.94         |    0.2943 | Wilcoxon |
| Heart_Dmean         | 0.83 ± 0.17       | 0.93 ± 0.09         |    0.0051 | T-test   |
| Heart_V1.5Gy        | 6.43 ± 5.64       | 7.60 ± 4.02         |    0.2039 | T-test   |
| Heart_V7Gy          | 0.01 ± 0.04       | 0.00 ± 0.00         |    0.1088 | Wilcoxon |
| Ipsi_Lung_Dmax      | 23.45 ± 1.39      | 23.03 ± 1.23        |    0.0658 | T-test   |
| Ipsi_Lung_Dmean     | 2.45 ± 0.52       | 2.47 ± 0.34         |    0.8035 | T-test   |
| Ipsi_Lung_V8Gy      | 7.80 ± 2.44       | 7.27 ± 1.81         |    0.2218 | T-test   |
| MAE_CTV             | 0.00 ± 0.00       | 0.39 ± 0.22         |    0      | Wilcoxon |
| MAE_Exter           | 0.00 ± 0.00       | 0.29 ± 0.09         |    0      | Wilcoxon |
| MAE_Total           | 0.00 ± 0.00       | 0.08 ± 0.03         |    0      | Wilcoxon |
| SSIM_CTV            | 1.00 ± 0.00       | 0.92 ± 0.02         |    0      | Wilcoxon |
| SSIM_Exter          | 1.00 ± 0.00       | 0.93 ± 0.02         |    0      | T-test   |
| SSIM_Total          | 1.00 ± 0.00       | 0.98 ± 0.01         |    0      | T-test   |

## Evaluation Results for WR (Whole Right) (n=20)
| Metric              | GT (mean ± std)   | Pred (mean ± std)   |   p-value | Method   |
|:--------------------|:------------------|:--------------------|----------:|:---------|
| CTV_Dmax            | 27.37 ± 0.25      | 28.60 ± 0.00        |    0      | T-test   |
| CTV_Dmean           | 26.06 ± 0.38      | 25.96 ± 0.09        |    0.5706 | Wilcoxon |
| CTV_V95%            | 97.28 ± 1.22      | 97.43 ± 1.23        |    0.6961 | T-test   |
| Contra_Breast_Dmax  | 9.46 ± 4.87       | 9.76 ± 3.91         |    0.7584 | T-test   |
| Contra_Breast_Dmean | 1.43 ± 0.68       | 1.76 ± 0.70         |    0.1557 | T-test   |
| Contra_Lung_Dmax    | 5.00 ± 2.10       | 5.42 ± 1.60         |    0.4614 | T-test   |
| Contra_Lung_Dmean   | 0.75 ± 0.30       | 0.78 ± 0.13         |    0.72   | T-test   |
| DICE_50%            | 1.00 ± 0.00       | 0.93 ± 0.03         |    0      | Wilcoxon |
| DICE_95%            | 1.00 ± 0.00       | 0.95 ± 0.01         |    0      | Wilcoxon |
| External_Dmax       | 27.37 ± 0.25      | 28.60 ± 0.00        |    0      | T-test   |
| External_Dmean      | 1.82 ± 0.39       | 1.78 ± 0.30         |    0.4183 | T-test   |
| Heart_Dmax          | 2.87 ± 1.40       | 2.80 ± 0.80         |    0.4091 | Wilcoxon |
| Heart_Dmean         | 0.83 ± 0.18       | 0.85 ± 0.11         |    0.5658 | T-test   |
| Heart_V1.5Gy        | 6.31 ± 5.00       | 5.38 ± 3.22         |    0.9854 | Wilcoxon |
| Heart_V7Gy          | 0.01 ± 0.02       | 0.00 ± 0.00         |    0.3173 | Wilcoxon |
| Ipsi_Lung_Dmax      | 24.54 ± 1.05      | 24.01 ± 0.93        |    0.1893 | Wilcoxon |
| Ipsi_Lung_Dmean     | 2.93 ± 0.52       | 2.80 ± 0.22         |    0.3028 | T-test   |
| Ipsi_Lung_V8Gy      | 9.17 ± 2.85       | 8.31 ± 1.74         |    0.2408 | T-test   |
| MAE_CTV             | 0.00 ± 0.00       | 0.44 ± 0.26         |    0      | Wilcoxon |
| MAE_Exter           | 0.00 ± 0.00       | 0.32 ± 0.13         |    0      | Wilcoxon |
| MAE_Total           | 0.00 ± 0.00       | 0.08 ± 0.04         |    0      | Wilcoxon |
| SSIM_CTV            | 1.00 ± 0.00       | 0.91 ± 0.02         |    0      | T-test   |
| SSIM_Exter          | 1.00 ± 0.00       | 0.93 ± 0.03         |    0      | T-test   |
| SSIM_Total          | 1.00 ± 0.00       | 0.98 ± 0.01         |    0      | T-test   |
