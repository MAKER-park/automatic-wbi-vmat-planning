# Dose Prediction Evaluation Report

## Evaluation Results for WL (Whole Left) (n=20)
| Metric              | GT (mean ± std)   | Pred (mean ± std)   |   p-value | Method   |
|:--------------------|:------------------|:--------------------|----------:|:---------|
| CTV_Dmax            | 27.42 ± 0.29      | 26.71 ± 0.08        |    0      | T-test   |
| CTV_Dmean           | 26.04 ± 0.29      | 26.07 ± 0.07        |    0.0121 | Wilcoxon |
| CTV_V95%            | 96.94 ± 1.08      | 98.51 ± 0.47        |    0      | T-test   |
| Contra_Breast_Dmax  | 6.55 ± 3.40       | 8.65 ± 3.60         |    0.003  | T-test   |
| Contra_Breast_Dmean | 1.34 ± 0.76       | 1.85 ± 0.59         |    0.0008 | T-test   |
| Contra_Lung_Dmax    | 4.52 ± 1.91       | 6.65 ± 1.70         |    0.0013 | T-test   |
| Contra_Lung_Dmean   | 0.71 ± 0.28       | 1.01 ± 0.23         |    0.0001 | T-test   |
| DICE_50%            | 1.00 ± 0.00       | 0.94 ± 0.02         |    0      | Wilcoxon |
| DICE_95%            | 1.00 ± 0.00       | 0.95 ± 0.01         |    0      | T-test   |
| External_Dmax       | 27.42 ± 0.29      | 26.71 ± 0.08        |    0      | T-test   |
| External_Dmean      | 1.61 ± 0.38       | 1.72 ± 0.35         |    0.0018 | T-test   |
| Heart_Dmax          | 3.95 ± 2.54       | 3.72 ± 1.11         |    0.1429 | Wilcoxon |
| Heart_Dmean         | 0.83 ± 0.17       | 0.95 ± 0.09         |    0.0008 | T-test   |
| Heart_V1.5Gy        | 6.43 ± 5.64       | 6.48 ± 4.30         |    0.9615 | T-test   |
| Heart_V7Gy          | 0.01 ± 0.04       | 0.00 ± 0.00         |    0.1088 | Wilcoxon |
| Ipsi_Lung_Dmax      | 23.45 ± 1.39      | 23.89 ± 0.79        |    0.0705 | T-test   |
| Ipsi_Lung_Dmean     | 2.45 ± 0.52       | 2.53 ± 0.35         |    0.3532 | T-test   |
| Ipsi_Lung_V8Gy      | 7.80 ± 2.44       | 6.60 ± 1.77         |    0.0194 | T-test   |
| MAE_CTV             | 0.00 ± 0.00       | 0.35 ± 0.21         |    0      | Wilcoxon |
| MAE_Exter           | 0.00 ± 0.00       | 0.30 ± 0.08         |    0      | Wilcoxon |
| MAE_Total           | 0.00 ± 0.00       | 0.08 ± 0.03         |    0      | Wilcoxon |
| SSIM_CTV            | 1.00 ± 0.00       | 0.93 ± 0.02         |    0      | Wilcoxon |
| SSIM_Exter          | 1.00 ± 0.00       | 0.93 ± 0.02         |    0      | T-test   |
| SSIM_Total          | 1.00 ± 0.00       | 0.98 ± 0.01         |    0      | T-test   |

## Evaluation Results for WR (Whole Right) (n=20)
| Metric              | GT (mean ± std)   | Pred (mean ± std)   |   p-value | Method   |
|:--------------------|:------------------|:--------------------|----------:|:---------|
| CTV_Dmax            | 27.37 ± 0.25      | 26.61 ± 0.09        |    0      | T-test   |
| CTV_Dmean           | 26.06 ± 0.38      | 26.06 ± 0.05        |    0.0484 | Wilcoxon |
| CTV_V95%            | 97.28 ± 1.22      | 98.48 ± 0.73        |    0.0005 | T-test   |
| Contra_Breast_Dmax  | 9.46 ± 4.87       | 9.40 ± 3.73         |    0.936  | T-test   |
| Contra_Breast_Dmean | 1.43 ± 0.68       | 1.60 ± 0.59         |    0.4033 | T-test   |
| Contra_Lung_Dmax    | 5.00 ± 2.10       | 5.79 ± 2.01         |    0.2016 | T-test   |
| Contra_Lung_Dmean   | 0.75 ± 0.30       | 0.81 ± 0.17         |    0.4964 | T-test   |
| DICE_50%            | 1.00 ± 0.00       | 0.93 ± 0.03         |    0      | Wilcoxon |
| DICE_95%            | 1.00 ± 0.00       | 0.95 ± 0.01         |    0      | T-test   |
| External_Dmax       | 27.37 ± 0.25      | 26.61 ± 0.09        |    0      | T-test   |
| External_Dmean      | 1.82 ± 0.39       | 1.82 ± 0.31         |    0.9808 | T-test   |
| Heart_Dmax          | 2.87 ± 1.40       | 2.70 ± 0.72         |    0.8983 | Wilcoxon |
| Heart_Dmean         | 0.83 ± 0.18       | 0.84 ± 0.10         |    0.4091 | Wilcoxon |
| Heart_V1.5Gy        | 6.31 ± 5.00       | 3.43 ± 2.09         |    0.0362 | Wilcoxon |
| Heart_V7Gy          | 0.01 ± 0.02       | 0.00 ± 0.00         |    0.3173 | Wilcoxon |
| Ipsi_Lung_Dmax      | 24.54 ± 1.05      | 24.61 ± 0.70        |    0.8061 | T-test   |
| Ipsi_Lung_Dmean     | 2.93 ± 0.52       | 2.89 ± 0.22         |    0.7914 | T-test   |
| Ipsi_Lung_V8Gy      | 9.17 ± 2.85       | 8.01 ± 1.45         |    0.1139 | T-test   |
| MAE_CTV             | 0.00 ± 0.00       | 0.39 ± 0.26         |    0      | Wilcoxon |
| MAE_Exter           | 0.00 ± 0.00       | 0.34 ± 0.13         |    0      | Wilcoxon |
| MAE_Total           | 0.00 ± 0.00       | 0.08 ± 0.04         |    0      | Wilcoxon |
| SSIM_CTV            | 1.00 ± 0.00       | 0.92 ± 0.02         |    0      | T-test   |
| SSIM_Exter          | 1.00 ± 0.00       | 0.92 ± 0.03         |    0      | T-test   |
| SSIM_Total          | 1.00 ± 0.00       | 0.98 ± 0.01         |    0      | T-test   |
