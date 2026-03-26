# ASOI-Python

A Python implementation of the Anomaly Separation and Overlap Index (ASOI) algorithm.

TODO: Add a summary of what this metric is and what it is used for.

## Abstract
> Evaluating unsupervised anomaly detection presents significant challenges due to the absence of ground truth labels and the complex nature of anomaly distributions. In this study, we introduce two novel intrinsic evaluation metrics: the Anomaly Separation Index (ASI) and the Anomaly Separation and Overlap Index (ASOI), designed to overcome the limitations of traditional metrics, which cannot assess model performance without labels. ASI quantifies the degree of separation between detected anomalies and normal distributions, while ASOI incorporates both separation and distributional overlap between them, providing an innovative evaluation approach for anomaly detection models, enabling performance assessment even in the absence of ground truth labels. Extensive experiments through precision degradation tests and unsupervised anomaly detection algorithms were conducted on multiple datasets. The results indicate that the metrics consistently correlate with traditional metrics, such as the score, in various benchmark datasets characterized by complex feature interactions and varying levels of anomaly contamination. ASOI showed a higher correlation with the score compared to ASI and several other classical intrinsic metrics. Furthermore, the findings underscore the utility of ASOI as an internal validation measure for model optimization in unsupervised anomaly tasks. The proposed metrics are computationally efficient, scalable, and adaptable to a variety of anomaly detection scenarios, making them practical for real-world applications across industries such as cybersecurity, fraud detection, and predictive maintenance.

## Research Paper
https://link.springer.com/article/10.1007/s40747-025-02204-0

Authors:
- Jiyan Salim Mahmud
- Zakarya Farou
- Imre Lendák

## Implementation Details
TODO: Detail the specifics and decisions made with this implementation.

## Original Algorithm
<img width="537" height="711" alt="asoi_algorithm" src="https://github.com/user-attachments/assets/0d32818d-4bda-49b4-b1e4-cb7ec4ba92c9" />
