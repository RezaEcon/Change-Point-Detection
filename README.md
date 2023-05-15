# Change-Point-Detection
Detecting Turning Points in Business Cycles using Machine Learning

Turning points in business cycles are defined as the onset of a recession or an expansion which are quite difficult to be predicted. Unlike the conventional econometric approaches, we approach the problem of turning (change) point detection from the viewpoint of binary classification task. Due to the small ratio of changes to total data (as the number of recessions is relatively low), we face heavily class-imbalance challenge in this problem. We explore a wide variety of machine learning-based solutions for this problem: from standalone base classifiers to the hybrid classifier ensemble algorithm with/out a feature selection step.

The proposed classification methods were examined on 3 large monthly databases for macroeconomic analysis, in US, UK and Canada. Moving from basic standalone classifiers to hybrid ensemble methods, we progressively add more goal-driven complexity to the method toward reaching to the target performance. In order for classifiers to receive time series data, we transform them into features of rolling windows: 
![Presentation1_2-2-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/97c5a525-c9d8-435b-b6b2-bf4ec1ec0b72)

The utilized methods can be described as follows:

Method 0: Employing an ensemble of classifiers that receive all features collectively.
![Presentation1_3-3](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/0aeba8d9-9856-4e26-bf8d-5352bf5020d9)

Method 1: Initially, each classifier is provided with an individual time series. The final decision for each window is based on the majority vote among the labels predicted by trained classifiers on the respective time series.
![Presentation1_4-4-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/2e13009f-324e-464f-a206-72a1cc36de84)

Method 2: Similar to method 1 in the first stage, but an additional feature selection step is included in the second stage. The majority vote is calculated over a set of features with the highest performance from the first stage.
![Presentation1_5-5-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/8f4e2a82-aff0-4395-bf94-2f9f5874c402)

Method 3: Resembling method 2, except that the labels predicted for the selected features are fed into a new classifier, rather than solely relying on the majority vote.
![Presentation1_6-6-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/dbeff26c-5bd3-435b-8b9d-2371e7c49f31)

Method 4: Similar to method 3, except that an ensemble of classifiers is employed in the second stage instead of a single classifier.
![Presentation1_7-7-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/f586aedc-c11d-4629-bbb6-5a0439c0fe9d)

Methods 5, 6, 7, 8: Essentially methods 1, 2, 3, 4 respectively, with the distinction that an ensemble of classifiers is used in the first stage, rather than a single classifier.
![Presentation1_8-8-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/d3cd07e5-0c2a-48a4-a039-ad4c201cd6a8)
![m6](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/3b3950d2-50d3-4e77-8a8b-9d98a17511c8)
![Presentation1_9-9-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/fd253e1c-afe8-4ea0-82da-d48f4deaf968)
![Presentation1_10-10-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/0f01c664-9a7d-4fa9-80ff-a4e807510df1)
