# Change-Point-Detection
Detecting Turning Points in Business Cycles using Machine Learning

Turning points in business cycles are defined as the onset of a recession or an expansion which are quite difficult to be predicted. Unlike the conventional econometric approaches, we approach the problem of turning (change) point detection from the viewpoint of binary classification task. Due to the small ratio of changes to total data (as the number of recessions is relatively low), we face heavily class-imbalance challenge in this problem. We explore a wide variety of machine learning-based solutions for this problem: from standalone base classifiers to the hybrid classifier ensemble algorithm with/out a feature selection step.

The proposed classification methods were examined on 3 large monthly databases for macroeconomic analysis, in US, UK and Canada. Moving from basic standalone classifiers to hybrid ensemble methods, we progressively add more goal-driven complexity to the method toward reaching to the target performance. 

In order to feed classifiers with Time Series data, a transformation into windows is introduced:

![Presentation1_1-1](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/b96dc4ad-cc96-4d89-8746-fcc8b50b647f)

The utilized methods can be described as follows:
Method 0: Employing an ensemble of classifiers that receive all features collectively.
[Presentation1_2-2.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479476/Presentation1_2-2.pdf)

Method 1: Initially, each classifier is provided with an individual time series. The final decision for each window is based on the majority vote among the labels predicted by trained classifiers on the respective time series.
[Presentation1_3-3.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479477/Presentation1_3-3.pdf)

Method 2: Similar to method 1 in the first stage, but an additional feature selection step is included in the second stage. The majority vote is calculated over a set of features with the highest performance from the first stage.
[Presentation1_4-4.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479479/Presentation1_4-4.pdf)

Method 3: Resembling method 2, except that the labels predicted for the selected features are fed into a new classifier, rather than solely relying on the majority vote.
[Presentation1_5-5.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479480/Presentation1_5-5.pdf)

Method 4: Similar to method 3, except that an ensemble of classifiers is employed in the second stage instead of a single classifier.
[Presentation1_6-6.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479481/Presentation1_6-6.pdf)

Methods 5, 6, 7, 8: Essentially methods 1, 2, 3, 4 respectively, with the distinction that an ensemble of classifiers is used in the first stage, rather than a single classifier.
[Presentation1_7-7.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479484/Presentation1_7-7.pdf)
[Presentation1_8-8.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479485/Presentation1_8-8.pdf)
[Presentation1_9-9.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479486/Presentation1_9-9.pdf)
[Presentation1_10-10.pdf](https://github.com/RezaEcon/Change-Point-Detection/files/11479483/Presentation1_10-10.pdf)
