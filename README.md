# Change-Point-Detection
Detecting Turning Points in Business Cycles using Machine Learning

Turning points in business cycles are defined as the onset of a recession or an expansion which are quite difficult to be predicted. Unlike the conventional econometric approaches, we approach the problem of turning (change) point detection from the viewpoint of binary classification task. Due to the small ratio of changes to total data (as the number of recessions is relatively low), we face heavily class-imbalance challenge in this problem. We explore a wide variety of machine learning-based solutions for this problem: from standalone base classifiers to the hybrid classifier ensemble algorithm with/out a feature selection step.

The proposed classification methods were examined on 3 large monthly databases for macroeconomic analysis, in US, UK and Canada. Moving from basic standalone classifiers to hybrid ensemble methods, we progressively add more goal-driven complexity to the method toward reaching to the target performance. 

In order to feed classifiers with Time Series data, a transformation into windows is introduced:

![Presentation1_2-2-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/267dbe33-8b02-4cb8-86f5-a0ea37c1dc5e)

The utilized methods can be described as follows:

Method 0: Employing an ensemble of classifiers that receive all features collectively.

![Presentation1_3-3](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/c1074607-d798-4dc3-97df-063846f8d34f)

Method 1: Initially, each classifier is provided with an individual time series. The final decision for each window is based on the majority vote among the labels predicted by trained classifiers on the respective time series.

![Presentation1_4-4-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/32a0874d-c840-4ca4-a869-b30f1370b154)

Method 2: Similar to method 1 in the first stage, but an additional feature selection step is included in the second stage. The majority vote is calculated over a set of features with the highest performance from the first stage.
![Presentation1_5-5-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/430900ad-c6db-423e-8ba4-335accf04a11)


Method 3: Resembling method 2, except that the labels predicted for the selected features are fed into a new classifier, rather than solely relying on the majority vote.
![Presentation1_6-6-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/5d076882-3cbc-4369-b27e-110ac7b9c39b)


Method 4: Similar to method 3, except that an ensemble of classifiers is employed in the second stage instead of a single classifier.
![Presentation1_7-7-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/62d385af-c8ff-4968-b6fc-561ba903a974)


Methods 5, 6, 7, 8: Essentially methods 1, 2, 3, 4 respectively, with the distinction that an ensemble of classifiers is used in the first stage, rather than a single classifier.
![Presentation1_8-8-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/6b4f6b10-0656-4587-8c47-661a22e6fc83)
![download](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/debd4071-270a-4dfd-9e88-d1edb6895a57)
![Presentation1_9-9-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/16d2a726-574d-4471-aa73-086aa8150f06)
![Presentation1_10-10-page-001](https://github.com/RezaEcon/Change-Point-Detection/assets/105849750/412de898-6c22-4bc4-98d8-64d294ac097f)
