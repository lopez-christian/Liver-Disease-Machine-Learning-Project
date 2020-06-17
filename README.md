# Applying Various Machine Learning Algorithms and Hyperparameter Tuning

## Using The Indian Liver Patient Records Dataset From Kaggle 

<p align="center">
<img width="661" alt="Screen Shot 2020-06-15 at 6 39 33 PM" src="https://user-images.githubusercontent.com/53641091/84722268-910f9100-af37-11ea-8950-8210f5ccd481.png">
</p>

For this project I went ahead and implemented a number of machine learning algorithms on the dataset: Indian Liver Patient Records - Patient records collected from North East of Andhra Pradesh, India [click here](https://www.kaggle.com/uciml/indian-liver-patient-records). The goal was to better predict whether an individual is likely to develop liver disease given certain features which included the age, gender, total bilirubin, direct bilirubin, alkaline phosphotase, alamine aminotransferase, aspartate aminotransferase, total proteins, albumin, and albumin and globulin ratio of the individual.  

<p align="center">
<img width="478" alt="Screen Shot 2020-06-16 at 8 57 44 PM" src="https://user-images.githubusercontent.com/53641091/84853424-094b8480-b014-11ea-9442-b2d3180ace06.png">
</p>

<p align="center">
<img width="1010" alt="Screen Shot 2020-06-16 at 12 04 03 PM" src="https://user-images.githubusercontent.com/53641091/84817262-7dad0600-afc9-11ea-97eb-69f1247040e7.png">
</p>

The data mining and exploration step dealt some interesting insights regarding the data. There were some compelling countplots and undelying correlations that I came across. I won't list them all, but I will say that most of the people that were tested were male patients. The description of the dataset did not provide any background as to why this is so. Also, most of the patients that tested positive were male. The discrepencies were astonishing. Adults aged 24-63 were also significantly impacted by this liver disease, as opposed to young patients, and the elderly. There were also high correlations between certain features like direct bilirubin v. total bilirubin, alamine aminotransferase v. aspartate aminotransferase, total proteins v. albumin, and albumin v. albumin and globulin ratio. 

<p align="center">
<img width="396" alt="Screen Shot 2020-06-16 at 9 44 59 PM" src="https://user-images.githubusercontent.com/53641091/84857043-ff7a4f00-b01c-11ea-9b4d-e4aa6ff965d0.png">
</p>

<p align="center">
<img width="284" alt="Screen Shot 2020-06-16 at 10 04 46 PM" src="https://user-images.githubusercontent.com/53641091/84857220-66980380-b01d-11ea-857e-0b6c85d6afcb.png">
</p>

<p align="center">
<img width="565" alt="Screen Shot 2020-06-16 at 9 37 26 PM" src="https://user-images.githubusercontent.com/53641091/84857112-26388580-b01d-11ea-9cca-545ecd14fd38.png">
</p>

<p align="center">
<img width="992" alt="Screen Shot 2020-06-15 at 6 53 26 PM" src="https://user-images.githubusercontent.com/53641091/84723028-881fbf00-af39-11ea-8587-dd7ddfc2bd72.png">
</p>

<p align="center">
<img width="305" alt="Screen Shot 2020-06-16 at 9 07 17 PM" src="https://user-images.githubusercontent.com/53641091/84857335-a7901800-b01d-11ea-8324-4c9076ce01b1.png">
</p>

<p align="center">
<img width="218" alt="Screen Shot 2020-06-16 at 9 50 59 PM" src="https://user-images.githubusercontent.com/53641091/84856528-d0afa900-b01b-11ea-9096-9d0db3576e3c.png">
</p>

<p align="center">
<img width="849" alt="Screen Shot 2020-06-16 at 9 42 47 PM" src="https://user-images.githubusercontent.com/53641091/84856661-25532400-b01c-11ea-8232-de09f94262af.png">
</p>

<p align="center">
<img width="390" alt="Screen Shot 2020-06-15 at 6 43 31 PM" src="https://user-images.githubusercontent.com/53641091/84722596-7e498c00-af38-11ea-98d9-e2f37bdc8d3f.png">
</p>

<p align="center">
<img width="232" alt="Screen Shot 2020-06-16 at 9 04 38 PM" src="https://user-images.githubusercontent.com/53641091/84853863-02714180-b015-11ea-8895-83024314cd94.png">
</p>

<p align="center">
<img width="314" alt="Screen Shot 2020-06-16 at 9 43 30 PM" src="https://user-images.githubusercontent.com/53641091/84856159-ba551d80-b01a-11ea-9815-fbfcfefed776.png">
</p>

<p align="center">
<img width="407" alt="Screen Shot 2020-06-15 at 6 44 23 PM" src="https://user-images.githubusercontent.com/53641091/84722714-d7b1bb00-af38-11ea-9a34-19eae2d874ce.png">
</p>

<p align="center">
<img width="252" alt="Screen Shot 2020-06-16 at 9 12 23 PM" src="https://user-images.githubusercontent.com/53641091/84854279-149faf80-b016-11ea-859e-309782591a46.png">
</p>

<p align="center">
<img width="655" alt="Screen Shot 2020-06-16 at 9 37 38 PM" src="https://user-images.githubusercontent.com/53641091/84855720-a230ce80-b019-11ea-868a-ca2c2027ad76.png">
</p>

<p align="center">
<img width="494" alt="Screen Shot 2020-06-16 at 9 13 39 PM" src="https://user-images.githubusercontent.com/53641091/84854345-431d8a80-b016-11ea-93ba-720d3c9f80e0.png">
</p>

<p align="center">
<img width="424" alt="Screen Shot 2020-06-16 at 10 10 27 PM" src="https://user-images.githubusercontent.com/53641091/84857610-4ddc1d80-b01e-11ea-8ac3-1d023682ff99.png">
</p>

<p align="center">
<img width="159" alt="Screen Shot 2020-06-16 at 10 10 44 PM" src="https://user-images.githubusercontent.com/53641091/84857620-50d70e00-b01e-11ea-8594-287904003fbe.png">
</p>


The dataset was manipulated and cleaned using an assortment of libraries that included Pandas and Matplotlib. There was quite some feature engineering to do that included renaming certain columns, dealing null values, handling outliers, and log-transforming and min-max scaling the continous features. There was also some binning that had to be done regarding the age column in order to make the results easier to interpret and visualize. One-hot encoding was performed on any discrete features. 

<p align="center">
<img width="945" alt="Screen Shot 2020-06-15 at 7 00 05 PM" src="https://user-images.githubusercontent.com/53641091/84723466-80ace580-af3a-11ea-802d-6208f2238b56.png">
</p>

<p align="center">
<img width="461" alt="Screen Shot 2020-06-15 at 7 00 20 PM" src="https://user-images.githubusercontent.com/53641091/84723512-98846980-af3a-11ea-9173-c6840cae4520.png"><img width="461" alt="Screen Shot 2020-06-15 at 7 00 27 PM" src="https://user-images.githubusercontent.com/53641091/84723515-9a4e2d00-af3a-11ea-8be1-821557d4d1b0.png">
</p>

After creating our machine learning-ready-dataset we went ahead and applied our machine learning algorithms. These were classification algorithms that included Support Vector Machines (SVM), K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting Machines (GBM), eXtreme Gradient Boosting (XGBoost), and AdaBoost. These produced varying test metrics, and AUC measures. The. highest ranking proved to be the Random Forest algorithm with Accuracy Score: 0.7678571428571429 and AUC: 0.64140625. The AUCs were used to create and ROC/AUC Curve plot that compared all the algorithms. Having arrived at the conlusion that Random Forest was the best-performing out of the bunch, I went ahead and executed some hyperparameter tuning and optimizations using RandomizedSearchCV. Another ROC/AUC Curve plot was generated to show the newly incorporated Random Forest + Optimization AUC. The Random Forest + Optimization algorithm had Accuracy Score: 0.7440476190476191 and AUC: 0.55703125. This algorithm did slightly worse on both measures when compared to the Random Forest on default parameters. 

<p align="center>
<img width="298" alt="Screen Shot 2020-06-16 at 10 19 17 PM" src="https://user-images.githubusercontent.com/53641091/84858179-82040e00-b01f-11ea-888c-5a3f849a4575.png">
<img width="263" alt="Screen Shot 2020-06-16 at 10 19 30 PM" src="https://user-images.githubusercontent.com/53641091/84858189-86c8c200-b01f-11ea-9620-8eca8a78b288.png">
</p>

<p align="center>
<img width="154" alt="Screen Shot 2020-06-16 at 10 19 44 PM" src="https://user-images.githubusercontent.com/53641091/84858191-8af4df80-b01f-11ea-8f4a-bd4ff94009e6.png">
</p>

<p align="center">
<img width="843" alt="Screen Shot 2020-06-15 at 7 51 54 PM" src="https://user-images.githubusercontent.com/53641091/84726443-b0abb700-af41-11ea-8e2d-d4d62b7b028e.png">
</p>

## Key takeways:

*1. There is evidence of correlations between certain features. What this tells us is that certain features can be indicative of other features being elevated as well. An example of this would be the high positive correlation between direct bilirubin and total bilirubin. If a patient were to come in with high levels of direct bilirubin, we would be safe to assume that the likelihood that they also have a high incidence of total bilirubin is quite high. The health care practitioner could choose to only administer certain tests and not others, which could potentially save a both the healthcare practitioner and patient vasts amounts of time and resources.*

*2. Males comprised most of the dataset by a vast amount. There are many more males than there are females affected by the liver disease as well. We are not made aware of how the data was acquired, but the disparity between genders is astonishing. The measures for all features associated with the disease were much greater in males than in females. These types of discoveries can lead to targeted preventive care for male subjects when they come in for rudimentary check-ups or health issues. Healthcare facilities and various other organizations can take a step in addressing the issue-at-hand and make the public aware of the consequences that are associated with liver disease.*

*3. Adults between the ages of 24-63 years old seem to be the most in danger. This can be attributed to these adults being tested more often for liver disease than are young people and the elderly. We may need to focus more on testing these other demographics. There is a strong possibility that we are missing the greater picture here. If we were to focus on testing the youth and improving their dietary intake, decreasing alcohol consumption, addressing pollution, decontaminating food, and eradicating drug use, we may arrive at a stage where we can prevent our youth from developing liver disease later in their lives.*
