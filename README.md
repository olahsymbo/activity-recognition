# Human Activity Recognition using Decision Tree


This project focuses on using decision tree classifier for human activity recognition from data captured using mobile sensor.
The data consists of a group of 30 individuals within the age of 19-48 years, who volunteered to participate. 
Each subject performed six different activities: 

1-	WALKING, 

2-	WALKING_UPSTAIRS, 

3-	WALKING_DOWNSTAIRS, 

4-	SITTING, 

5-	STANDING, 

6-	LAYING. 

####

During the motion sessions, each subject is wearing a smartphone on their waist.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.  

- Download the sample data from UCI repository https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

- Run the following script to train and test the activity recognition model:

  ```
  python3 actionrecogn.py
  ```

- You can also run the following script to perform feature selection using LASSO, before training and testing the model

  ```
  python3 actionrecogn_featsel.py
  ```
