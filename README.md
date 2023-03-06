# Patient_Selection_for_Diabetes_Drug_Clinical_Trial_using_Machine_Learning

As a data scientist, I have a groundbreaking diabetes drug that needs to conduct clinical trials to ensure its safety and efficacy. The drug requires administering it over at least 5-7 days of hospitalization with frequent monitoring and patient medication adherence training with a mobile application.

To identify suitable patients for our clinical trial, I will build a predictive regression model that can identify patients who are likely to require at least 5-7 days of hospitalization and would not incur significant additional costs for administering the diabetes drug. To achieve this, I will utilize a synthetic dataset based on the UCI Diabetes readmission dataset and focus on building the right data representation at the encounter level. I will also employ appropriate filtering and preprocessing/feature engineering techniques on key medical code sets to ensure unbiased model predictions across key demographic groups.


## DataSet
 I am using a modified version of the dataset from UC Irvine (https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008). Please note that it is limited in its representation of some key features such as diagnosis codes which are usually an unordered list in 835s/837s (the HL7 standard interchange formats used for claims and remits).

Data Schema The dataset reference information can be https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/data_schema_references/ . There are two CSVs that provide more details on the fields and some of the mapped values.
