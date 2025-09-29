# Resume_Screening
# Abstract
The rapid growth of job applications in modern recruitment makes manual resume screening a time-consuming and inefficient process. Resume Screening Automation offers a scalable solution by applying Natural Language Processing (NLP) and Machine Learning techniques to extract, process, and classify resume data. In this system, resumes are first preprocessed using text-cleaning methods such as tokenization, stop-word removal, and lemmatization. A TF-IDF Vectorizer is then applied to convert the unstructured text into numerical feature representations. These features are used to train a Random Forest Classifier, which learns to categorize resumes into predefined job roles such as Data Scientist, AI Researcher, Cybersecurity Analyst, and Software Engineer. By combining the TF-IDF vectorization technique with the Random Forest model, the system achieves high accuracy in predictions while maintaining interpretability. This automated approach reduces recruitment effort, minimizes human bias, and improves efficiency by enabling recruiters to focus on the most suitable candidates
# Import Libraries
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# Import CSV file
import csv file by using panada library.
# Remove Commas
In Resume ,there are many commas in skill column so first remove commas by using replace function.
# Save Clean Dataset
After removing commas ,save comma free dataset.
<img width="1184" height="531" alt="image" src="https://github.com/user-attachments/assets/431df8f8-c053-4a1d-ad71-13e0944a9e8c" />
# Count job_Roles
<img width="511" height="257" alt="image" src="https://github.com/user-attachments/assets/88e67146-15a1-46b1-ac92-8b02ef9c3dc1" />
# Visualisation of count
<img width="510" height="411" alt="image" src="https://github.com/user-attachments/assets/6c22a971-4abe-4698-b67c-d8c3a39b9c65" />
# Apply Vectorizing and Tfidf
<img width="215" height="93" alt="image" src="https://github.com/user-attachments/assets/104dd9ac-4898-4970-9941-41398075fb23" />
# Feature Engeering and split the columns:
In feature engeering job roles take as targetted variable.
# Fit the Model
Fit random forest classifier model.
# Evaluation
<img width="1133" height="539" alt="image" src="https://github.com/user-attachments/assets/093567d4-7618-44e0-90e1-1018c226fba2" />
Evaluation si an excellent by model.
# Visualisation:
<img width="643" height="609" alt="image" src="https://github.com/user-attachments/assets/c8fc8594-9985-4d59-9bc8-acb586835b7a" />
# Predict Job_Roles
# Example_1 _resume
resume_text = "Deep Learning  Machine Learning  Python  SQL"
Put these skills and model screened that this candidate is suitable for data sientist role.
<img width="582" height="59" alt="image" src="https://github.com/user-attachments/assets/91d38b15-ee69-4224-a31e-0b0e6192308b" />
# Example_2 _resume
resume_text = "TensorFlow  Pytorch  NLP  Python"
output:
<img width="569" height="54" alt="image" src="https://github.com/user-attachments/assets/aac82a8c-df5e-41f0-b4f0-37d9fe0ebca0" />
# Save File
save this file for later use.
<img width="469" height="62" alt="image" src="https://github.com/user-attachments/assets/8fa4c2f1-826a-4bb8-ba15-7fd51096b82e" />








