import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split, KFold
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
#import shap
import matplotlib.pyplot as plt

os.chdir("/path/to/Classifying-DNA-strands-based-on-FASTA-sequence-main/")
print(os.getcwd())

# making list of files in positive, negative directories
positive_train = sorted(list(os.listdir("crabtree_positive")))
negative_train = sorted(list(os.listdir("crabtree_negative")))

# checking if "combined.fasta" exists. If not, creating a file "combined.fasta" that contains all the 64 sequences
if not Path("combined.fasta").exists():
    for files in positive_train :
        with open("crabtree_positive"/Path(files), "r") as src, open("combined.fasta", 'a') as dest:
            dest.write(src.read()+"\n")

    for files in negative_train :
        with open("crabtree_negative"/Path(files), "r") as src, open("combined.fasta", 'a') as dest:
            dest.write(src.read()+"\n")
            
# parsing sequences from "combined.fasta" & putting them into a dataframe with labels
seq = []
for sequence in SeqIO.parse("combined.fasta","fasta"):
    seq.append(str(sequence.seq))
full_df = pd.DataFrame(seq, columns=["sequence"])
sense_list = [1]*len(positive_train) + [0]*len(negative_train)
full_df["sense"]=sense_list
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# breaking each sequence into 6 nt long kmers & converting all letters to lower case
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# making a column for the kmers & an array containing labels
full_df['words'] = full_df.apply(lambda x: Kmers_funct(x["sequence"]), axis=1)
full_label = np.array(full_df["sense"])

# putting the sequence of kmers into an array
full_texts = np.array([" ".join(words) for words in full_df["words"]])

# Loading word embeddings table
new_df = pd.read_csv("output.csv")

# defining KFold for cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# defining KFold for cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=2020)

lreg_scores = []
svc_scores = []
rfc_scores = []
xgb_scores = []

x_train_kfold = []
x_test_kfold = []
y_train_kfold = []
y_test_kfold = []
lreg_preds = []
svc_preds = []
rfc_preds = []
xgb_preds = []

lreg_list = []
svc_list = []
rfc_list = []
xgb_list = []

# running 10-fold CV
for fold_number, (train_idx, test_idx) in enumerate(kf.split(new_df), 1):
    # splitting data
    x_train, x_test = new_df.iloc[train_idx], new_df.iloc[test_idx]
    y_train, y_test = full_label[train_idx], full_label[test_idx]    
    x_train_kfold.append(x_train)
    x_test_kfold.append(x_test)
    y_train_kfold.append(y_train)
    y_test_kfold.append(y_test)

    # fitting Logistic Regression Classifier
    lreg = LogisticRegression(max_iter=500,C=0.01)
    lreg.fit(x_train, y_train)
    y_pred_lreg = lreg.predict(x_test)
    lreg_preds.append(y_pred_lreg) 
    lreg_scores.append(f1_score(y_test, y_pred_lreg))
    lreg_list.append(lreg)

    # fitting Support Vector Classifier
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)
    svc_preds.append(y_pred_svc) 
    svc_scores.append(f1_score(y_test, y_pred_svc))
    svc_list.append(svc)

    # fitting Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=30, max_features='sqrt', min_samples_split=2,  min_samples_leaf=20, random_state=42)
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    rfc_preds.append(y_pred_rfc) 
    rfc_scores.append(f1_score(y_test, y_pred_rfc))
    rfc_list.append(rfc)

    # fitting XGBoost Classifier
    xgb = XGBClassifier(subsample=0.1,min_child_weight=10,random_state=42)
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    xgb_preds.append(y_pred_xgb) 
    xgb_scores.append(f1_score(y_test, y_pred_xgb))
    xgb_list.append(xgb)
    
print(f"Logistic Regression mean f1-score: {np.mean(lreg_scores):.2f}, Std. dev.: {np.std(lreg_scores):.2f}")
print(f"Support Vector Classifier mean f1-score: {np.mean(svc_scores):.2f}, Std. dev.: {np.std(svc_scores):.2f}")
print(f"Random Forest Classifier mean f1-score: {np.mean(rfc_scores):.2f}, Std. dev.: {np.std(rfc_scores):.2f}")
print(f"XGBoost Classifier mean f1-score: {np.mean(xgb_scores):.2f}, Std. dev.: {np.std(xgb_scores):.2f}")
