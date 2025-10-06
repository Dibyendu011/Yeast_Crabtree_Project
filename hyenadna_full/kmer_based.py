import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split, KFold
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import os
os.chdir("/home/ibab/Downloads/Classifying-DNA-strands-based-on-FASTA-sequence-main/")
os.getcwd()

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

# defining KFold for cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier()

lreg_scores = []
svc_scores = []
rfc_scores = []
xgb_scores = []
knn_scores = []

x_train_kfold = []
x_test_kfold = []
y_train_kfold = []
y_test_kfold = []
lreg_preds = []
svc_preds = []
rfc_preds = []
xgb_preds = []
knn_preds = []

lreg_list = []
svc_list = []
rfc_list = []
xgb_list = []
knn_list = []
cvec_list = []

# running 10-fold CV
for fold_number, (train_idx, test_idx) in enumerate(kf.split(full_texts), 1):
    
    # splitting data
    x_train, x_test = full_texts[train_idx], full_texts[test_idx]
    y_train, y_test = full_label[train_idx], full_label[test_idx]
    
    # passing split data to countvectorizer
    cvec = CountVectorizer(ngram_range=(4,4))
    x_train = cvec.fit_transform(x_train)
    x_test = cvec.transform(x_test)
    cvec_list.append(cvec)
    x_train_kfold.append(x_train)
    x_test_kfold.append(x_test)
    y_train_kfold.append(y_train)
    y_test_kfold.append(y_test)

    # fitting Logistic Regression Classifier
    lreg = LogisticRegression(C=0.1)
    lreg.fit(x_train, y_train)
    y_pred_lreg = lreg.predict(x_test)
    lreg_preds.append(y_pred_lreg) 
    lreg_scores.append(f1_score(y_test, y_pred_lreg))
    lreg_list.append(lreg)

    # fitting Support Vector Classifier
    svc = SVC(probability=True)
    svc.fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)
    svc_preds.append(y_pred_svc) 
    svc_scores.append(f1_score(y_test, y_pred_svc))
    svc_list.append(svc)

    # fitting Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    rfc_preds.append(y_pred_rfc) 
    rfc_scores.append(f1_score(y_test, y_pred_rfc))
    rfc_list.append(rfc)

    # fitting XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    xgb_preds.append(y_pred_xgb) 
    xgb_scores.append(f1_score(y_test, y_pred_xgb))
    xgb_list.append(xgb)

    # fitting KNeighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    knn_preds.append(y_pred_knn)
    knn_scores.append(f1_score(y_test, y_pred_knn))
    knn_list.append(knn)

print(f"Logistic Regression mean f1-score: {np.mean(lreg_scores):.2f}, Std. dev.: {np.std(lreg_scores):.2f}")
print(f"Support Vector Classifier mean f1-score: {np.mean(svc_scores):.2f}, Std. dev.: {np.std(svc_scores):.2f}")
print(f"Random Forest Classifier mean f1-score: {np.mean(rfc_scores):.2f}, Std. dev.: {np.std(rfc_scores):.2f}")
print(f"XGBoost Classifier mean f1-score: {np.mean(xgb_scores):.2f}, Std. dev.: {np.std(xgb_scores):.2f}")
print(f"KNeighbors Classifier mean f1-score: {np.mean(knn_scores):.2f}, Std. dev.: {np.std(knn_scores):.2f}")

roc_auc_score(y_test_kfold[4], lreg_list[4].predict_proba(x_test_kfold[4])[:, 1])
fpr, tpr, thresholds = roc_curve(y_test_kfold[4], lreg_list[4].predict_proba(x_test_kfold[4])[:,1]) 
plt.plot(fpr,tpr,color="red",label="Predicted")
plt.plot([0,1],[0,1],color="blue",label="random guessing curve (AUC=0.5)")
plt.title(f"AUC score : {roc_auc_score(y_test_kfold[4],lreg_list[4].predict_proba(x_test_kfold[4])[:,1]):.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

explainer = shap.Explainer(lreg_list[4], x_train_kfold[4], feature_names=cvec_list[4].get_feature_names_out())
shap_values = explainer(x_test_kfold[4])
shap.plots.beeswarm(shap_values, show=False)
plt.show()

global_importance = pd.DataFrame({'feature': cvec_list[4].get_feature_names_out(), 
                                  'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
                                  }).sort_values('mean_abs_shap', ascending=False)
print(f"\nGlobal feature importance table :\n{global_importance.head(5)}")
