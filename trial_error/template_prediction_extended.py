import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import shap
from shap.maskers import Independent
import matplotlib.pyplot as plt

positive_train = sorted(list(os.listdir("crabtree_positive")))
negative_train = sorted(list(os.listdir("crabtree_negative")))

for files in positive_train :
    with open("crabtree_positive"/Path(files), "r") as src, open("combined.fasta", 'a') as dest:
        dest.write(src.read()+"\n")

for files in negative_train :
    with open("crabtree_negative"/Path(files), "r") as src, open("combined.fasta", 'a') as dest:
        dest.write(src.read()+"\n")
        
seq = []
for sequence in SeqIO.parse("combined.fasta","fasta"):
    seq.append(str(sequence.seq))
full_df = pd.DataFrame(seq, columns=["sequence"])
sense_list = ["positive"]*len(positive_train) + ["negative"]*len(negative_train)
full_df["sense"]=sense_list

full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

full_df['words'] = full_df.apply(lambda x: Kmers_funct(x["sequence"]), axis=1)
full_label = np.array(full_df["sense"])

full_labels = list(full_df.sense)
full_texts = list(full_df["words"])
for item in range(len(full_texts)):
    full_texts[item] = " ".join(full_texts[item])
cvec = CountVectorizer(ngram_range=(4,6),analyzer="char")
full_vec = cvec.fit_transform(full_texts)
count_array = full_vec.toarray()
vec_df = pd.DataFrame(data=count_array, columns=cvec.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(vec_df,full_label,test_size=8/len(full_label),random_state=42)

lreg = LogisticRegression(max_iter=500)
cv_results = cross_validate(lreg,x_train,y_train,cv=10,scoring="f1_weighted",return_estimator=True)
print(f"Mean of 10-fold CV f1-scores : {np.mean(cv_results["test_score"])}\nStandard Deviation of 10-fold CV f1-scores : {np.std(cv_results["test_score"])}")

best_model_idx = np.argmax(cv_results["test_score"])
best_model = cv_results["estimator"][best_model_idx]
y_pred = best_model.predict(x_test)
print(f"Model Accuracy on test data : {best_model.score(x_test,y_test)}")

pd.DataFrame(classification_report(y_test,y_pred,output_dict=True))
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()

label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)
y_proba = best_model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_proba[:,1]) 
plt.plot(fpr,tpr,color="red",label="Predicted")
plt.plot([0,1],[0,1],color="blue",label="random guess line (AUC=0.5)")
plt.title("AUC score : "+str(roc_auc_score(y_test_encoded,y_proba[:, 1])))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

masker = Independent(x_train)
explainer = shap.LinearExplainer(best_model, masker=masker)
shap_values = explainer(x_test)
rng = np.random.default_rng(42)
shap.summary_plot(shap_values.values, x_test, rng=rng)
plt.show()

global_importance = pd.DataFrame({'feature': x_test.columns, 
                                  'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
                                  }).sort_values('mean_abs_shap', ascending=False)
print(f"\nGlobal feature importance table :\n{global_importance.head(5)}")

shap_df = pd.DataFrame({"feature": x_test.columns,
                        "value": shap_values.data[6],
                        "shap_value": shap_values.values[6] })
shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
print(f"\nFeature importance for one test sequence :\n{shap_df.head(10)}")
