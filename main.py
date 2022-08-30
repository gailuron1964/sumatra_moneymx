import json
import os
import streamlit as st
from sumatra import Client
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SUMATRA_API_KEY = '02OihyaHKK23Kgnb7PW9Q2HpvKKIrIoQ37MTzkQG'
SUMATRA_SDK_KEY = 'wusRdeGEGx3LOzVFqkwU67RKc2cNouUV71e8ACqp'
os.environ['SUMATRA_SDK_KEY'] = SUMATRA_SDK_KEY
os.environ['SUMATRA_API_KEY'] = SUMATRA_API_KEY

st.set_page_config(
  page_title="Sumatra demo", page_icon="🐯", initial_sidebar_state="collapsed"
)

def enriched():
  if 'enriched' not in st.session_state:
    with st.spinner(text="Replaying 107,028 events..."):
      sumatra = Client('console.qa.sumatra.ai')
      sumatra.create_branch_from_dir('topology')
      sumatra.create_timeline_from_file('timeline', 'moneymx_reformat.jsonl')
      st.session_state.enriched = sumatra.materialize(timeline='timeline')
  return st.session_state.enriched

def labeled():
  if 'labeled' not in st.session_state:
    with st.spinner(text="Joining with labels"):
      spei_outgoing = enriched().get_events("spei_outgoing")
      disputes = pd.read_csv("disputes.csv")
      labeled = pd.merge(spei_outgoing, disputes, 'left', left_on='the_id', right_on='sumatra_id')
      labeled['is_fraud'] = ~labeled.sumatra_id.isna()
      st.session_state.labeled = labeled
  return st.session_state.labeled

st.markdown("### Risk features")
with open("topology/features.scowl", "r") as f:
  st.code(f.read())

st.markdown("### Augmented events")  
st.write(enriched().get_events("spei_outgoing")[:10])

st.markdown("### Labels")
st.write(labeled().is_fraud.value_counts())

st.markdown("### Training & testing")
options = [
  'amount',
  'name_similarity',
  'money_out_48h',
  'past_pair_money_transferred',
  'unique_senders_to_beneficiary',
  'max_failed_logins',
  'days_since_device_update'
]
features = st.multiselect("Features", options, default=options)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed = pd.DataFrame(imp.fit_transform(labeled()[features+['is_fraud']]), columns=features+['is_fraud'])
train, test = train_test_split(imputed, shuffle=False, test_size=0.2)

clf = RandomForestClassifier(n_estimators=5)
clf.fit(train[features], train.is_fraud)

test_score = clf.predict_proba(test[features])[:,1]
fpr, tpr, thresholds = roc_curve(test.is_fraud, test_score)

plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 18
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test Set Performance (Random Forest)")
plt.plot(fpr, tpr, label='Random forest classifier')
st.pyplot(plt.gcf())
