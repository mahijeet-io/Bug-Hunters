import pandas as pd
from sklearn.ensemble import IsolationForest
import networkx as nx
from pyvis.network import Network
import streamlit as st
from matplotlib import pyplot as plt

st.title("💰 AI Financial Crime Detection System")

















###AI-Driven Anomaly Detection & Risk Scoring

df = pd.read_csv("transactions.csv")

##Anomaly Detection

df['txn_count'] = df.groupby('sender')['amount'].transform('count')
df['avg_amount'] = df.groupby('sender')['amount'].transform('mean')
X = df[['amount', 'txn_count', 'avg_amount']]

model = IsolationForest(contamination=0.2)

##Risk Scoring

df['anomaly'] = model.fit_predict(X)
df['anomaly_score'] = model.decision_function(X)

##UI for the same
#transaction details
st.subheader("📊 Transaction Data")
df.index = df.index + 1
st.dataframe(df)

#Suspicious Accounts
st.subheader("🚨 Suspicious Accounts")

suspicious_df = df[df['anomaly']==-1]#rows containing on;y anomoly as -1
suspicious_accounts = set(suspicious_df['sender']).union(set(suspicious_df['receiver']))
top_suspicious = suspicious_df.sort_values(by='anomaly_score')

suspicious_accounts_df = pd.DataFrame(list(suspicious_accounts), columns=['Suspicious Account'])
suspicious_accounts_df.index+=1
suspicious_accounts_details = df[df['anomaly']==-1][['sender', 'receiver', 'amount', 'anomaly_score']]
st.dataframe(suspicious_accounts_details, use_container_width=True)

df["risk_score"] = 0
df["warning"] = ""
df["is_fraud"] = False

df['timestamp'] = pd.to_datetime(df["timestamp"])
df = df.reset_index(drop=True)

for i in range(len(df)):
    amount = df.loc[i, "amount"]
    risk = 0
    warnings = []
    hour = df.loc[i, 'timestamp'].hour

    if amount > 15000:
        risk += 50
        warnings.append("High Amount Transaction")

    if hour >= 0 and hour <= 5:
        risk += 20
        warnings.append("Odd Hour Transaction")

    elif amount > 18000:
        risk += 30
        warnings.append("Very High Amount")

    sender = df.loc[i, "sender"]
    sender_transactions = df[df["sender"]==sender]
    average_amount = sender_transactions["amount"].mean()

    if amount > average_amount*3:
        risk += 30
        warnings.append("Unusual Behaviour Detected")
    
    if df.loc[i, "anomaly"]==-1:
        risk += 40
        warnings.append("AI Detected Anomaly")

    if risk >= 70:
        fraud = True

    else:
        fraud = False

    df.loc[i, "risk_score"]=risk
    df.loc[i, "warning"]=", ".join(warnings)
    df.loc[i, "is_fraud"]=fraud

st.subheader("🚨 Fraud Detection Alerts")

fraud_df = df[df['is_fraud'] == True][['sender','receiver','amount','risk_score','warning']].copy()
fraud_df.index = fraud_df.index + 1
st.dataframe(fraud_df, use_container_width=True)

st.subheader("Anomaly Insights")

def scatter_plot():
    plt.figure()

    normal = df[df['anomaly'] == 1]
    anomalous = df[df['anomaly'] == -1]

    plt.scatter(normal.index, normal['amount'], color='blue', label='Normal')
    plt.scatter(anomalous.index, anomalous['amount'], color='red', label='Anomalous')

    plt.title("AI-Based Anomaly Detection")
    plt.xlabel("Transaction Index")
    plt.ylabel("Amount")
    plt.legend()

    st.pyplot(plt)

def amtvsavg():
    plt.figure()

    plt.scatter(df['avg_amount'], df['amount'], c=df['anomaly'], cmap='coolwarm')

    plt.title("Behavioral Deviation Pattern")
    plt.xlabel("Average Transaction Amount")
    plt.ylabel("Current Transaction Amount")

    st.pyplot(plt)

def trend():
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    trend = df.groupby('date')['anomaly'].apply(lambda x: (x == -1).sum())

    plt.figure()
    trend.plot()

    plt.title("Anomaly Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Anomalies")

    st.pyplot(plt)

st.markdown("Anomaly Detection")
scatter_plot()
st.markdown("📉 Behavioral Deviation")
amtvsavg()
st.markdown("Trends")
trend()

st.subheader("📊 Final System Summary")

total_txn = len(df)
anomalies = len(df[df['anomaly'] == -1])
frauds = len(df[df['is_fraud'] == True])

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_txn)
col2.metric("Anomalies Detected", anomalies)
col3.metric("Fraud Cases", frauds)

st.markdown("---")

if frauds > 0:
    st.markdown("**System Risk Level: HIGH**")
    st.badge("⚠️Attention Required!", color='red')
elif anomalies > 0:
    st.markdown("**System Risk Level: MEDIUM**")
    st.badge("Needs Review", color='orange')
else:
    st.markdown("**System Risk Level: LOW**")
    st.badge("System is good", color='green')




