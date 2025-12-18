import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

# page configuration
st.set_page_config("Simple Linear Regression",layout="centered")

# load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

load_css("styles.css")

# title
st.markdown("""
    <div class="card">
        <h1>Simple Linear Regression</h1>
        <p>Predicts <b>Tip Amount</b> from <b>Total Bill</b> using Simple Linear Regression</p>               
    </div>
""",unsafe_allow_html=True)

# load dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

# dataset preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

# prepare data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)   

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# metrics
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

# visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip Amount ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)


# performance
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae : .2f}")
c2.metric("RMSE",f"{rmse : .2f}")
c3,c4=st.columns(2)
c3.metric("R2",f"{r2 : .2f}")
c4.metric("Adjusted R2",f"{adj_r2 : .2f}")
st.markdown('</div>',unsafe_allow_html=True)

# m & c(slope and intercept)
st.markdown(f"""
    <div class="card">
    <h3>Model Interception</h3>
    <p>
       <b>Co-efficient:</b> {model.coef_[0] : .3f}</br>
       <b>Intercept:</b> {model.intercept_ : .3f}
    </p>
    </div>
    """,unsafe_allow_html=True)

# prediction part
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")
total_bill_min= df["total_bill"].min()
total_bill_max= df["total_bill"].max()
bill = st.slider(
    "Total Bill Amount ($):",float(total_bill_min), float(total_bill_max),30.0)
tip = model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box"><h3>Predicted Tip Amount : ${tip : .2f}</h3></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



