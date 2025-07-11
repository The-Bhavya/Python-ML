import streamlit as st
import numpy as np
import joblib


# load the model 
with open('models\simple_linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)


# title
st.set_page_config(page_title='Salary Predictor')
st.title('Salary Predictot App')
st.subheader('Predict the salary based on your experience')


#sidebar
st.sidebar.header('Enter your details')
experience = st.sidebar.slider("years of Experience",
                            min_value = 0.0,max_value=20.0,step=0.5)


# button to predict
if st.sidebar.button('Predict'):
    # predict salary
    salary = model.predict(np.array([[experience]]))[0]

    # display the result
    st.success(f'Predicted Salary: Rs. {salary:,.2f}')    #success is green colour


    # additional information
    st.info("This prediction is based on a simple Linear regression model")


#footer
st.markdown("----------------")
st.markdown("Made with streamlit ")

