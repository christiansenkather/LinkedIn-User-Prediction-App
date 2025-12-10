import streamlit as st
import pandas as pd
import joblib
import altair as alt

from FinalProject import ss

# Load your trained logistic regression model
model = joblib.load("model.pkl")

st.title("LinkedIn User Live Predictions")
st.write("Predict whether someone uses LinkedIn based on user demographic information.")
st.write("Click tabs below to navigate between user-inputted data for prediction and model visuals & comparisons.")

# Create tabs for organization
tab1, tab2, tab3  = st.tabs(["Prediction by Input", "Model Visuals", "User Comparison"])


# Tab 1: User Inputted Prediction Model

with tab1:
    st.header("Input User Demographics")
   
   # Making user input fields for their demographics
   # Setting start upon streamlit open as mid value
    income = st.slider("Income (1–9)", min_value=1, max_value=9, value=5)
    education = st.slider("Education (1–8)", min_value=1, max_value=8, value=4)
    parent = st.selectbox("Parent?", ["No", "Yes"])
    married = st.selectbox("Married?", ["No", "Yes"])
    female = st.selectbox("Female?", ["No", "Yes"])
    age = st.slider("Age", min_value=18, max_value=98, value=40)

    # Convert categorical inputs, similar to what was done in Jupyter
    parent = 1 if parent == "Yes" else 0
    married = 1 if married == "Yes" else 0
    female = 1 if female == "Yes" else 0

    # Prepare DataFrame for prediction
    input_data = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [married],
    "female": [female],
    "age": [age]
})

    # Button to run prediction
    if st.button("Predict"):
        prob = model.predict_proba(input_data)[:, 1][0]
        prediction = model.predict(input_data)[0]
        # st.session_state["probability"] = prob

        st.subheader("Prediction Results")
        st.write(f"**Predicted Probability of LinkedIn Use:** {prob:.2f}")
        if prediction == 1:
            st.success("This person is predicted **to use LinkedIn**.")
        else:
            st.error("This person is predicted **NOT to use LinkedIn**.")



# Tab 2: Model Visuals: Same as those in Jupyter just built with Altair for Streamlit

with tab2:
    # 1. Age distribution
    st.subheader("LinkedIn Use Age Distribution")
    age_chart = alt.Chart(ss).mark_boxplot().encode(
        x=alt.X("sm_li:O", title="LinkedIn User (0 = No, 1 = Yes)"),
        y=alt.Y("age:Q", title="Age")
    )
    st.altair_chart(age_chart)

    # 2. LinkedIn use rate by income
    st.subheader("LinkedIn Use Rate by Income")
    income_df = ss.groupby("income")["sm_li"].mean().reset_index()
    income_chart = alt.Chart(income_df).mark_bar().encode(
        x=alt.X("income:O", title="Income Level (1-9)"),
        y=alt.Y("sm_li:Q", title="Proportion of Users"),
        tooltip=["income", "sm_li"]
    )
    st.altair_chart(income_chart)

    # 3. LinkedIn use rate by education
    st.subheader("LinkedIn Use Rate by Education Level")
    edu_df = ss.groupby("education")["sm_li"].mean().reset_index()
    edu_chart = alt.Chart(edu_df).mark_bar().encode(
        x=alt.X("education:O", title="Education Level (1-8)"),
        y=alt.Y("sm_li:Q", title="Proportion of Users"),
        tooltip=["education", "sm_li"]
    )
    st.altair_chart(edu_chart)


# Tab 3: User Comparison
with tab3:
    st.header("Compare Your Input to Model Averages")

    # Need to find averages ONLY when a participant is a LinkedIn user
    users = ss[ss["sm_li"] == 1]

    # Calculate dataset averages
    avg_income = users["income"].mean()
    avg_education = users["education"].mean()
    avg_age = users["age"].mean()
    avg_parent = users["parent"].mean()
    avg_married = users["married"].mean()
    avg_female = users["female"].mean()

    # Making a comparison table
    st.subheader("Comparison Table")

    comparison_df = pd.DataFrame({
        "Feature": ["Income", "Education", "Parent?", "Married?", "Female?", "Age"],
        "User Input": [income, education, parent, married, female, age],
        "LinkedIn User Average": [avg_income, avg_education, avg_parent, avg_married, avg_female, avg_age]
    })
    st.table(comparison_df.set_index("Feature"))
    st.write("Note: Income varies from 1-9, Education from 1-8, Parent/Married/Female are binary (0 = No, 1 = Yes), and Age is in years")

    # Bar chart to see comparisons
    st.subheader("Visual Comparison")
    comparison_chart = alt.Chart(comparison_df.melt(id_vars=["Feature"], var_name="Type", value_name="Value")).mark_bar().encode(
        x=alt.X("Feature:N", title="Feature"),
        y=alt.Y("Value:Q", title="Value"),
        color=alt.Color("Type:N", title="Group"),
        column=alt.Column("Type:N", title=""),
        tooltip=["Feature", "Type", "Value"]
    ).properties(width=300)
    st.altair_chart(comparison_chart)