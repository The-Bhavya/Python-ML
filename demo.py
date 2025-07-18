import streamlit as st
import pandas as pd

# Sample data
data = {'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
        'Value': [10, 15, 13, 18]}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Display the chart
st.line_chart(df)


import streamlit as st
import pandas as pd

# Sample data
data = {'Category': ['A', 'B', 'C', 'D'],
        'Value': [20, 35, 30, 25]}
df = pd.DataFrame(data)

col1, col2 = st.columns(2)


with col2:
    st.bar_chart(df.set_index('Category'))

    st.line_chart('diabetes.csv')
