import streamlit as st
import pandas as pd
import joblib
# Function to load the model
@st.cache_data
def load_model():
    with open('big_mart_sales_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Big Mart Sales Prediction Web App')

    # ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
    #    'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
    #    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'],
    # User inputs
    item_weight = st.number_input('Item Weight', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
    item_visibility = st.number_input('Item Visibility', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    item_type = st.selectbox('Item Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'])
    item_mrp = st.number_input('Item MRP', min_value=0.0, max_value=300.0, value=0.0, step=0.1)
    outlet_identifier = st.selectbox('Outlet Identifier', ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'])
    outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1985, max_value=2010, value=1985, step=1)
    outlet_size = st.selectbox('Outlet Size', ['Medium', 'High', 'Small'])
    outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 3', 'Tier 2'])
    outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3'])
    
    # Item_Fat_Content: Low Fat: 0, Regular: 1
    # Outlet_Size: High: 0, Medium: 1, Small: 2
    # Outlet_Location_Type: Tier 1: 0, Tier 2: 1, Tier 3: 2
    # Outlet_Type: Grocery Store: 0, Supermarket Type1: 1, Supermarket Type2: 2, Supermarket Type3: 3
    # Item_Type: 0-15
    user_inputs = {
        'Item_Weight': item_weight,
        'Item_Fat_Content': 0 if item_fat_content == 'Low Fat' else 1,
        'Item_Visibility': item_visibility,
        'Item_Type': 0 if item_type =='Dairy' else 1 if item_type =='Soft Drinks' else 2 if item_type =='Meat' else 3 if item_type =='Fruits and Vegetables' else 4 if item_type =='Household' else 5 if item_type =='Baking Goods' else 6 if item_type =='Snack Foods' else 7 if item_type =='Frozen Foods' else 8 if item_type =='Breakfast' else 9 if item_type =='Health and Hygiene' else 10 if item_type =='Hard Drinks' else 11 if item_type =='Canned' else 12 if item_type =='Breads' else 13 if item_type =='Starchy Foods' else 14 if item_type =='Others' else 15 if item_type =='Seafood' else 0,
        'Item_MRP': item_mrp,
        'Outlet_Identifier': 0 if outlet_identifier =='OUT049' else 1 if outlet_identifier =='OUT018' else 2 if outlet_identifier =='OUT010' else 3 if outlet_identifier =='OUT013' else 4 if outlet_identifier =='OUT027' else 5 if outlet_identifier =='OUT045' else 6 if outlet_identifier =='OUT017' else 7 if outlet_identifier =='OUT046' else 8 if outlet_identifier =='OUT035' else 9 if outlet_identifier =='OUT019' else 0,
        'Outlet_Establishment_Year': outlet_establishment_year,
        'Outlet_Size': 0 if outlet_size == 'High' else 1 if outlet_size == 'Medium' else 2,
        'Outlet_Location_Type': 0 if outlet_location_type == 'Tier 1' else 1 if outlet_location_type == 'Tier 2' else 2,
        'Outlet_Type': 0 if outlet_type == 'Grocery Store' else 1 if outlet_type == 'Supermarket Type1' else 2 if outlet_type == 'Supermarket Type2' else 3        
    }
    
    if st.button('Predict'):
        prediction = loaded_model.predict(pd.DataFrame(user_inputs, index=[0]))
        st.markdown(f'The predicted sales of Big Mart is: {prediction[0]:,.2f}')
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: XGBoost Regressor')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'big_mart_sales_prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="big_mart_sales_prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'Train.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="bigmart_sales_data.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/BigMart-Sales-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is created to predict the sales of Big Mart based on the input features provided by the user.')
    st.write('The model used in this web app is a XGBoost Regressor model which is trained on the Big Mart Sales dataset.')
    st.write('The dataset used in this web app is taken from the Kaggle Datastes. The dataset contains 8523 rows and 12 columns.')
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/BigMart-Sales-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
