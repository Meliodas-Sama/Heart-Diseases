# Copyright 2018-2021 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




from nbformat import write
from pandas import get_dummies


def intro():
    import streamlit as st

    st.sidebar.success("Select a Page above.")

    st.markdown(
        """
        Heart Diseases Project as a Home Work for Data Mining Subject.

##### Using **_Streamlit_**, **_Numpy_** and **_SKLearn_** **_Python_** libraries our project can:

* Classify if a person has a Heart Disease or not using one of the following models:
    * Gaussian Naive Bayes
    * ID3

* Print the evaluation metrics for the models. 

Made with :heart: as a HomeWork for the ADM subject at my university by Walid_164688.
    """
    )
    


# Turn off black formatting for this function to present the user with more
# compact code.
# fmt: off
def Classification():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import GaussianNB
    
    data = []
    original_data_df = pd.read_excel('data/heart_disease_male.xls', true_values=['True', 'yes'], false_values=['no', 'False'] ,skiprows=[1], na_values=['#NAME?', '?']).bfill()
    for column in original_data_df.columns[:-1]:
        if original_data_df.dtypes[column] == np.int64:
            data.append(st.number_input(column))
        else:
            temp = list(set(original_data_df[column]))
            temp.insert(0, '_')
            data.append(st.selectbox(column,temp))
    # st.write(data)
    
    # data = original_data_df.copy()
    # data ['exercice_angina'] = [False if x == 'no' else True for x in data['exercice_angina']]
    # data_dummies = pd.get_dummies(data)

    with open('.\Models\OurGNB.p','rb') as file:
        gNBModel = pickle.load(file)
    
    # data = pd,get_dummies(data)
    result, probs = GaussianNB.Predict(gNBModel, [data])
    st.write("\nAccording to our GNB model the disease is:", result[0],"\n\nYes Prob: ", probs[0][0],", No Prob: ", probs[0][1])
    # st.write(gNBModel.predict([[1,2,3,4,1,1,1,1,1,1,1,0]]))
    

    




def Evaluation():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import GaussianNB
    import utils

    original_data_df = pd.read_excel('data/heart_disease_male.xls', true_values=['True', 'yes'], false_values=['no', 'False'] ,skiprows=[1], na_values=['#NAME?', '?']).bfill()

    with open('.\Models\OurGNB.p','rb') as file:
        gNBModel = pickle.load(file)
    
    st.markdown("## Our Gaussian Naive Bays Evaluation")

    utils.Evaluate(gNBModel, original_data_df)
    

# fmt: on

# Turn off black formatting for this function to present the user with more
# compact code.
# fmt: off
