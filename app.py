import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def intro():
    st.image("pictures/EjeZ.png", use_column_width=True)

    st.markdown("<style>h1 {text-align: justify;}</style>", unsafe_allow_html=True)
    st.title("Two-Particle System: Analysis and Prediction of the Invariant Mass Based on Automated Learning") 

    st.markdown("""<p style='font-size: 18px; text-align: justify'>
                This project involves a comprehensive data analysis to examine how the various components of a particle influence each other during a collision with another particle. 
                By employing both machine learning and deep learning techniques, we aim to predict the invariant mass of the resulting system. This study not only enhances our 
                understanding of the dynamics involved in particle collisions but also seeks to develop accurate predictive models with significant applications 
                in the field of particle physics. <br>
                The motivation for undertaking this project stems from its significant importance in advancing scientific knowledge and the high costs associated with such 
                research. As will be explained later, this study aims to construct a model capable of approximating the invariant mass of two particles without the need 
                for all experimental data. Instead, the model will rely solely on the movement of the particle along the beam direction and the charge of each particle.</p>
        """, unsafe_allow_html=True)
    
    
    st.markdown("<h3 style='color:gray; font-size: 18px'>Below is an preview example of the data taken from the CERN.</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:gray; font-size: 14px'>European Council for Nuclear Research. </h3>", unsafe_allow_html=True)
        
    if 'show_df' not in st.session_state:
        st.session_state.show_df = True

    if st.session_state.show_df:
        df_no_treatment_demo = pd.read_csv('data/dielectron_cleaned.csv')

        st.write(df_no_treatment_demo)
        st.markdown("<p style='color:gray; font-size: 12px; text-align: right'>https://opendata.cern.ch/record/304.</h3>", unsafe_allow_html=True)


    st.write("The data is stored in a DataBase (MySQL) we have created")
    st.image("Database/Relational.png")



def data_analysis():
    st.title("Data Analysis")

    st.write("""We need to analyse the data to understand how the various components of a particle influence each other during a collision with another particle. 
            The full Data Analysis can be found in the PowerBI attached to this project in GitHub, but we are going to talk about the insights we found in this report.""")
    st.markdown("""<p style='color:gray; font-size: 14px; text-align: justify'>We are dealing with a set of experimental data, so instead of removing any outliers, 
                we will add a boolean column to indicate whether or not each data point is an outlier.""", unsafe_allow_html=True)

    st.title("Variables and Meaning")

    # Usar session_state para mantener el estado
    if 'show_message' not in st.session_state:
        st.session_state.show_message = False

    # Crear un botón que alterna el estado
    if st.button("Show/Hide"):
        st.session_state.show_message = not st.session_state.show_message  # Cambiar el estado

    # Mostrar u ocultar el mensaje según el estado
    if st.session_state.show_message:
        data = {
            "Variable": ["Run", "Event", "E", "px", "py", "pz", "pt", "eta", "phi", "Q", "M"],
            "Description": [
                "The run number of the event.",
                "The event number.",
                "The total energy of the electron (GeV).",
                "The x-component of the electron's momentum (GeV).",
                "The y-component of the electron's momentum (GeV).",
                "The z-component of the electron's momentum (GeV).",
                "The transverse momentum of the electron (GeV).",
                "The pseudorapidity of the electron.",
                "The phi angle of the electron (rad).",
                "The charge of the electron.",
                "The invariant mass of two electrons (GeV)."
            ]
        } 
        st.dataframe(pd.DataFrame(data))

    st.title("Analysis Insights") 
    st.write("1) **The mean energy of both particles is modulated by the Run type.**")
    st.markdown("""<p style='font-size: 16px; text-align: justify'>Notably, the maximum mean energy is achieved when two particles possessing 
                disparate charges are involved.</p>""", unsafe_allow_html=True)
            
    
    st.image("PowerBI/MeanEnergy_v_Event_-1v1.png")
    st.markdown("<p style='color:gray; font-size: 12px; text-align: right'>Distribution of mean energy by run for different charged particles</h3>", unsafe_allow_html=True)

    st.image("PowerBI/MeanEnergy_v_Event_1v1.png")
    st.markdown("<p style='color:gray; font-size: 12px; text-align: right'>Distribution of mean energy by run for same charged particles</h3>", unsafe_allow_html=True)

    st.write("2) **The pseudorapidity approaches zero as the energy reaches its minimum.**") 
    st.markdown("""<p style='font-size: 16px; text-align: justify'>In contrast, maximum energy is attained at pseudorapidity values of ±2.5. 
            This indicates a significant dependence of energy on pseudorapidity.</p>""", unsafe_allow_html=True) 
            
    
    st.image("PowerBI/E_v_pseudorapidity.png")

    st.write("3) **pz1 has a strong correlation with pseudorapidity while px and py doesn´t.**")
    st.markdown("""<p style='font-size: 16px; text-align: justify'><p style='font-size: 16px; text-align: justify'>The reason of this is due to the fact that pseudorapidity depends on the angle theta. As we see below:</p>""", unsafe_allow_html=True)
    st.latex(r'''\eta = -\ln\left(\tan\left(\frac{\theta}{2}\right)\right)''')
    st.image("PowerBI/pxpypz_v_pseudorapidity.png")


    st.markdown("""<p style='font-size: 16px; text-align: justify'>Theta is the angle between the z-axis and the beam direction.</p>""", unsafe_allow_html=True)
    st.image("pictures/theta_angle.png")
    st.write("In a range from 0 to 90 degrees to theta, the value of pseudorapidity decreases as we increase the value of theta.")
    st.markdown("""<p style='font-size: 16px; text-align: justify'>When pseudorapidity tends to 0 is because theta is 90º (or pi rads), 
                which implies the particle is moving perpendicular to the z-axis. </p>""", unsafe_allow_html=True)


    st.write("4) **Invariant mass and Energy follow a pattern.**")
    st.markdown("""<p style='font-size: 16px; text-align: justify'>It seems the Total Energy can´t be lower than the Invariant Mass.
            According to the principles of special relativity E >= M. <br>
            Further analysis indicates that when the invariant mass is high and the charges of the colliding particles differ 
            (i.e., one is positively charged and the other is negatively charged), it is more probable that the outcomes of such experiments will yield particles 
            of different charges. </p>""", unsafe_allow_html=True)
    st.image("PowerBI/TotalEnergy_v_m.png")

             
def data_science():
    st.title("Data Science")

    st.markdown("""
    ## Introduction to Our Data Science Approach
    <p style='font-size: 16px; text-align: justify'>
    In the initial phase of our project, we aimed to predict the invariant mass (M) of two colliding particles based on several parameters, specifically the total energy (E), 
    as well as the momentum components (px), (py), and (pz).  
    However, upon further investigation, we discovered that there are existing equations capable of calculating the theoretical value of (M) with greater accuracy than any prediction model.  
    <p style='font-size: 16px; text-align: justify'>
    Additionally, our initial approach faced another challenge: the requirement for experimental data from a collision to calculate (M). 
    If the experiment had already been conducted, the necessity of employing machine learning for this calculation was called into question.  
    <p style='font-size: 16px; text-align: justify'>
    Recognizing the need for a more innovative approach, we redefined our objective to predict the invariant mass (M) while minimizing reliance on experimental variables, 
    focusing exclusively on (pz). This decision was motivated by the desire to simplify the prediction model while ensuring its effectiveness. 
    By concentrating solely on (pz), we aimed to enhance the interpretability of our model and potentially uncover new insights into the relationship between momentum and invariant mass. 
    The purpose of this methodology was to explore the possibility of substituting certain experimental processes with machine learning techniques, ultimately saving both time and resources.
    <p style='font-size: 16px; text-align: justify'>
    This streamlined approach not only distinguished our work from prior studies but also established a foundation for a more profound exploration of the underlying physics governing particle collisions.
    </p>""", unsafe_allow_html=True)

    st.markdown("""<h2>Machine Learning Models Perfomance</h2>""", unsafe_allow_html=True)

    st.markdown("""We tried different models of Machine Learning, getting the following results:""", unsafe_allow_html=True)
    
    norm_models = {
            "Model": ["KNN", "Linear Regression", "Decision Tree", "Random Forest", "SVR", 'XGBoost'],
            "MAE": [
                10.4085,
                19.2581,
                12.6343, 
                9.6463,
                17.6996,
                9.1662
            ],
            "MSE" : [
                16.5322,
                24.8982,
                21.4528,
                15.7044,
                26.0569,
                14.9891

            ],
            "R2" : [
                0.5712,
                0.0274,
                0.2780,
                0.6131,
                -0.0652,
                0.6475
            ]
        } 
    
    standarized_models = pd.DataFrame(norm_models)
    formatted_df_norm = standarized_models.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    standarized_models = {
            "Model": ["KNN", "Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost"],
            "MAE": [
                10.3297,
                19.2581,
                12.6156, 
                9.6485,
                19.9789,
                9.1786
            ],
            "MSE" : [
                16.4559,
                24.8983,
                21.4225,
                15.7095,
                19.9789,
                14.9921

            ],
            "R2" : [
                0.5752,
                0.0274,
                0.2800,
                0.6128,
                0.3738,
                0.6474
            ]
        } 
    
    standarized_models = pd.DataFrame(standarized_models)
    formatted_df_standarized = standarized_models.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Normalized Models")
        st.dataframe(formatted_df_norm)

    with col2:
        st.markdown("### Standarized Models")
        st.dataframe(formatted_df_standarized)
    

    st.markdown("""<p style='font-size: 16px; text-align: justify'>As can be observe, the standardized and normalized models yielded similar results for the models with the highest R2 values.</h2>""", unsafe_allow_html=True)

    st.markdown("""<p style='font-size: 16px; text-align: justify'>Based on the data we obtained, we decided to attempt hyperparameter tuning using Random Forest and XGBoost Models.</h2>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        rf_hyperparameters = {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'criterion': ['absolute_error', 'friedman_mse', 'poisson', 'squared_error']
        }

        df_rf_hyper_param = pd.DataFrame({
            'Parámetro': list(rf_hyperparameters.keys()),
            'Valores': [', '.join(map(str, valores)) for valores in rf_hyperparameters.values()]
        })

        if 'show_rf_hyper' not in st.session_state:
            st.session_state.show_rf_hyper = False


        if st.button("Show/Hide Random Forest Hyper-Parameters"):
            st.session_state.show_rf_hyper = not st.session_state.show_rf_hyper  


        if st.session_state.show_rf_hyper:
            st.table(df_rf_hyper_param)

    with col4:
        xgboost_hyperparameters = {
            'n_estimators': [100, 200, 300, 400, 500],           
            'max_depth': [3, 5, 7, 9, 11],                      
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],       
            'subsample': [0.6, 0.8, 1.0],                       
            'colsample_bytree': [0.6, 0.8, 1.0],                
            'gamma': [0, 0.1, 0.2, 0.3],                     
            'reg_alpha': [0, 0.1, 1, 10],                        
            'reg_lambda': [0, 0.1, 1, 10]   
        }

        df_xgboost_hyper_param = pd.DataFrame({
            'Parámetro': list(xgboost_hyperparameters.keys()),
            'Valores': [', '.join(map(str, valores)) for valores in xgboost_hyperparameters.values()]
        })

        if 'show_xgboost_hyper' not in st.session_state:
            st.session_state.show_xgboost_hyper = False


        if st.button("Show/Hide XGBoost Hyper-Parameters"):
            st.session_state.show_xgboost_hyper = not st.session_state.show_xgboost_hyper  


        if st.session_state.show_xgboost_hyper:
            st.table(df_xgboost_hyper_param)


    st.markdown("""
    Using these parameters, the XGBoost model achieved an `R² value of 0.6555` during cross-validation. 
    Given the complexity of the subject matter we are predicting, this result is considered satisfactory. 
    However, we aim to further improve our predictive accuracy. Therefore, we will proceed to explore the use of Neural Networks for this task.
    """)

    st.markdown("""<h2>Neural Networks</h2>""", unsafe_allow_html=True)

    st.markdown("""We utilized TensorFlow to explore the possibility of enhancing the model's performance.
                Prior to being fed into the model, the data underwent standardization and conversion to `float32`. 
                This conversion is essential because float32 operations are computationally faster than those involving `float64`.""")


    st.write("falta")


def conclusions():

    st.title("Machine Learning Conclusions")

    st.markdown("""<p style='font-size: 16px; text-align: justify'>
                Due to the fact that we achieved almost the same results using both XGBoost and TensorFlow, we are opting to choose XGBoost as the best model 
                for now because it is significantly faster. However, it is noteworthy that the best XGBoost hyperparameter configuration we found is slightly 
                worse than the results from the random trials of the Neural Networks. Therefore, we are confident that with more time and optimization, 
                we could achieve even better results.  <br>
                <p style='font-size: 16px; text-align: justify'>
                While an R² value of approximately 0.65 might initially appear suboptimal, it is important to consider that we are contending with the 
                fundamental laws of physics. Achieving a prediction accuracy that can explain around 65% of the variance in such a complex domain is indeed 
                a significant accomplishment. <br>  
                <p style='font-size: 16px; text-align: justify'>
                Given the potential financial and time-saving benefits that an even more accurate model could offer, we are committed to further enhancing 
                our predictive accuracy in future iterations. </p>""", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://jun-makino.sakura.ne.jp/animations/sph/sph_col_rp0.5_nb512.gif" alt="Collision GIF">
        </div>
        """,
        unsafe_allow_html=True
    )   


def experimental_framework():
    import pandas as pd 
    import numpy as np 
    import pickle 
    import streamlit as st 

    st.title("Experimental Framework")

    pickle_in = open("xg_model_randomized_search.pkl", "rb")
    xg_boost_model = pickle.load(pickle_in)

    def prediction(run, pz1, pz2, is_same_charge, is_outlier):   
   
        prediction = xg_boost_model.predict( 
            [[run, pz1, pz2, is_same_charge, is_outlier]]) 
        # print(prediction) 
        return prediction 
    
    df = pd.read_csv("dielectron.csv")

    run_options = df["Run"].unique()


    run = int(st.selectbox("Select the run(s)", run_options))
    pz1 = float(st.text_input("Enter the pz of particle 1", "0"))
    pz2 = float(st.text_input("Enter the pz of particle 2", "0"))

    is_same_charge = st.checkbox("Do they have the same charge?")
    is_outlier = st.checkbox("Is it an outlier?")

    if st.button("Predict"): 
        with st.spinner("Making prediction..."):
            try:
                result = prediction(run, pz1, pz2, is_same_charge, is_outlier) 
                st.success(f'The output is **{result[0]:.4f}**!')

                # Display explosion effect with an image or GIF
                st.image("https://cdn.dribbble.com/users/315053/screenshots/2390908/media/3a5775599ed9f2a352fef39362c0e346.gif", width=300)  # Adjust width as necessary
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        

def main():
    st.sidebar.title("Navegation")
    page = st.sidebar.selectbox("Select a page", ["Introduction", "Data Analysis", "Data Science", "Insights and conclusions", "Experimental Framework"]) 
    st.sidebar.markdown("<br>" * 16, unsafe_allow_html=True)
    st.sidebar.markdown("IronHack Final Project")
    st.sidebar.markdown("""  
                    ## This project has been developed by:
                    Iván Alonso - https://github.com/ivanalonsom  
                    """)

    if page == "Introduction":
        intro()
    elif page == "Data Analysis":
        data_analysis()
    elif page == 'Data Science':
        data_science()
    elif page == 'Insights and conclusions':
        conclusions()
    elif page == 'Experimental Framework':
        experimental_framework()


if __name__ == "__main__":
    main()

