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
    as well as the momentum components (p_x), (p_y), and (p_z).  
    However, upon further investigation, we discovered that there are existing equations capable of calculating the theoretical value of (M) with greater accuracy than any prediction model.  
    <p style='font-size: 16px; text-align: justify'>
    Additionally, our initial approach faced another challenge: the requirement for experimental data from a collision to calculate (M). 
    If the experiment had already been conducted, the necessity of employing machine learning for this calculation was called into question.  
    <p style='font-size: 16px; text-align: justify'>
    Recognizing the need for a more innovative approach, we redefined our objective to predict the invariant mass (M) while minimizing reliance on experimental variables, 
    focusing exclusively on (p_z). This decision was motivated by the desire to simplify the prediction model while ensuring its effectiveness. 
    By concentrating solely on (p_z), we aimed to enhance the interpretability of our model and potentially uncover new insights into the relationship between momentum and invariant mass. 
    The purpose of this methodology was to explore the possibility of substituting certain experimental processes with machine learning techniques, ultimately saving both time and resources.
    <p style='font-size: 16px; text-align: justify'>
    This streamlined approach not only distinguished our work from prior studies but also established a foundation for a more profound exploration of the underlying physics governing particle collisions.
    </p>""", unsafe_allow_html=True)

    st.markdown("""<h2>Machine Learning Models</h2>""", unsafe_allow_html=True)










# def deep_learning():

#     st.write("RF:")

#     code = """
#         df_total_predict_202021 = df_total_values_PCA.copy()

#     future_years = [2020, 2021]

#     for year in future_years:

#     #Drop PCA var
#     df_total_predict_202021 = df_total_predict_202021.drop(columns=['Values_year_PCA'], errors='ignore')

#     #Recalculate year_minus
#     df_total_predict_202021 = get_previous_years(df_total_predict_202021)

#     #Apply PCA
#     df_total_predict_202021 = apply_pca(df_total_predict_202021, columns_pca)

#     #Predict next year
#     X_year = df_total_predict_202021[df_total_predict_202021['year'] == year - 1][features_nn]
#     y_pred_year = best_rf_model.predict(X_year)

#     new_year_df = df_total_predict_202021[df_total_predict_202021['year'] == year - 1].copy()
#     new_year_df['year'] = year
#     new_year_df['value'] = y_pred_year

#     df_total_predict_202021 = pd.concat([df_total_predict_202021, new_year_df], ignore_index=True)

#     y_year = df_total_values[df_total_values['year'] == year]['value']
#     """

#     st.code(code, language='python')


#     if 'show_df' not in st.session_state:
#         st.session_state.show_df = True

#     if st.session_state.show_df:
#         df_predicted = pd.read_csv('data/df_201321_with_202021_predicted.csv')
#         st.write("The dataframes after the treatment and prediction has been performed:")
#         st.write(df_predicted)


# def conclusions():
#     st.write("The obtained score for Random Forest is:")
#     st.code("Mean Absolute Error (MAE): 1033.0388923532064 \nRoot Mean Squared Error (RMSE): 4707.907926358289 \nR² Score: 0.9896681902859178")

#     st.write("Real data from INE vs Calculated and predicted data:")
#     st.image("data/real_v_pred.jpg", use_column_width=True)

    




st.sidebar.title("Navegation")
page = st.sidebar.selectbox("Select a page", ["Introduction", "Data Analysis", "Data Science", "Insights and conclusions"]) 
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
# elif page == 'Insights and conclusions':
#     conclusions()




