import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- TVD Calculation Function ---
def calculate_tvd(data, rkb=0):
    data['TVD'] = 0.0 + rkb  # Initialize TVD column
    for i in range(1, len(data)):
        md1, md2 = data.loc[i - 1, 'MD'], data.loc[i, 'MD']
        inc1, inc2 = np.radians(data.loc[i - 1, 'Inclination']), np.radians(data.loc[i, 'Inclination'])
        az1, az2 = np.radians(data.loc[i - 1, 'Azimuth']), np.radians(data.loc[i, 'Azimuth'])
        
        delta_md = md2 - md1
        cos_dogleg = np.cos(inc2) * np.cos(inc1) + np.sin(inc2) * np.sin(inc1) * np.cos(az2 - az1)
        dogleg = np.arccos(np.clip(cos_dogleg, -1, 1))  # Clip to avoid numerical issues
        
        if dogleg > 1e-6:  # Avoid division by zero
            rf = (2 / dogleg) * np.tan(dogleg / 2)
        else:
            rf = 1.0
        
        delta_tvd = (delta_md / 2) * (np.cos(inc1) + np.cos(inc2)) * rf
        data.loc[i, 'TVD'] = data.loc[i - 1, 'TVD'] + delta_tvd
    return data

# --- Function to Convert DataFrame to Excel for Download ---
@st.cache_data
def convert_df(df):
    # Create a BytesIO object to hold the Excel file
    output = io.BytesIO()
    # Use Pandas ExcelWriter to write the data to the BytesIO stream
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Survey Data")
    # Seek to the beginning of the BytesIO stream before returning
    output.seek(0)
    return output

# --- Streamlit App ---
st.title("Directional Survey & 3D Trajectory Viewer")

# Upload survey file
survey_file = st.file_uploader("Upload your directional survey data (Excel)", type=['xlsx'])

if survey_file:
    # Read uploaded file
    survey_data = pd.read_excel(survey_file)
    st.subheader("Uploaded Data")
    st.write(survey_data)
    
    # Let user choose the column containing Well Names
    survey_columns = list(survey_data.columns)
    well_name_col = st.sidebar.selectbox("Select the Well Name Column", survey_columns)
    
    well_names = survey_data[well_name_col].unique()
    
    if isinstance(well_names[0], str):
        st.sidebar.success("You chose the correct well name column.")
        
        # Multi-select wells
        selected_wells = st.sidebar.multiselect("Select Well(s) to Plot", well_names)
        
        # Input RKB value
        rkb = st.sidebar.number_input("Enter RKB (ft)", value=0.0, step=0.1)
        
        #add note to user about headlines
        st.sidebar.info(
            "📌 Make sure your Excel file includes the following columns with exact names:\n\n"
            "- `MD`\n"
            "- `Inclination`\n"
            "- `Azimuth`\n"
            "- `X_Offset_EW`\n"
            "- `Y_Offset_NS`\n"
            "- `Well Name`\n\n"
            "These are case-sensitive and required for correct TVD and trajectory calculation."
        )
        
        # Initialize 3D figure
        fig = go.Figure()

        # Create an empty dataframe to store data for all wells
        all_wells_data = pd.DataFrame()

        # Process and plot each selected well
        for well in selected_wells:
            df_well = survey_data[survey_data[well_name_col] == well].reset_index(drop=True)
            
            # Check necessary columns exist
            required_cols = {'MD', 'Inclination', 'Azimuth', 'X_Offset_EW', 'Y_Offset_NS'}
            if not required_cols.issubset(df_well.columns):
                st.error(f"Missing columns in uploaded file. Required: {required_cols}")
                break
            
            # Calculate TVD
            df_well = calculate_tvd(df_well, rkb)
            
            # Add trace to 3D plot
            fig.add_trace(go.Scatter3d(
                x=df_well['X_Offset_EW'],
                y=df_well['Y_Offset_NS'],
                z=df_well['TVD'],
                mode='lines',
                name=well,
                line=dict(width=4)
            ))

            # Append the well data to all_wells_data
            all_wells_data = pd.concat([all_wells_data, df_well], ignore_index=True)

        # Update figure layout
        fig.update_layout(
            scene=dict(
                xaxis_title='E-W Displacement',
                yaxis_title='N-S Displacement',
                zaxis_title='TVD',
                zaxis=dict(autorange='reversed')  # TVD increases downward
            ),
            title=dict(
                text="3D Trajectories of Selected Wells",
                font=dict(size=20),
                x=0.5,
                xanchor="center"
            ),
            margin=dict(l=10, r=10, b=10, t=60),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=1)
        )

        # Display the 3D plot
        st.plotly_chart(fig, use_container_width=True)

        # Show the DataFrame with Calculated TVD for all selected wells
        st.subheader("Survey Data with Calculated TVD for Selected Wells")
        st.write(all_wells_data)

        # Provide the option to download the data
        excel_data = convert_df(all_wells_data)
        st.download_button(
            label="Download Calculated TVD Data",
            data=excel_data,
            file_name='survey_with_tvd.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

else:
    st.info("Please upload an Excel file to proceed.")
