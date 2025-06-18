# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:29:26 2025

@author: blas_fe
"""
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
import cmcrameri.cm as cmc
# plt.rcParams['text.usetex'] = True
from pathlib import Path

######## CONSTANTS ##########
# %%
Rair = 287.052874  # J kg−1 K−1

density = 1.29205  # kg/m3
# Area = 8e-5 #m2
# ViscosityCoefficient1 = 0.00001825 # As defined in the excel sheet
# ViscosityCoefficient2 = 0.00001849 #
# TCoef1 = 20.0 # As defined in the excel sheet
# TCoef2 = 25.0
# DensityCoeff = (101325/1000*28.96)/ (8.31446*273.15) #From excel sheet


# Create Cantera gas to obtain properties at different Temp and P conditions
CanteraAir = ct.Solution('air.yaml')
CanteraAir.TP = 273.15, ct.one_atm
StandardDensity = CanteraAir.density
StandardViscosity = CanteraAir.viscosity
StandardKV = StandardViscosity/StandardDensity

## DATA INPUT AND PREPROCESSING #########
# %%


# Define the folder path where the files are located
base_dir = Path("H:\HEAT")  # Change this to your root folder

# Initialize an empty dictionary to store DataFrames
all_data = {}
Average = {}
StandardDev = {}
ADF = {}
STD_DF = {}
# Loop through all files in the folder
for folder in base_dir.iterdir():
    if folder.is_dir():
        folder_name = folder.name
        all_data[folder_name] = {}
        Average[folder_name] = {}
        StandardDev[folder_name] = {}
        for csv_file in folder.glob("*.txt"):
            file_name = csv_file.stem
            # Read the file and store the DataFrame in the dictionary
            df = pd.read_csv(csv_file, sep='\t', encoding='ISO-8859-1',
                             skiprows=[1, 2], dtype={'date': str, 'time': str})  # adjust encoding/sep as needed
            df['date_time'] = pd.to_datetime(df.pop('date')+' '
                                             + df.pop('time'),
                                             format='%d.%m.%Y %H:%M:%S.%f')
            # df = df.set_index('date_time')
            df.rename(columns={"ai1:_P1_fuel_HEAT": "P1", "ai1:_P1_fuel_HEAT_Stabw": "P1std",
                               "AIR-PIL": "Outer air", "AIR-PIL.set": "Outer air setpoint",
                               "AIRHEAT": "Inner air", "AIRHEAT.set": "Inner air setpoint",
                               'ai12:_HEAT_T1': 'T1 Heat In', 'ai12:_HEAT_T1_Stabw': 'T1 Heat In std',
                               'ai13:_HEAT_T2': 'T2 Core Out', 'ai13:_HEAT_T2_Stabw': 'T2 Core Out std',
                               'ai14:_HEAT_T3': 'T3 Core In', 'ai14:_HEAT_T3_Stabw': 'T3 Core In std',
                               'ai15:_HEAT_T4': 'T4 Heat Out', 'ai15:_HEAT_T4_Stabw': 'T4 Heat Out std',
                               'HEAT_#96:_Volume_[sl/min]': 'Coriolis [l/min]',
                               'HEAT_#96:_mass_[g/min]': 'Coriolis [g/min]',
                               'HEAT_#96:_mass_[g/s]': 'Coriolis [g/s]',
                               'ai12:_HEAT_T1_heat_in': 'T1 Heat In',
                               'ai12:_HEAT_T1_heat_in_Stabw': 'T1 Heat In std',
                               'ai13:_HEAT_T2_core_out': 'T2 Core Out',
                               'ai13:_HEAT_T2_core_out_Stabw': 'T2 Core Out std',
                               'ai14:_HEAT_T3_core_in': 'T3 Core In',
                               'ai14:_HEAT_T3_core_in_Stabw': 'T3 Core In std',
                               'ai15:_HEAT_T4_heat_out': 'T4 Heat Out',
                               'ai15:_HEAT_T4_heat_out_Stabw': 'T4 Heat Out std'
                               }, inplace=True)
            df.drop(columns=['comment', 'ai0:_P0_plenum', "ai0:_P0_plenum_Stabw",
                             '0.0', '0.1', '0.2', '0.3',
                             "ai0:_T0_plenum", "ai0:_T0_plenum_Stabw", "ai0:_g0_137_hood",
                             "ai0:_g0_137_hood_Stabw", "ai1:_T1_main_air", "ai1:_T1_main_air_Stabw",
                             "ai1:_g1_135_bur", "ai1:_g1_135_bur_Stabw", "ai2:_T2_pilot_air_1",
                             "ai2:_T2_pilot_air_1_Stabw", "ai2:_g2_137_scr", "ai2:_g2_137_scr_Stabw",
                             "ai3:_T3_pilot_air_2", "ai3:_T3_pilot_air_2_Stabw", "ai3:_g3_135_mfc",
                             "ai3:_g3_135_mfc_Stabw", "ai4:_T4_por_in", "ai4:_T4_por_in_Stabw",
                             "ai6:_T6_shield", "ai6:_T6_shield_Stabw", "CH4-PIL", "CH4-PIL.set",
                             "H2-MAIN", "H2-MAIN.set", "H2-PILO", "H2-PILO.set", "N2",
                             "N2.set", "AIR-MAI", "AIR-MAI.set", 'NH3-MAI', 'NH3-MAI.set',
                             'shield_airshield_air_#96:_Volume_[sl/min]',
                             'shield_airshield_air_#96:_mass_[g/min]',
                             'shield_airshield_air_#96:_mass_[g/s]',
                             'X', 'Y', 'Z'],
                    inplace=True, errors='ignore')
            Average[folder_name][file_name] = df.mean(numeric_only=True)
            StandardDev[folder_name][file_name] = df.std(numeric_only=True)
            df['seconds passed'] = (
                df['date_time']-df['date_time'].iloc[0]).dt.total_seconds()
            # Store the DataFrame in the dictionary, with the filename (without extension) as the key
            all_data[folder_name][file_name] = df

        # Get the dataframe with time averaged values for
        # every probe case in a dictionary. So ADF['probe1'] contains
        # the averages of each test case in a dataframe
        ADF[folder_name] = pd.DataFrame.from_dict(
            Average[folder_name], orient='index')
        STD_DF[folder_name] = pd.DataFrame.from_dict(
            StandardDev[folder_name], orient='index')

        # Extract the number after '_d' in the index and add it as a new column
        ADF[folder_name]['Diameter [m]'] = ADF[folder_name].index.str.extract(
            r'_d(\d+)', expand=False).astype(float)/1e+6


# Look if the Coriolis values are empty, and if so take the flow rate from
# the name case, i.e. if the case is the mf_033, set coriolis FR to 3.3 l/min
# def assign_values_from_index(row):
#     pattern = r'_mf_(\d{3})'
#     # Extract the value from the index name using the pattern
#     match = re.search(pattern, str(row.name))

#     if match:
#         # Extracted value
#         extracted_value = float(match.group(1)) / 10

#         # # Replace NaN with the extracted value if any NaNs are found in the row
#         # row.fillna(extracted_value, inplace=True)
#         for col in row.index:
#             if pd.isna(row[col]):
#                 # Example: customize based on column name
#                 if col == 'Coriolis [l/min]':
#                     row[col] = extracted_value  # Just use the extracted value
#                 elif col == 'Coriolis [g/min]':
#                     row[col] = extracted_value * 1.294  # Double the extracted value for column 'B'
#                 elif col == 'Coriolis [g/s]':
#                     row[col] = extracted_value*1.294/60 # Add 10 to the extracted value for column 'C'
#     return row

# Apply the function row-wise
# ADF = ADF.apply(assign_values_from_index, axis=1)


############# COMPUTATIONS ############
# %%
for key in ADF:
    # Cross sectional area
    ADF[key]['Ac [m2]'] = np.pi*(ADF[key]['Diameter [m]'] / 2)**2
    # Unit change
    ADF[key]['T1 Heat In [K]'] = ADF[key]['T1 Heat In'] + 273.15
    ADF[key]['T2 Core Out [K]'] = ADF[key]['T2 Core Out'] + 273.15
    ADF[key]['T3 Core In [K]'] = ADF[key]['T3 Core In'] + 273.15
    ADF[key]['T4 Heat Out [K]'] = ADF[key]['T4 Heat Out'] + 273.15
    ADF[key]['P1 [Pa]'] = ADF[key]['P1'] * 1e+5

    #
    ADF[key]['T1 Heat In - T4 Heat Out'] = ADF[key]['T1 Heat In'] - \
        ADF[key]['T4 Heat Out']
    ADF[key]['T2 Core Out - T3 Core In'] = ADF[key]['T2 Core Out'] - \
        ADF[key]['T3 Core In']

    # Get the mass flow from the volume flow of the MFC.
    # Assume standard density at the MFC
    ADF[key]['Mass flow MFC [kg/s]'] = ADF[key]['Inner air'] * density / 60 / 1000

    # Compute the density of each case by using Cantera at a given pressure and temperature
    # CHANGE WHEN REAL T and P are KNOWN, instead of inlet values, average between
    # Inlet and Outlet?
    Densities = {}
    Viscosities = {}
    AverageTemps = {}
    AveragePressures = {}
    for index, row in ADF[key].iterrows():
        AverageTemps[index] = (row['T3 Core In']+273.15 +
                               row['T2 Core Out']+273.15) / 2.0
        AveragePressures[index] = (row['P1']*1e+5+2e+5)/2
        CanteraAir.TP = AverageTemps[index], AveragePressures[index]
        Densities[index] = CanteraAir.density
        Viscosities[index] = CanteraAir.viscosity

        CanteraAir.TP = row['T2 Core Out [K]'],  ct.one_atm
        ADF[key].at[index, 'Core Out Viscosity [Pa s]'] = CanteraAir.viscosity
        ADF[key].at[index, 'Core Out Density [kg/m3]'] = CanteraAir.density
        ADF[key].at[index, 'Core Out cp [J/kgK]'] = CanteraAir.cp
        ADF[key].at[index, 'Core Out cv [J/kgK]'] = CanteraAir.cv
        ADF[key].at[index, 'Core Out gamma []'] = ADF[key].at[index,
                                                              'Core Out cp [J/kgK]']/ADF[key].at[index, 'Core Out cv [J/kgK]']

        CanteraAir.TP = row['T3 Core In [K]'], row['P1 [Pa]']
        ADF[key].at[index, 'Core In Viscosity [Pa s]'] = CanteraAir.viscosity
        ADF[key].at[index, 'Core In Density [kg/m3]'] = CanteraAir.density
        ADF[key].at[index, 'Core In cp [J/kgK]'] = CanteraAir.cp
        ADF[key].at[index, 'Core In cv [J/kgK]'] = CanteraAir.cv
        ADF[key].at[index, 'Core In gamma []'] = (ADF[key].at[index, 'Core In cp [J/kgK]']
                                                  / ADF[key].at[index, 'Core In cv [J/kgK]'])

    ADF[key]['AverageTemps'] = pd.DataFrame.from_dict(
        AverageTemps, orient='index')
    ADF[key]['AveragePressures'] = pd.DataFrame.from_dict(
        AveragePressures, orient='index')
    ADF[key]['Densities'] = pd.DataFrame.from_dict(Densities, orient='index')
    ADF[key]['Viscosities'] = pd.DataFrame.from_dict(
        Viscosities, orient='index')
    ADF[key]['Corrected pressure'] = (ADF[key]['P1'] *
                                      (ADF[key]['Densities']/StandardDensity)**0.75 *
                                      (ADF[key]['AverageTemps']/(ADF[key]['T1 Heat In']+273.15))**0.7 *
                                      ((ADF[key]['Viscosities']/ADF[key]['Densities'])/StandardKV)**0.25)

    ADF[key]['Inlet velocity [m/s]'] = (ADF[key]['Mass flow MFC [kg/s]']
                                        / ADF[key]['Ac [m2]']
                                        / ADF[key]['Core In Density [kg/m3]'])

    ADF[key]['Outlet velocity [m/s]'] = (ADF[key]['Mass flow MFC [kg/s]']
                                         / ADF[key]['Ac [m2]']
                                         / ADF[key]['Core Out Density [kg/m3]'])

    ADF[key]['Inlet Mach []'] = (ADF[key]['Inlet velocity [m/s]']/np.sqrt(ADF[key]['Core In gamma []'] *
                                                                          ADF[key]['T3 Core In [K]']*Rair))

    ADF[key]['Outlet Mach []'] = (ADF[key]['Outlet velocity [m/s]']/np.sqrt(ADF[key]['Core Out gamma []'] *
                                                                            ADF[key]['T2 Core Out [K]']*Rair))
## PLOTTING ##
# %%


sc = plt.scatter(data=ADF['001_a45_d1000_w7_l60'],
                 x='Mass flow MFC [kg/s]',
                 y='T1 Heat In - T4 Heat Out',
                 c='T1 Heat In [K]',
                 cmap='cmc.batlow')  

plt.title('Outer flow temperature difference')
plt.xlabel('Flowrate [kg/s]')
plt.ylabel('T Outer Outlet - T Outer Inlet [K]')
plt.grid(True)

# Add colorbar with a title
cbar = plt.colorbar(sc)
cbar.set_label('Outer flow setpoint temperature [K]')  # Title
plt.gcf().set_dpi(300)
plt.show()

# %%


sc = plt.scatter(data=ADF['001_a45_d1000_w7_l60'],
                 x='Mass flow MFC [kg/s]',
                 y='T2 Core Out - T3 Core In',
                 c='T1 Heat In [K]',
                 cmap='cmc.batlow')  # You can choose any colormap


plt.title('Inner flow temperature difference')
plt.xlabel('Flowrate [kg/s]')
plt.ylabel('T Inner Outlet - T Inner Inlet [K]')
plt.grid(True)

# Add colorbar with a title
cbar = plt.colorbar(sc)
cbar.set_label('Outer flow setpoint temperature [K]')  # Title
plt.gcf().set_dpi(300)
plt.show()
# %%
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# First subplot
sc1 = axs[0].scatter(data=ADF['001_a45_d1000_w7_l60'],
                     x='Coriolis [g/s]',
                     y='P1',
                     c='T1 Heat In [K]',
                     cmap=cmc.batlow,
                     vmin=275,
                     vmax=800)

axs[0].set_title('Measured pressured drop at different flow rates')
axs[0].set_ylabel('Pressure drop [bar]')
axs[0].grid(True)

# Second subplot
sc2 = axs[1].scatter(data=ADF['001_a45_d1000_w7_l60'],
                     x='Coriolis [g/s]',
                     y='Corrected pressure',
                     c='T1 Heat In [K]',
                     cmap=cmc.batlow,
                     vmin=275,
                     vmax=800)

axs[1].set_title('Corrected pressure')
axs[1].set_xlabel('Mass flowrate [g/s]')
axs[1].set_ylabel(r'$\Delta P T_r^{0.7}\rho_r^{0.75}\nu_r^{0.25}$ [bar]')
axs[1].grid(True)

# Create one colorbar shared between both subplots
cbar = fig.colorbar(sc2, ax=axs, orientation='vertical',
                    fraction=0.05, pad=0.02)
cbar.set_label('Outer flow setpoint temperature [K]')


fig.set_dpi(600)
plt.show()

# %%

fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Create the colormap normalization
norm = plt.Normalize(vmin=275, vmax=800)
cmap = cmc.batlow  # Ensure cmc.batlow is imported from cmocean or cmcrameri

# First subplot: Inlet Mach
sc1 = axs[0].scatter(data=ADF['001_a45_d1000_w7_l60'], x='Coriolis [g/s]', y='Inlet Mach []',
                     c=ADF['001_a45_d1000_w7_l60']['T1 Heat In [K]'],
                     cmap=cmap, norm=norm)
axs[0].set_title('Mach number Inlet')
axs[0].set_ylabel('Mach')
axs[0].grid(True)

# Second subplot: Outlet Mach
sc2 = axs[1].scatter(data=ADF['001_a45_d1000_w7_l60'], x='Coriolis [g/s]', y='Outlet Mach []',
                     c=ADF['001_a45_d1000_w7_l60']['T1 Heat In [K]'],
                     cmap=cmap, norm=norm)
axs[1].set_title('Mach number Outlet')
axs[1].set_xlabel('Flowrate [g/s]')
axs[1].set_ylabel('Mach')
axs[1].grid(True)

# Shared colorbar
cbar = fig.colorbar(sc2, ax=axs, orientation='vertical', shrink=0.95, pad=0.02)
cbar.set_label('Outer flow setpoint temperature [K]')

# plt.tight_layout()
plt.gcf().set_dpi(600)
plt.show()
# %%
# Plot one probe against the other
# Create a figure and axis
fig, ax = plt.subplots()

sc = plt.scatter(data = ADF['001_a45_d1000_w7_l60'],
                 x = 'Mass flow MFC [kg/s]',
                 y = 'P1',
                 c = 'T1 Heat In [K]',
                 cmap = 'cmc.batlow',
                 marker ='o',
                 label = '001 (Circles)')  

sc = plt.scatter(data = ADF['005_a0_d1000_w0_l60'],
                 x = 'Mass flow MFC [kg/s]',
                 y = 'P1',
                 c = 'T1 Heat In [K]',
                 cmap = 'cmc.batlow',
                 marker ='^',
                 label = '002 (Triangles)') 


# Add labels and title
ax.set_xlabel('Mass flow [kg/s]')
ax.set_ylabel('P1 [bar]')
ax.set_title('Pressure drop for different probes')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# %%

# Checks if the temperature has reached steady state
# by plotting the temperature against time of each case.
# Generates many plots, slow to run and check.
# for key in all_data:
#     for keyy in all_data[key]:
#         all_data[key][keyy].plot(x='seconds passed',y='T1 Heat In', kind='scatter')
#         plt.ylabel('Temp deg C')
#         plt.title(keyy)


################### OBSOLETE CODE #######################################
# Names=[]
# OuterInletTemperatures = []
# FlowRate=[]
# P1Total=[]
# TempDiferenceOuterFlow = []
# TempDiferenceInnerFlow = []
# for key, series in Average.items():
#     Names.append(key)
#     P1Total.append(series["P1"])
#     OuterInletTemperatures.append(series["T1 Heat In"]+273.15)
#     Temp=series["T1 Heat In"]+273.15
#     Pre=series["P1"]
#     TempDiferenceOuterFlow.append(series["T1 Heat In"]-series["T4 Heat Out"])
#     TempDiferenceInnerFlow.append(series["T2 CI"]-series["T3 CO"])
#     if 'Coriolis [l/min]' not in series:
#         pattern = r'_mf_(\d{3})'  # This will match exactly three digits after '_mf_'

#         # Search for the pattern
#         match = re.search(pattern, key)
#         extracted_value = float(match.group(1))/10
#         FlowRate.append(extracted_value)
#         fr=extracted_value
#     else:
#         FlowRate.append(series["Coriolis [l/min]"])
#         fr=series["Coriolis [l/min]"]
#     #plt.scatter(fr,Pre,c=Temp,label=key)

# plt.scatter(FlowRate, P1Total,c=OuterInletTemperatures)
# plt.colorbar(label="Temperature [K]")
# plt.xlabel("Flowrate [l/min]")
# plt.ylabel("Pressure [bar]")
# plt.title("Pressure drop against flowrate")
# #plt.legend()
# plt.grid(True)
# plt.show()


# plt.scatter(FlowRate,TempDiferenceOuterFlow,c=OuterInletTemperatures)
# plt.xlabel("Flowrate [l/min]")
# plt.ylabel("Temperature diference [K]")
# plt.title("Temp dif Outer flow")
# plt.show()

# plt.scatter(FlowRate,TempDiferenceInnerFlow,c=OuterInletTemperatures)
# plt.xlabel("Flowrate [l/min]")
# plt.ylabel("Temperature diference [K]")
# plt.title("Temp dif Inner flow")
# plt.grid(True)
# plt.show()

# Average[os.path.splitext(filename)]['Average'] =
# Now you can access each DataFrame by its filename (without the extension)
# For example:
# print(dfs['A45D1000W07L60_100_300'].head())  # replace with an actual file name

# df["PInlet [Pa]"] = ((DataGlobal["P2 [bar]"]+DataGlobal['P3 [bar]'])/2*1e+5)
# DataGlobal["POutlet [Pa]"] = (DataGlobal["P1 [bar]"]*1e+5)
# # Compute the pressure at the center of the probe, average between inlet and outlet
# DataGlobal['Average P [Pa]'] = (DataGlobal["PInlet [Pa]"] + DataGlobal["POutlet [Pa]"])/2
# # Pressure differential
# DataGlobal['Delta P [Pa]'] = (DataGlobal["POutlet [Pa]"]-DataGlobal["PInlet [Pa]"])
