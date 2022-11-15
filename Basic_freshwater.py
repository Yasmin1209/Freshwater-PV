# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:42:31 2022

@author: yobbi
"""


import pandas as pd
import pvlib
import numpy as np
import linecache
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

@st.cache
def data():
    data = pd.ExcelFile('Input.xlsx')
    data = pd.read_excel(data, 'north_sea', index_col = "date") #ask later to Sara about time
    data.index = pd.to_datetime(data.index)

    return data
    
data = data()


lat = 52
long = 5
tilt = 15
orientation = 90

solar_position = pvlib.solarposition.pyephem(data.index, lat, long)

'''get irradiance data'''
ghi = data["allsky_ghi"]  # in W/m^2
    # dirint DNI in W/m^2
dni = pvlib.irradiance.dirint(ghi, solar_position.zenith, data.index, temp_dew=data.dew_p)
dni = dni.fillna(0)
# step 1d
solar_position.zenith = np.radians(solar_position.zenith)  # converting the angle to radians to calcualte DHI
dhi = ghi - dni * np.cos(solar_position.zenith)
solar_position.zenith = np.degrees(solar_position.zenith)  # in degrees again
irradiance = pvlib.irradiance.get_total_irradiance(tilt,orientation,solar_position.zenith,solar_position.azimuth,
dni,ghi,dhi,albedo=data.allsky_albedo,)
gti = irradiance.poa_global
poa_diffuse = irradiance.poa_diffuse
poa_direct = irradiance.poa_direct
aoi = pvlib.irradiance.aoi(tilt, orientation, solar_position.zenith, solar_position.azimuth)

'''Calculation temperature''' #change later
def calculate_cell_temp(data, gti_final):

    temp_air_F = 9 * data.temp_air / 5 + 32  # convert degrees C to fahrenheit
    rel_humid = data.rel_humid

    # equation for heat index calc:
#    a0, a1, a2, a3, a4, a5, a6, a7, a8 = -42.2, 2.05, 10.14, -0.22, -6.84*10**-3, -5.48*10**-2, 1.23*10**-3, 8.531*10**-4, -1.9910*10**-6
#    heat_index_new = a0 + a1*temp_air_F + a2*rel_humid + a3*temp_air_F + a4*temp_air_F**2 + a5*rel_humid**2+a6*temp_air_F**2*rel_humid+a7*temp_air_F*rel_humid**2+a8*temp_air_F**2*rel_humid**2 #very high value
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 * (10 ** (-3))
    c6 = -5.481717 * (10 ** (-2))
    c7 = 1.22874 * (10 ** (-3))
    c8 = 8.5282 * (10 ** (-4))
    c9 = -1.99 * (10 ** (-6))
    heat_index = (
        c1
        + (c2 * temp_air_F)
        + (c3 * rel_humid)
        + (c4 * temp_air_F * rel_humid)
        + (c5 * (temp_air_F ** 2))
        + (c6 * (rel_humid ** 2))
        + (c7 * (temp_air_F ** 2) * rel_humid)
        + (c8 * temp_air_F * (rel_humid ** 2))
        + (c9 * (temp_air_F ** 2) * (rel_humid ** 2))
    )
    # this condition is part of the function of heat index.
    condition = (temp_air_F <= 80) | (rel_humid <= 40)
    heat_index.loc[condition] = temp_air_F[condition]
    heat_index = (heat_index - 32) * 5 / 9  # convert back to degrees C

    # upload module specification sheet
    #calculate cell temperature with new method
#    Upv_new = gti_final/1500 + 2


    # next step is to calculate the cell temperature using Mattei 1 model
    Upv = 26.6 + 2.3 * data.w_speed
    m_eff = 0.193
    Tstc = 25
    gamma_r = -0.0038  # in %/C
    temp_cell = (Upv * heat_index + gti_final * (0.81 - m_eff * (1 - gamma_r * Tstc))
    ) / (Upv + gamma_r * m_eff * gti_final)    # heat transfer between PV and fluid (water for offshore and air for on land)
    # Solve for equilibrium temperature (temp_eq)
    # Fluid and material properties
    m_p = 8050  # for now this is for one cubic meter of steel
    c_p = 490  # in J/K
    properties_p = m_p * c_p
    m_f = 1025  # for now this is for one cubic meter of water
    c_f = 4200  # in J/K
    properties_f = m_f * c_f
    temp_f = data.sst
    temp_e = temp_cell * properties_p
    temp_ee = temp_f.multiply(properties_f)
    temp_eq = (temp_e + temp_ee) / (properties_f + properties_p)
    temp_pv = pd.concat([data.temp_air, heat_index, temp_cell, temp_eq, data.sst], axis=1,)
    temp_pv = temp_pv.rename(columns={"temp_air": "dry_bulb_air_T", 0: "heat_index", 1: "cell_T", 2: "equil_T",
            "sst": "seawater_T",        }    )
    temp_pv.index = data.index
#    temp_pv.to_csv(seaname + "_cell_temp.csv")

    return (heat_index, temp_cell, temp_eq, module, temp_pv)


'''Energy yield calculations'''
module = pvlib.pvsystem.retrieve_sam("SandiaMod").SunPower_SPR_315E_WHT__2007__E__
airmass_relative = pvlib.atmosphere.get_relative_airmass(solar_position.zenith)
airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass_relative)
eff_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi, module)
sapm = pvlib.pvsystem.sapm(eff_irradiance, data.temp_air, module) #adjust temperature
DCoutput = pd.Series(sapm.p_mp)

def get_ac_from_dc(power_dc, *, nominal_power_ac, efficiency_nom):

    # a) Return 0 if the DC power is 0
    if power_dc == 0:
        return 0
    # Calculate the rated DC power
    nominal_power_dc = nominal_power_ac / efficiency_nom

    # b) Return the rated power of the inverter if the DC power is larger or equal to the rated DC power
    if power_dc >= nominal_power_dc:
        return nominal_power_ac
    # Calculate the efficiency
    zeta = power_dc / nominal_power_dc
    efficiency = -0.0162 * zeta - (0.0059 / zeta) + 0.9858

    return efficiency * power_dc

ac_output = DCoutput.apply(get_ac_from_dc, nominal_power_ac=315, efficiency_nom=0.96) #why p_mp final average instead of just p_mp?
'''
def get_performance_ind(ac_output, eff_irradiance_final_avg):

    g_stc = 1000  # in W/m^2
    Pp = 315  # nominal power of one panel in Wp
    gti_sum = sum(eff_irradiance_final_avg)  # in Wh/m^2
    annual_yield = sum(ac_output)*0.86  # in Wh why do you take the sum of the ac_output instead of ac_output_final_avg?
    y_f = annual_yield / Pp
    y_r = gti_sum / g_stc
    PR = y_f / y_r * 100
    CF = annual_yield / (Pp * 365 * 24) * 100
    return PR, CF, gti_sum, annual_yield



PR, CF, gti_sum, annual_yield = get_performance_ind(ac_output, eff_irradiance)
Per_ind = [PR, CF, annual_yield]
Parameters = ["PR (%)", "CF (%)", "Total annual yield (MWh)", ]
results = {"parameters": ["PR (%)", "CF (%)", "Total annual yield (MWh)",
        ],"results": [PR, CF, annual_yield],}
results = pd.DataFrame(Per_ind)
results.index = Parameters

results = {"parameters": ["Performance_ratio_%","Capacity_factor_%",    
            "Annual_yield_MWh"    ], "results": [PR, CF, annual_yield],    }
results = pd.DataFrame(results)
results.index = results.parameters
results = results["results"]
'''
#test = tabulate(results, headers='keys', tablefmt='github', showindex=False)

def tot_an_yield(ac_efficiency, number_modules, eff_irradiance_final_avg):
    g_stc = 1000  # in W/m^2
    Pp = 315  # nominal power of one panel in Wp    
    tot_ac = ac_output.resample("M").mean()*ac_efficiency  
    tot_yield = ac_efficiency*number_modules*tot_ac
    gti_sum2 = sum(eff_irradiance_final_avg)
    annual_yield2 = sum(ac_output)*ac_efficiency
    y_f = annual_yield2 / Pp
    y_r = gti_sum2 / g_stc
    PR2 = y_f / y_r * 100
    CF2 = annual_yield2 / (Pp * 365 * 24) * 100
    return PR2,CF2,tot_ac, tot_yield

#def new_yield(ac_efficiency):
#    new_yield = ac_efficiency*
AC_plot = ac_output.resample("M").mean()
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("solar_basic")
    st.bar_chart(AC_plot)


with dataset:
    st.header("solar_radiation")
    st.text('th')
    st.write(data.head())

    st.bar_chart(AC_plot)
#    st.table(results)
    
with model_training:
    st.header("sensitivity analysis") 
    sel_col, disp_col = st.columns(2)
    number_of_modules = sel_col.slider("Number of panels", min_value = 1, max_value = 10, value = 5, step = 1)
    ac_efficiency = sel_col.slider("AC efficiency", min_value = 0.6, max_value = 1.0, value = 0.86, step = 0.01)
    disp_col.subheader("Performance indicators of system are")
    (PR2,CF2,tot_ac, tot_yield) = tot_an_yield(ac_efficiency, 1, eff_irradiance)
    Per_ind2 = [PR2, CF2, tot_yield]
    Parameters = ["PR (%)", "CF (%)", "Total annual yield (MWh)", ]
    results = {"parameters": ["Performance_ratio_%","Capacity_factor_%",    
            "Annual_yield_MWh"    ], "results": [PR2, CF2, tot_yield],    }
    results = pd.DataFrame(results)
    results.index = results.parameters
    results = results["results"]
    disp_col.write(results)
#    st.dataframe(results)
    st.header("Monthly AC power output")
#    tot_ac, tot_yield = tot_an_yield(ac_efficiency,number_of_modules) 
    st.line_chart(tot_ac)
    

    