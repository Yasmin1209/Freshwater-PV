# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:54:10 2022

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
# %%

lat = 52
long = 4
data = pd.ExcelFile('input.xlsx')
data = pd.read_excel(data, 'north_sea', index_col = "date") #ask later to Sara about time
data.index = pd.to_datetime(data.index)

def calculate_irr_comp(data):
    # *import input data file from 'Upload_raw_data' output
    # step 1a
    ### Calculation of Solar Zenith and Azimuth angles
    solar_position = pvlib.solarposition.pyephem(data.index, lat, long)

    # step 1b
    ### Calcultation of angle of incidence
    # Due to floater design, 1/2 of the panels will be facing one direction and the other half will be facing 180 degrees
    # step 1c
    ghi = data["allsky_ghi"]  # in W/m^2
    # dirint DNI in W/m^2
    dni = pvlib.irradiance.dirint(ghi, solar_position.zenith, data.index, temp_dew=data.dew_p)
    dni = dni.fillna(0)
    # step 1d
    solar_position.zenith = np.radians(solar_position.zenith)  # converting the angle to radians to calcualte DHI
    dhi = ghi - dni * np.cos(solar_position.zenith)
    solar_position.zenith = np.degrees(solar_position.zenith)  # in degrees again

    # step 1e
    # this function already considers ground reflection. GTI is in W/m^2
    gti = {}
    poa_diffuse = {}
    poa_direct = {}
    aoi = {}

    for orientation in [90, 270]:
        gti[orientation] = pd.DataFrame()
        poa_diffuse[orientation] = pd.DataFrame()
        poa_direct[orientation] = pd.DataFrame()
        aoi[orientation] = pd.DataFrame()

        for tilt in range(2, 14, 2):
            irradiance = pvlib.irradiance.get_total_irradiance(tilt,orientation,solar_position.zenith,solar_position.azimuth,
                dni,ghi,dhi,albedo=data.allsky_albedo,)
            gti[orientation][tilt] = irradiance.poa_global
            poa_diffuse[orientation][tilt] = irradiance.poa_diffuse
            poa_direct[orientation][tilt] = irradiance.poa_direct
            aoi[orientation][tilt] = pvlib.irradiance.aoi(tilt, orientation, solar_position.zenith, solar_position.azimuth)
    return (gti, poa_direct, poa_diffuse, aoi, solar_position, dni, dhi, ghi)

(gti, poa_direct, poa_diffuse, aoi, solar_position, dni, dhi, ghi,) = calculate_irr_comp(data)

def get_best_tilt(gti, poa_diffuse, poa_direct, aoi):
    # plot poa to identify best tilt angle
    data1 = gti[90].sum() / 1000
    data2 = gti[270].sum() / 1000
    best_gti = {}
    best_poa_diffuse = {}
    best_poa_direct = {}
    best_aoi = {}
    add_gti = data1 + data2
    for orientation in gti:

        best_tilt = add_gti.idxmax()
        best_gti[orientation] = gti[orientation][best_tilt]
        best_poa_diffuse[orientation] = poa_diffuse[orientation][best_tilt]
        best_poa_direct[orientation] = poa_direct[orientation][best_tilt]
        best_aoi[orientation] = aoi[orientation][best_tilt]
    return best_tilt, best_gti, best_poa_diffuse, best_poa_direct, best_aoi

(best_tilt, best_gti, best_poa_diffuse, best_poa_direct, best_aoi,) = get_best_tilt(gti, poa_diffuse, poa_direct, aoi)

'''plot orientation for different tilt angles'''
east = gti[90].sum() / 1000
east_sum = sum(east)
west = gti[270].sum() / 1000
west_sum = sum(west)

width = 0.3
labels = ("$\delta$", (r"$\delta$" + "+2"),    (r"$\delta$" + "+4"),    (r"$\delta$" + "+6"),    (r"$\delta$" + "+8"),
    (r"$\delta$" + "+10"),
)
plt.xticks(range(len(east)), labels)
plt.title("Irradiance for different tilt angles")
plt.xlabel("Tilt")
plt.ylabel("Annual irradiance (kWh/m²)")
plt.bar(np.arange(len(east)), east, width=width)
plt.bar(np.arange(len(west)) + width, west, width=width)
plt.legend(["Orientation E", "Orientation W"], loc="lower right")
plt.ylim(0, 1600)
#plt.savefig(names + "_opt_tilt")
plt.show()    

# %%
def get_wave_effect_on_tilt(data, gti, poa_diffuse, poa_direct, aoi):

    (best_tilt, best_gti, best_poa_diffuse, best_poa_direct, best_aoi) = get_best_tilt(gti, poa_diffuse, poa_direct, aoi)
    w_height = data.ssh
    w_height = abs(w_height) * 2
    w_f = 3  # hypothetical width of floater in m
    w_steepness_limit = 1 / 7
    max_w_height = 2 * w_f * w_steepness_limit  # in m
    min_w_height = w_f * w_steepness_limit

    condition_max = w_height > max_w_height
    w_height.loc[condition_max] = 0
    condition_min = w_height < min_w_height
    w_height.loc[condition_min] = 0

    delta_tilt = np.arctan(w_height / w_f)
    delta_tilt = np.degrees(delta_tilt)

    delta_tilt_df = pd.DataFrame(
        data=delta_tilt, index=delta_tilt.index
    )  # convert to dataframe

    return delta_tilt_df, best_tilt

(delta_tilt_df, best_tilt,) = get_wave_effect_on_tilt(data, gti, poa_diffuse, poa_direct, aoi)

## Change in tilt (degrees), best_tilt (degrees) --> Final tilt at 5 time steps (degrees)
## Goal: Obtain final tilt at 5 time steps (dictionary with 5 time series), accounting for change in tilt due to wave effect.
## 1. Upload the wave_effect at each time step t:
## It is assumed the hour is divided into 5 time steps t:
##    t=1) The floater is at the delta_tilt facing E
##    t=2) The floater is at best_tilt facing E
##    t=3) The floater is horizontal (angle 0) facing E, W
##    t=4) The floater is at best_tilt facing W
##    t=5) The floater is at the delta_tilt facing W
## 2. For every hour, we calculate the five time steps t
## 3. wave_e is a dictionary and stores the calculations from step 2
## 4. Function outputs the final tilt in a dictionary of 5 time series.
## For simplicity, we assume the orientations remain the same.
## In reality, at one time step t, the orientations change 180 degrees, and at another time step t the orientation is 0.
## This only represents 2/5 of one hour, so it is a good assumption to make that the panels will maintain
## its original orientation (3/5 of the hour)


def get_tilt_function(delta_tilt_df, best_tilt):

    wave_e = {}  # dictionary to store the 5 step angles
    x = delta_tilt_df["ssh"]
    wave_e["step_one"] = x * 0 + best_tilt
    wave_e["step_one"][x > 0] = x + best_tilt
    wave_e["step_two"] = x * 0 + best_tilt
    wave_e["step_two"][x > 0] = (x * 0) + 2 * best_tilt
    wave_e["step_three"] = x * 0 + best_tilt
    wave_e["step_three"][x > 0] = (x * 0) + best_tilt
    wave_e["step_four"] = x * 0 + best_tilt
    wave_e["step_four"][x > 0] = 0 * x
    wave_e["step_five"] = x * 0 + best_tilt
    wave_e["step_five"][x > 0] = abs(x - best_tilt)

    return wave_e
wave_e = get_tilt_function(delta_tilt_df, best_tilt)

tilt = wave_e
one = tilt["step_one"].mean()
two = tilt["step_two"].mean()
three = tilt["step_three"].mean()
four = tilt["step_four"].mean()
five = tilt["step_five"].mean()

tilt_avg_NS = (one + two + three + four + five) / 5
## final_tilt (degrees), orientation (degrees), solar zenith angle (degrees),
## solar azimuth (degrees), DNI (W/m^2), GHI (W/m^2), DHI (W/m^2), albedo (NA)  --> final GTI (W/m^2), final POA diffuse (W/m^2),
## final POA direct (W/m^2), final angle of incidence (degrees)
## Goal: Obtain final tilt (time series), accounting for change in tilt due to wave effect.
## 1. Create dictionaries to store for loop outputs
## 2. For both orientations, pvlib.irradiance.get_total_irradiance and pvlib.irradiance.aoi
## 3. Store results in dictionaries


def correct_irr_components_wave_e(
    data, delta_tilt_df, best_tilt, solar_position, dni, dhi, ghi):
    wave_e = get_tilt_function(delta_tilt_df, best_tilt)
    gti_final = {}
    poa_diffuse_final = {}
    poa_direct_final = {}
    irradiance = {}
    aoi_final = {}
    for orientation in [90, 270]:
        gti_final[orientation] = pd.DataFrame()
        poa_diffuse_final[orientation] = pd.DataFrame()
        poa_direct_final[orientation] = pd.DataFrame()
        aoi_final[orientation] = pd.DataFrame()
        irradiance[orientation] = 0
        aoi_final[orientation] = 0

        for key in wave_e:
            irradiance[orientation] += 0.2 * pvlib.irradiance.get_total_irradiance(wave_e[key], orientation,
                solar_position.zenith, solar_position.azimuth, dni, ghi, dhi, albedo=data.allsky_albedo, )
            aoi_final[orientation] += 0.2 * pvlib.irradiance.aoi(
                wave_e[key], orientation, solar_position.zenith, solar_position.azimuth            )
        gti_final[orientation] = irradiance[orientation].poa_global
        poa_diffuse_final[orientation] = irradiance[orientation].poa_diffuse
        poa_direct_final[orientation] = irradiance[orientation].poa_direct
    return gti_final, poa_diffuse_final, poa_direct_final, aoi_final, wave_e


(gti_final, poa_diffuse_final, poa_direct_final, aoi_final, wave_e) = correct_irr_components_wave_e(
    data, delta_tilt_df, best_tilt, solar_position, dni, dhi, ghi)


#ask later how to improve this notation
# %%
def calculate_cell_temp(data, gti_final):

    # first calculate heat index

    temp_air_F = 9 * data.temp_air / 5 + 32  # convert degrees C to fahrenheit
    rel_humid = data.rel_humid

    # equation for heat index calc:
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = -42.2, 2.05, 10.14, -0.22, -6.84*10**-3, -5.48*10**-2, 1.23*10**-3, 8.531*10**-4, -1.9910*10**-6
    heat_index_new = a0 + a1*temp_air_F + a2*rel_humid + a3*temp_air_F + a4*temp_air_F**2 + a5*rel_humid**2+a6*temp_air_F**2*rel_humid+a7*temp_air_F*rel_humid**2+a8*temp_air_F**2*rel_humid**2 #very high value
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
    module = pvlib.pvsystem.retrieve_sam("SandiaMod").SunPower_SPR_315E_WHT__2007__E__
    #calculate cell temperature with new method
    #Upv_new = gti_final/1500 + 2


    # next step is to calculate the cell temperature using Mattei 1 model
    Upv = 26.6 + 2.3 * data.w_speed
    m_eff = 0.193
    Tstc = 25
    gamma_r = -0.0038  # in %/C
    temp_cell = (Upv * heat_index + gti_final * (0.81 - m_eff * (1 - gamma_r * Tstc))
    ) / (Upv + gamma_r * m_eff * gti_final)
    U0, U1 = 30.02, 6.28
    temp_cell_new = Tstc + gti_final/((U0) + U1*data.w_speed)
    # heat transfer between PV and fluid (water for offshore and air for on land)
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

    return (heat_index, heat_index_new, temp_cell, temp_eq, module, temp_pv)

'''figure Temperature'''


# %%
'''Yield one panel'''
def calculate_yield_one_panel(gti_final, poa_diffuse_final, poa_direct_final, aoi_final, solar_position, data):
    (heat_index, heat_index_new, temp_cell, temp_eq, module, temp_pv) = calculate_cell_temp(data, gti_final)
    airmass_relative = pvlib.atmosphere.get_relative_airmass(solar_position.zenith)
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass_relative)
    eff_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_direct_final, poa_diffuse_final, airmass_absolute, aoi_final, module,)
    sapm = pvlib.pvsystem.sapm(eff_irradiance, temp_eq, module)
    p_mp = pd.Series(sapm.p_mp)
    return (heat_index, heat_index_new, temp_cell, temp_eq, sapm, p_mp, eff_irradiance, temp_pv, module)

temp_celll = []
temp_eql = []
sapml = []
DCl = []
eff_irradiancel = []
ac_outputl = []

    # calculate yield for one panel in both orientations

for orientation in [90, 270]:
    (heat_index, heat_index_new, temp_cell, temp_eq, sapm, p_mp, eff_irradiance, temp_pv,module,) = calculate_yield_one_panel(
         gti_final[orientation], poa_diffuse_final[orientation], poa_direct_final[orientation],          aoi_final[orientation], solar_position,data)
    temp_celll.append(temp_cell)
    temp_cell_final_avg = pd.concat(temp_celll, axis=1).mean(axis=1)
    temp_cell_final_avg = temp_cell_final_avg.rename("cell_T")
    temp_eql.append(temp_eq)
    temp_eq_final_avg = pd.concat(temp_eql, axis=1).mean(axis=1)
    temp_eq_final_avg = temp_eq_final_avg.rename("equil_T")
    sapml.append(sapm)
    DCl.append(p_mp)
    p_mp_final_avg = pd.concat(DCl, axis=1).mean(axis=1)
    eff_irradiancel.append(eff_irradiance)
    eff_irradiance_final_avg = pd.concat(eff_irradiancel, axis=1).mean(axis=1)


dry_bulb_air_T = temp_pv["dry_bulb_air_T"].resample("M").mean()
heat_index_T = temp_pv["heat_index"].resample("M").mean()
cell_T = temp_cell_final_avg.resample("M").mean()
equil_T = temp_eq_final_avg.resample("M").mean()

'''plot temperature'''
seawater_T = temp_pv["seawater_T"].resample("M").mean()
months = ["Jan", "Feb","Mar", "Apr","May", "Jun", "Jul","Aug","Sep","Oct", "Nov", "Dec"]
temp_pv_plot = pd.DataFrame([dry_bulb_air_T, heat_index_T, cell_T, equil_T, seawater_T])
temp_pv_plot = temp_pv_plot.T
temp_pv_plot.index = months
sns.set_style("darkgrid", {"axes.facecolor": ".85"})  # else: whitegrid, dark, white, ticks
plt.grid(True)
temp = sns.lineplot(data=temp_pv_plot, markers=True)
temp.set(xlabel="Time", ylabel="Temperature (Celsius)")
plt.legend(["DB Air T.", "Heat Index", "Cell T.", "Equilibrium T.", "Sea water T."])
plt.ylim(0, 35)
plt.figure(figsize=(20, 14))
plt.show()
#    ac_outputl.append(ac_output)

DCoutput = sapm.p_mp
DCfinal_avg = pd.concat(DCl, axis=1).mean(axis=1)

# %%
'''get AC output'''
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


'''Calculate and plot AC_output'''
ac_output = 0.86*DCfinal_avg.apply(get_ac_from_dc, nominal_power_ac=315, efficiency_nom=0.96) #why p_mp final average instead of just p_mp?
ac_output_monthly = ac_output.resample("M").sum()
ac_output_monthly.index = months
ac_output_plot = ac_output_monthly
plt.grid(True)
temp1 = sns.lineplot(data=ac_output_plot, markers=True)
temp1.set(xlabel="Time", ylabel="Energy yield of one panel (Wh)")
plt.legend(["north-sea"])
plt.ylim(0, 62000)
plt.figure(figsize=(20, 14))
plt.show()

# %%
def get_performance_ind(ac_output, eff_irradiance_final_avg):

    g_stc = 1000  # in W/m^2
    Pp = 315  # nominal power of one panel in Wp
    gti_sum = sum(eff_irradiance_final_avg)  # in Wh/m^2
    annual_yield = sum(ac_output)  # in Wh why do you take the sum of the ac_output instead of ac_output_final_avg?
    y_f = annual_yield / Pp
    y_r = gti_sum / g_stc
    PR = y_f / y_r * 100
    CF = annual_yield / (Pp * 365 * 24) * 100
    return PR, CF, gti_sum, annual_yield

PR, CF, gti_sum, annual_yield = get_performance_ind(ac_output, eff_irradiance_final_avg)
#Upv_new = gti_final/1500 + 2
for orientation in [90,270]:
    U0, U1 = 30.02, 6.28
    Tstc = 25
    temp_cell_t = Tstc + gti_final[orientation]/((U0) + U1*data.w_speed)
    sumtemp = sum(temp_cell_t)