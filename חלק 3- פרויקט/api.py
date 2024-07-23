#!/usr/bin/env python
# coding: utf-8

# ## Last Part- Data Project
# #### Submitted By:
# - **Niv Harel**: 208665869
# - **Eytan Muzafi**: 209160308
# 
# #### Github: [https://github.com/nivrl/Data_Course_3rd_year.git]

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from car_data_prep import prepare_data

## Original columns:
columns = ['manufactor',
                 'Year',
                 'model',
                 'Hand',
                 'Gear',
                 'capacity_Engine',
                 'Engine_type',
                 'Prev_ownership',
                 'Curr_ownership',
                 'Area',
                 'City',
                 'Pic_num',
                 'Cre_date',
                 'Repub_date',
                 'Description',
                 'Color',
                 'Km',
                 'Test',
                 'Supply_score']

## Column names for the prediction model that we built:
all_columns=['manufactor_אופל',
 'manufactor_אלפא רומיאו',
 'manufactor_ב מ וו',
 'manufactor_דייהטסו',
 'manufactor_הונדה',
 'manufactor_וולבו',
 'manufactor_טויוטה',
 'manufactor_יונדאי',
 'manufactor_לקסוס',
 'manufactor_מזדה',
 'manufactor_מיני',
 'manufactor_מיצובישי',
 'manufactor_מרצדס',
 'manufactor_ניסאן',
 'manufactor_סובארו',
 'manufactor_סוזוקי',
 'manufactor_סיטרואן',
 'manufactor_סקודה',
 'manufactor_פולקסווגן',
 'manufactor_פורד',
 "manufactor_פיג'ו",
 'manufactor_קיה',
 'manufactor_קרייזלר',
 'manufactor_רנו',
 'manufactor_שברולט',
 'model_108',
 'model_120I',
 'model_159',
 'model_2',
 'model_200',
 'model_2008',
 'model_208',
 'model_220',
 'model_25',
 'model_3',
 'model_300C',
 'model_301',
 'model_307CC',
 'model_308',
 'model_316',
 'model_318',
 'model_320',
 'model_325',
 'model_5',
 'model_5008',
 'model_508',
 'model_523',
 'model_525',
 'model_530',
 'model_6',
 'model_A1',
 'model_A3',
 'model_A4',
 'model_A5',
 'model_A6',
 'model_ACCORD',
 'model_ALL ROAD',
 'model_ASX',
 'model_AX',
 'model_B3',
 'model_B4',
 'model_C-CLASS',
 'model_C-CLASS TAXI',
 'model_C-CLASS קופה',
 'model_C-HR',
 'model_C1',
 'model_C3',
 'model_C30',
 'model_C4',
 'model_C5',
 'model_CADDY COMBI',
 'model_CIVIC',
 'model_CLK',
 'model_CX',
 'model_DS3',
 'model_E- CLASS',
 'model_E-CLASS',
 'model_E-CLASS קופה / קבריולט',
 'model_FR-V',
 'model_GS300',
 'model_GT3000',
 'model_I-MIEV',
 'model_I10',
 'model_I20',
 'model_I25',
 'model_I30',
 'model_I30CW',
 'model_I35',
 'model_I40',
 'model_INSIGHT',
 'model_IS250',
 'model_IS300H',
 'model_JAZZ',
 'model_M1',
 'model_ONE',
 'model_Q3',
 'model_R8',
 'model_RC',
 'model_RCZ',
 'model_RS5',
 'model_S-CLASS',
 'model_S3',
 'model_S5',
 'model_S60',
 'model_S7',
 'model_S80',
 'model_SLK',
 'model_SVX',
 'model_SX4',
 'model_SX4 קרוסאובר',
 'model_V- CLASS',
 'model_V40',
 'model_V40 CC',
 'model_X1',
 'model_XCEED',
 'model_XV',
 'model_אאוטבק',
 'model_אאוטלנדר',
 'model_אדם',
 'model_אודסיי',
 'model_אוואו',
 'model_אוונסיס',
 'model_אונסיס',
 'model_אוקטביה',
 'model_אוקטביה RS',
 'model_אוקטביה ספייס',
 'model_אוקטביה קומבי',
 'model_אוריס',
 'model_אורלנדו',
 "model_אטראז'",
 'model_איגניס',
 'model_איוניק',
 'model_אימפלה',
 'model_אימפרזה',
 'model_אינסיגניה',
 'model_אינסייט',
 'model_אלטו',
 'model_אלמרה',
 'model_אלנטרה',
 'model_אלתימה',
 'model_אס-מקס',
 'model_אסטרה',
 'model_אפלנדר',
 'model_אקווינוקס',
 'model_אקורד',
 'model_אקליפס',
 'model_בלנו',
 "model_ג'אז",
 "model_ג'אז הייבריד",
 "model_ג'ולייטה",
 "model_ג'וק JUKE",
 "model_ג'טה",
 'model_ג`אז',
 'model_ג`טה',
 'model_גולף',
 'model_גולף GTI',
 'model_גולף פלוס',
 'model_גלאקסי',
 "model_גראנד, וויאג'ר",
 'model_גראנד, וויאג`ר',
 'model_גרנד סניק',
 'model_גרנדיס',
 'model_האצ`בק',
 'model_וויאג`ר',
 'model_ולוסטר',
 'model_ורסו',
 'model_זאפירה',
 'model_חיפושית',
 'model_חיפושית חדשה',
 'model_טוראן',
 'model_טראקס',
 "model_טרג'ט",
 'model_טריוס',
 'model_יאריס',
 'model_ייטי',
 'model_לאונה',
 'model_לג`נד',
 'model_לנסר',
 'model_לנסר הדור החדש',
 'model_לנסר ספורטבק',
 'model_לקסוס CT200H',
 'model_לקסוס GS300',
 'model_לקסוס IS250',
 'model_לקסוס IS300H',
 'model_לקסוס RC',
 'model_מאליבו',
 'model_מגאן אסטייט / גראנד טור',
 'model_מוסטנג',
 'model_מוקה',
 'model_מוקה X',
 'model_מיטו',
 'model_מיטו / MITO',
 'model_מיקרה',
 'model_מקסימה',
 'model_מריבה',
 'model_נוט',
 'model_נירו',
 'model_נירו EV',
 'model_נירו PHEV',
 'model_סדן',
 'model_סדרה 1',
 'model_סדרה 3',
 'model_סדרה 5',
 'model_סוויפט',
 'model_סוויפט החדשה',
 'model_סול',
 'model_סונטה',
 'model_סוניק',
 'model_סופרב',
 'model_סטוניק',
 'model_סיד',
 'model_סיוויק',
 "model_סיוויק האצ'בק",
 "model_סיוויק האצ'בק החדשה",
 'model_סיוויק הייבריד',
 'model_סיוויק סדאן',
 'model_סיוויק סדאן החדשה',
 'model_סיוויק סטיישן',
 'model_סיטיגו / CITYGO',
 'model_סיריון',
 'model_סלבריטי',
 'model_סלריו',
 'model_סנטרה',
 'model_ספארק',
 'model_ספיה',
 'model_ספייס',
 'model_ספייס סטאר',
 'model_ספלאש',
 'model_סראטו',
 'model_פאביה',
 'model_פאביה ספייס',
 'model_פאסאט',
 'model_פאסאט CC',
 'model_פולו',
 'model_פוקוס',
 'model_פורטה',
 'model_פיאסטה',
 'model_פיקנטו',
 'model_פלואנס',
 'model_פלואנס חשמלי',
 'model_פריוס',
 'model_פרייד',
 'model_פרימרה',
 'model_קאונטרימן',
 'model_קאמרי',
 'model_קאנטרימן',
 'model_קווסט',
 'model_קונקט',
 'model_קופה',
 'model_קופר',
 'model_קורבט',
 'model_קורולה',
 'model_קורסה',
 'model_קורסה החדשה',
 'model_קורסיקה',
 'model_קליאו',
 'model_קליאו אסטייט',
 'model_קליאו דור 4',
 'model_קפצ`ור',
 'model_קרוז',
 'model_קרוז החדשה',
 'model_קרוסאובר',
 'model_קרניבל',
 'model_ראפיד',
 'model_ראפיד ספייסבק',
 'model_רומסטר',
 'model_ריו',
 'model_שירוקו',
 'model_שרמנט',
 'Gear_Manual',
 'Gear_Robotic',
 'Gear_Tiptronic',
 'Engine_type_Electric',
 'Engine_type_Gas',
 'Engine_type_Hybrid',
 'Engine_type_Petrol',
 'Engine_type_Turbo Diesel',
 'Engine_type_nan',
 'Prev_ownership_השכרה',
 'Prev_ownership_חברה',
 'Prev_ownership_ליסינג',
 'Prev_ownership_מונית',
 'Prev_ownership_ממשלתי',
 'Prev_ownership_פרטית',
 'Curr_ownership_השכרה',
 'Curr_ownership_חברה',
 'Curr_ownership_ליסינג',
 'Curr_ownership_פרטית',
 'City_Mid_Central',
 'City_North',
 'City_North_Central',
 'City_South',
 'City_South_Central',
 'City_Unknown',
 'Color_less',
 'Color_unknown',
 'Year',
 'Hand',
 'capacity_Engine',
 'Pic_num',
 'Repub_date',
 'Km',
 'rank_by_m_and_y',
 'rank_by_m']


#########################################################################################################

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    features = request.form.getlist('feature')
    features.append(None) ## It was added as a supply_score value in order to keep the original order for data_prep function
    
    ## Change the '' values from the HTML to None in python0
    features_dic={}
    for i in range(len(columns)):
        if features[i] == '' :
            features_dic[columns[i]] = None
        else:
            features_dic[columns[i]] = features[i]
            
    ## Convert the data types from the HTML file to the required types for our model
    for key in ['Year','Hand', 'Pic_num']:
        if isinstance(features_dic[key],str):
            features_dic[key]=int(features_dic[key])
    
    for key in ['Cre_date','Repub_date','Test',]:
        if isinstance(features_dic[key],str):
            features_dic[key]=features_dic[key].split('-')
            real_date = f"{features_dic[key][2]}/{features_dic[key][1]}/{features_dic[key][0]}"
            features_dic[key]=real_date
    
    ## Preparing the test set:
    df = pd.DataFrame([features_dic])
    prepared_features = prepare_data(df)
    
    ## First we built the main data frame based on the original columns.
    ## Then we replaced the values in this df to the user's input values 
    final_dict = {}
    for column in all_columns:
        final_dict[column] = 0
    final_features = pd.DataFrame([final_dict])

    for column in final_features.columns:
        if column in prepared_features.columns:
            if pd.notna(prepared_features.loc[0, column]):
                final_features.loc[0, [column]] = prepared_features.loc[0,[column]]

    ## Prediction of the price:
    prediction = model.predict(final_features)[0]

    output_text = f"The prediction for the price is: {round(prediction, 2)}"

    return render_template('index.html', prediction_text='{}'.format(output_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)



