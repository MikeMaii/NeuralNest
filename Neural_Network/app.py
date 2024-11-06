import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('NN_model.h5')

# Load category mappings
town_categories = joblib.load('town_categories.pkl')
street_name_categories = joblib.load('street_name_categories.pkl')

# Load the scaler used during training
scaler = joblib.load('NN_scaler.pkl')

# Load the data
town_data = pd.read_csv('Resale Flat Price.csv')

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

def get_advice(monthly_avg, year, predicted_price, town, street_name, floor, flat_type, flat_model, floor_area_sqm, lease_commence_date):
    """
    Calls OpenAI API to get advice on whether the property is a good buy.

    Parameters:
    - monthly_avg (float): The average resale price of all house types sold in the area from 2017 to 2024.
    - year (int): The current year when the property is being evaluated.
    - predicted_price (float): The predicted resale price of the property.
    - town (str): The town where the property is located.
    - street_name (str): The street name of the property location.
    - floor (int): The floor level of the property.
    - flat_type (str): The type of flat (e.g., '2 ROOM', '3 ROOM').
    - flat_model (str): The model of the flat (e.g., 'Improved', 'Model A').
    - floor_area_sqm (float): The floor area of the flat in square meters.
    - lease_commence_date (int): The year the lease of the property commenced.

    Returns:
    - str: Advice on whether the property is a good buy, based on resale trends and investment potential.
    """
    
    # Calculate remaining lease
    remaining_years = 99 - (year - lease_commence_date)
    
    # Prepare a prompt with the property details
    prompt = f"""I have a property with the following details:
    - Predicted resale price: S${predicted_price} at {year}
    - Monthly Average of all types of houses sold in the same area from 2017 to 2024: {monthly_avg}
    (Note that the difference in predicted price and monthly average might be because of the nature of the flat type/size of house)
    - Town: {town}
    - Street Name: {street_name}
    - Floor: {floor}
    - Flat Type: {flat_type}
    - Flat Model: {flat_model}
    - Floor Area (sqm): {floor_area_sqm}
    - Remaining Lease: {remaining_years}
    
    Based on these details, please advise whether this is a good buy in Singapore, considering resale trends and investment potential in point form. Talk about monthly average trends.
    Do not ask me to check for anything.
    Give a concise response.
    """
    
    # Make the OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a helpful real estate advisor"}, 
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract and return the response text
    advice = response.choices[0].message.content
    advice = advice.replace("*", "").replace("#", "")    
    return advice

def NN_predict(year, month, town, street_name, floor, flat_type, flat_model, floor_area_sqm, lease_commence_date):
    
    """
    Predicts the resale price of a property using a neural network model.

    Parameters:
    - year (int): The current year when the property is being evaluated.
    - month (int): The current month when the property is being evaluated.
    - town (str): The town where the property is located.
    - street_name (str): The street name of the property location.
    - floor (int): The floor level of the property.
    - flat_type (str): The type of flat (e.g., '2 ROOM', '3 ROOM').
    - flat_model (str): The model of the flat (e.g., 'Improved', 'Model A').
    - floor_area_sqm (float): The floor area of the flat in square meters.
    - lease_commence_date (int): The year the lease of the property commenced.

    Returns:
    - str: The predicted resale price in Singapore dollars, formatted as a string.
    """
    
    # Calculate remaining lease
    remaining_years = 99 - (year - lease_commence_date)
    
    # Encode 'town' and 'street_name'
    if town in town_categories and street_name in street_name_categories:
        town_code = town_categories.index(town)
        street_name_code = street_name_categories.index(street_name)
    else:
        return "Error: Town or Street Name not found in mappings."

    # Prepare inputs
    town_embedding_vector = np.array([[town_code]], dtype=np.int32)
    street_name_embedding_vector = np.array([[street_name_code]], dtype=np.int32)

    # Map flat_type and flat_model to encoded values
    flat_type_mapping = {
        '1 ROOM': 1,
        '2 ROOM': 2,
        '3 ROOM': 3,
        '4 ROOM': 4,
        '5 ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI-GENERATION': 7
    }
    flat_model_mapping = {
        '2-room': 1,
        'Improved': 2,
        'Simplified': 3,
        'Standard': 4,
        'Apartment': 5,
        'Type S1': 6,
        'Type S2': 7,
        'Model A': 8,
        'Model A2': 9,
        'New Generation': 10,
        'Adjoined flat': 11,
        'Improved-Maisonette': 12,
        'Maisonette': 13,
        'Model A-Maisonette': 14,
        'Multi Generation': 15,
        'Premium Apartment': 16,
        'Premium Maisonette': 17,
        'DBSS': 18,
        'Terrace': 19,
        'Premium Apartment Loft': 20,
        '3Gen': 21
    }

    encoded_flat_type = flat_type_mapping[flat_type]
    encoded_flat_model = flat_model_mapping[flat_model]

    # Convert inputs to correct types
    year = int(year)
    month = int(month)
    floor = int(floor)
    floor_area_sqm = float(floor_area_sqm)
    remaining_years = int(remaining_years)

    # Scale continuous features using the same scaler as during training
    numeric_features = np.array([[year, month, floor, floor_area_sqm, remaining_years]], dtype=np.float32)
    scaled_numeric_features = scaler.transform(numeric_features)

    # Prepare the numeric input by concatenating scaled continuous and categorical features
    numeric_input = np.concatenate([scaled_numeric_features, [[encoded_flat_type, encoded_flat_model]]], axis=1).astype(np.float32)  # Shape (1, 7)

    # Pass the three separate inputs as required by the model
    prediction = model.predict([town_embedding_vector, street_name_embedding_vector, numeric_input])

    # Format prediction output
    predicted_price = prediction[0][0]
    output = f"S${predicted_price:,.2f}"

    return output

def plot(town_data, street_name, forecast_years=1):
    """
    Generates a time series plot for the resale price trend over time for a specified street name.

    Parameters:
    - town_data (DataFrame): The DataFrame containing the data with columns 'street_name', 'time', and 'resale_price'.
    - street_name (str): The street name to filter the data by.

    Returns:
    - tuple: A tuple containing the formatted prediction as a string and the file path to the saved plot image.
    """
    
    # Filter data for the specified street name
    filtered_town_data = town_data[town_data['street_name'] == street_name]
    
    # Check if the filtered dataset is empty
    if filtered_town_data.empty:
        raise ValueError("No flat type in the area")
    
    # Extract sale year and month from the 'month' column
    filtered_town_data['sale_year'] = filtered_town_data['month'].apply(lambda x: int(x.split('-')[0]))  # Get year from 'month'
    filtered_town_data['sale_month'] = filtered_town_data['month'].apply(lambda x: int(x.split('-')[1]))  # Get month from 'month'

    # Convert the 'month' column to a numerical representation
    filtered_town_data['time'] = filtered_town_data['sale_year'] + (filtered_town_data['sale_month'] - 1) / 12
    
    # Sort data by 'time'
    filtered_data = filtered_town_data.sort_values(by='time')
    
    # Group by month and calculate the average resale price
    monthly_avg = filtered_data.groupby('time')['resale_price'].mean().reset_index()
    monthly_avg = monthly_avg[:-1]
    
    # Plot the average monthly resale price trend
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg['time'], monthly_avg['resale_price'], color='blue', label='Average Monthly Resale Price')
    plt.xlabel('Time (Year)')
    plt.ylabel('Average Resale Price')
    plt.title(f'Average Monthly Resale Price Trend Over Time for {street_name}')
    plt.legend()
    
    # Save the plot and close it
    plot_path = "resale_price_trend.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Format prediction and return results
    return plot_path, monthly_avg


def predict_price(year, month, town, street_name, floor, flat_type, flat_model, floor_area_sqm, lease_commence_date):
    
    predicted_price = NN_predict(year, month, town, street_name, floor, flat_type, flat_model, floor_area_sqm, lease_commence_date)
    plot_path, monthly_avg = plot(town_data, street_name)
    recommendation = get_advice(monthly_avg, year, predicted_price, town, street_name, floor, flat_type, flat_model, floor_area_sqm, lease_commence_date)
    
    return predicted_price, plot_path, recommendation    

# # Example usage
# predicted = predict_price(
#     year=2024, 
#     month=10, 
#     town='ANG MO KIO', 
#     street_name='ANG MO KIO AVE 10', 
#     floor=10, 
#     flat_type='2 ROOM', 
#     flat_model='Improved', 
#     floor_area_sqm=44, 
#     lease_commence_date=1979
# )

# print(predicted)

# Gradio
import gradio as gr

interface = gr.Interface(
    fn=predict_price,
    title="NeuralNest", 
    inputs=[
        gr.components.Number(label="Year", value = 2024), 
        gr.components.Number(label="Month", value = 11), 
        gr.components.Dropdown(choices=['ANG MO KIO',
                                        'BEDOK',
                                        'BISHAN',
                                        'BUKIT BATOK',
                                        'BUKIT MERAH',
                                        'BUKIT PANJANG',
                                        'BUKIT TIMAH',
                                        'CENTRAL AREA',
                                        'CHOA CHU KANG',
                                        'CLEMENTI',
                                        'GEYLANG',
                                        'HOUGANG',
                                        'JURONG EAST',
                                        'JURONG WEST',
                                        'KALLANG/WHAMPOA',
                                        'MARINE PARADE',
                                        'PASIR RIS',
                                        'PUNGGOL',
                                        'QUEENSTOWN',
                                        'SEMBAWANG',
                                        'SENGKANG',
                                        'SERANGOON',
                                        'TAMPINES',
                                        'TOA PAYOH',
                                        'WOODLANDS',
                                        'YISHUN'], value = 'ANG MO KIO', label="Town"), 
        gr.components.Dropdown(choices=['ANG MO KIO AVE 10',
            'ANG MO KIO AVE 4',
            'ANG MO KIO AVE 5',
            'ANG MO KIO AVE 1',
            'ANG MO KIO AVE 3',
            'ANG MO KIO AVE 9',
            'ANG MO KIO AVE 8',
            'ANG MO KIO AVE 6',
            'ANG MO KIO ST 52',
            'BEDOK NTH AVE 4',
            'BEDOK NTH AVE 1',
            'BEDOK NTH RD',
            'BEDOK STH AVE 1',
            'BEDOK RESERVOIR RD',
            'CHAI CHEE ST',
            'BEDOK NTH ST 3',
            'BEDOK STH RD',
            'CHAI CHEE AVE',
            'NEW UPP CHANGI RD',
            'CHAI CHEE DR',
            'BEDOK STH AVE 2',
            'BEDOK NTH AVE 3',
            'BEDOK RESERVOIR VIEW',
            'CHAI CHEE RD',
            'LENGKONG TIGA',
            'BEDOK CTRL',
            'JLN DAMAI',
            'BEDOK NTH AVE 2',
            'BEDOK STH AVE 3',
            'SIN MING RD',
            'SIN MING AVE',
            'BISHAN ST 12',
            'BISHAN ST 13',
            'BISHAN ST 22',
            'BISHAN ST 24',
            'BISHAN ST 23',
            'BRIGHT HILL DR',
            'SHUNFU RD',
            'BT BATOK ST 34',
            'BT BATOK ST 51',
            'BT BATOK ST 11',
            'BT BATOK ST 52',
            'BT BATOK ST 21',
            'BT BATOK EAST AVE 5',
            'BT BATOK WEST AVE 6',
            'BT BATOK CTRL',
            'BT BATOK WEST AVE 8',
            'BT BATOK EAST AVE 4',
            'BT BATOK ST 31',
            'BT BATOK ST 25',
            'BT BATOK EAST AVE 3',
            'BT BATOK WEST AVE 5',
            'BT BATOK ST 24',
            'JLN BT HO SWEE',
            'TELOK BLANGAH DR',
            'BEO CRES',
            'TELOK BLANGAH CRES',
            'TAMAN HO SWEE',
            'TELOK BLANGAH RISE',
            'TELOK BLANGAH WAY',
            'JLN BT MERAH',
            'JLN KLINIK',
            'TELOK BLANGAH HTS',
            'BT MERAH VIEW',
            'INDUS RD',
            'BT MERAH LANE 1',
            'TELOK BLANGAH ST 31',
            'MOH GUAN TER',
            'HAVELOCK RD',
            'HENDERSON CRES',
            'BT PURMEI RD',
            'KIM TIAN RD',
            'DEPOT RD',
            'JLN RUMAH TINGGI',
            'DELTA AVE',
            'JLN MEMBINA',
            'REDHILL RD',
            'LENGKOK BAHRU',
            'ZION RD',
            'PETIR RD',
            'PENDING RD',
            'BANGKIT RD',
            'SEGAR RD',
            'JELAPANG RD',
            'SENJA RD',
            'FAJAR RD',
            'BT PANJANG RING RD',
            'SENJA LINK',
            'LOMPANG RD',
            'GANGSA RD',
            'TOH YI DR',
            'FARRER RD',
            'JLN KUKOH',
            'ROWELL RD',
            'WATERLOO ST',
            'NEW MKT RD',
            'TG PAGAR PLAZA',
            'QUEEN ST',
            'BAIN ST',
            'CANTONMENT RD',
            'TECK WHYE LANE',
            'CHOA CHU KANG AVE 4',
            'CHOA CHU KANG AVE 3',
            'CHOA CHU KANG CRES',
            'CHOA CHU KANG ST 54',
            'CHOA CHU KANG CTRL',
            'JLN TECK WHYE',
            'CHOA CHU KANG ST 62',
            'CHOA CHU KANG NTH 6',
            'CHOA CHU KANG DR',
            'CHOA CHU KANG NTH 5',
            'CHOA CHU KANG ST 52',
            'CHOA CHU KANG AVE 2',
            'CLEMENTI WEST ST 2',
            'WEST COAST RD',
            'CLEMENTI WEST ST 1',
            'CLEMENTI AVE 4',
            'CLEMENTI AVE 5',
            'CLEMENTI ST 11',
            'CLEMENTI AVE 2',
            'CLEMENTI AVE 3',
            'CLEMENTI AVE 1',
            "C'WEALTH AVE WEST",
            'CIRCUIT RD',
            'BALAM RD',
            'MACPHERSON LANE',
            'EUNOS CRES',
            'UBI AVE 1',
            'HAIG RD',
            'OLD AIRPORT RD',
            'GEYLANG EAST AVE 1',
            'SIMS DR',
            'PIPIT RD',
            'GEYLANG EAST CTRL',
            'EUNOS RD 5',
            'CASSIA CRES',
            'BUANGKOK CRES',
            'HOUGANG AVE 3',
            'HOUGANG AVE 8',
            'HOUGANG AVE 1',
            'HOUGANG AVE 5',
            'HOUGANG ST 61',
            'HOUGANG ST 11',
            'HOUGANG AVE 7',
            'HOUGANG AVE 4',
            'HOUGANG AVE 2',
            'LOR AH SOO',
            'HOUGANG ST 92',
            'HOUGANG ST 52',
            'HOUGANG AVE 10',
            'HOUGANG ST 51',
            'UPP SERANGOON RD',
            'HOUGANG CTRL',
            'HOUGANG ST 91',
            'BUANGKOK LINK',
            'HOUGANG ST 31',
            'PANDAN GDNS',
            'TEBAN GDNS RD',
            'JURONG EAST ST 24',
            'JURONG EAST ST 21',
            'JURONG EAST AVE 1',
            'JURONG EAST ST 13',
            'JURONG EAST ST 32',
            'TOH GUAN RD',
            'JURONG WEST ST 93',
            'BOON LAY AVE',
            'HO CHING RD',
            'BOON LAY DR',
            'TAO CHING RD',
            'JURONG WEST ST 91',
            'JURONG WEST ST 42',
            'JURONG WEST ST 92',
            'BOON LAY PL',
            'JURONG WEST ST 52',
            'TAH CHING RD',
            'JURONG WEST ST 81',
            'YUNG SHENG RD',
            'JURONG WEST ST 25',
            'JURONG WEST ST 73',
            'JURONG WEST ST 72',
            'JURONG WEST AVE 3',
            'JURONG WEST AVE 5',
            'YUNG HO RD',
            'JURONG WEST ST 74',
            'JURONG WEST AVE 1',
            'JURONG WEST ST 71',
            'JURONG WEST ST 61',
            'JURONG WEST ST 65',
            'JURONG WEST CTRL 1',
            'JURONG WEST ST 64',
            'JURONG WEST ST 62',
            'JURONG WEST ST 41',
            'JURONG WEST ST 24',
            'JLN BATU',
            'JLN BAHAGIA',
            'LOR LIMAU',
            "ST. GEORGE'S RD",
            'KALLANG BAHRU',
            'DORSET RD',
            'GEYLANG BAHRU',
            'BENDEMEER RD',
            'WHAMPOA DR',
            'UPP BOON KENG RD',
            'RACE COURSE RD',
            'OWEN RD',
            'NTH BRIDGE RD',
            'TOWNER RD',
            'FARRER PK RD',
            'MCNAIR RD',
            'JLN TENTERAM',
            'BOON KENG RD',
            'JLN RAJAH',
            'MARINE DR',
            'MARINE CRES',
            'MARINE TER',
            'CHANGI VILLAGE RD',
            'PASIR RIS ST 71',
            'PASIR RIS ST 11',
            'PASIR RIS DR 3',
            'PASIR RIS DR 6',
            'PASIR RIS ST 21',
            'PASIR RIS DR 4',
            'PASIR RIS ST 53',
            'PASIR RIS DR 10',
            'PASIR RIS ST 52',
            'PASIR RIS ST 12',
            'PASIR RIS ST 51',
            'PASIR RIS ST 72',
            'PASIR RIS DR 1',
            'PUNGGOL FIELD',
            'EDGEDALE PLAINS',
            'PUNGGOL FIELD WALK',
            'EDGEFIELD PLAINS',
            'PUNGGOL RD',
            'PUNGGOL EAST',
            'PUNGGOL DR',
            'PUNGGOL CTRL',
            'PUNGGOL PL',
            "C'WEALTH CL",
            'STIRLING RD',
            'MEI LING ST',
            "C'WEALTH CRES",
            "C'WEALTH DR",
            'GHIM MOH RD',
            'DOVER RD',
            'HOLLAND AVE',
            'STRATHMORE AVE',
            'HOLLAND DR',
            'GHIM MOH LINK',
            'CLARENCE LANE',
            'DOVER CRES',
            'SEMBAWANG DR',
            'SEMBAWANG CL',
            'MONTREAL DR',
            'ADMIRALTY LINK',
            'ADMIRALTY DR',
            'SEMBAWANG CRES',
            'CANBERRA RD',
            'FERNVALE RD',
            'COMPASSVALE LANE',
            'ANCHORVALE RD',
            'RIVERVALE DR',
            'RIVERVALE CRES',
            'SENGKANG EAST WAY',
            'RIVERVALE ST',
            'RIVERVALE WALK',
            'FERNVALE LANE',
            'ANCHORVALE LINK',
            'COMPASSVALE RD',
            'COMPASSVALE CRES',
            'JLN KAYU',
            'COMPASSVALE WALK',
            'COMPASSVALE DR',
            'COMPASSVALE LINK',
            'COMPASSVALE BOW',
            'SENGKANG CTRL',
            'ANCHORVALE LANE',
            'ANCHORVALE DR',
            'COMPASSVALE ST',
            'SERANGOON AVE 4',
            'LOR LEW LIAN',
            'SERANGOON AVE 2',
            'SERANGOON NTH AVE 1',
            'SERANGOON AVE 1',
            'SERANGOON CTRL',
            'SERANGOON NTH AVE 4',
            'TAMPINES ST 22',
            'TAMPINES ST 41',
            'TAMPINES AVE 4',
            'TAMPINES ST 44',
            'TAMPINES ST 81',
            'TAMPINES ST 11',
            'TAMPINES ST 23',
            'TAMPINES ST 91',
            'TAMPINES ST 21',
            'TAMPINES ST 83',
            'TAMPINES ST 42',
            'TAMPINES ST 71',
            'TAMPINES ST 45',
            'TAMPINES ST 34',
            'TAMPINES ST 82',
            'TAMPINES AVE 9',
            'SIMEI ST 1',
            'SIMEI ST 5',
            'TAMPINES ST 72',
            'TAMPINES ST 84',
            'SIMEI ST 2',
            'TAMPINES CTRL 7',
            'TAMPINES ST 33',
            'TAMPINES ST 32',
            'TAMPINES AVE 5',
            'LOR 5 TOA PAYOH',
            'LOR 7 TOA PAYOH',
            'LOR 4 TOA PAYOH',
            'LOR 1 TOA PAYOH',
            'TOA PAYOH EAST',
            'POTONG PASIR AVE 1',
            'TOA PAYOH NTH',
            'LOR 8 TOA PAYOH',
            'LOR 3 TOA PAYOH',
            'POTONG PASIR AVE 3',
            'JOO SENG RD',
            'LOR 2 TOA PAYOH',
            'TOA PAYOH CTRL',
            'MARSILING DR',
            'WOODLANDS ST 13',
            'WOODLANDS DR 52',
            'WOODLANDS ST 41',
            'MARSILING CRES',
            'WOODLANDS ST 83',
            'WOODLANDS CIRCLE',
            'WOODLANDS DR 40',
            'WOODLANDS ST 31',
            'WOODLANDS DR 16',
            'WOODLANDS ST 81',
            'WOODLANDS RING RD',
            'WOODLANDS DR 53',
            'WOODLANDS DR 62',
            'WOODLANDS DR 70',
            'WOODLANDS DR 42',
            'WOODLANDS DR 50',
            'WOODLANDS AVE 6',
            'WOODLANDS DR 14',
            'WOODLANDS AVE 1',
            'WOODLANDS AVE 5',
            'MARSILING RISE',
            'WOODLANDS CRES',
            'WOODLANDS DR 73',
            'WOODLANDS DR 44',
            'YISHUN RING RD',
            'YISHUN AVE 3',
            'YISHUN ST 11',
            'YISHUN AVE 4',
            'YISHUN ST 22',
            'YISHUN ST 71',
            'YISHUN AVE 5',
            'YISHUN ST 21',
            'YISHUN ST 41',
            'YISHUN ST 61',
            'YISHUN AVE 6',
            'YISHUN AVE 11',
            'YISHUN CTRL',
            'YISHUN ST 81',
            'YISHUN ST 72',
            'YISHUN AVE 2',
            'ANG MO KIO ST 32',
            'ANG MO KIO ST 31',
            'BEDOK NTH ST 2',
            'BEDOK NTH ST 1',
            'JLN TENAGA',
            'BEDOK NTH ST 4',
            'BT BATOK WEST AVE 4',
            'CANTONMENT CL',
            'BOON TIONG RD',
            'SPOTTISWOODE PK RD',
            'REDHILL CL',
            'KIM TIAN PL',
            'CASHEW RD',
            "QUEEN'S RD",
            'CHANDER RD',
            'KELANTAN RD',
            'SAGO LANE',
            'UPP CROSS ST',
            'CHIN SWEE RD',
            'SMITH ST',
            'TECK WHYE AVE',
            'CHOA CHU KANG ST 51',
            'CHOA CHU KANG AVE 5',
            'CHOA CHU KANG AVE 1',
            'WEST COAST DR',
            'PAYA LEBAR WAY',
            'ALJUNIED CRES',
            'JOO CHIAT RD',
            'PINE CL',
            'HOUGANG ST 22',
            'HOUGANG AVE 9',
            'HOUGANG AVE 6',
            'HOUGANG ST 21',
            'JURONG WEST ST 75',
            'KANG CHING RD',
            'KG KAYU RD',
            'CRAWFORD LANE',
            'WHAMPOA WEST',
            'BEACH RD',
            'CAMBRIDGE RD',
            "ST. GEORGE'S LANE",
            'JELLICOE RD',
            'ELIAS RD',
            'HOLLAND CL',
            'TANGLIN HALT RD',
            "C'WEALTH AVE",
            'WELLINGTON CIRCLE',
            'CANBERRA LINK',
            'SENGKANG WEST AVE',
            'SENGKANG EAST RD',
            'SERANGOON CTRL DR',
            'SERANGOON AVE 3',
            'SERANGOON NTH AVE 3',
            'TAMPINES AVE 8',
            'TAMPINES ST 24',
            'TAMPINES ST 12',
            'SIMEI LANE',
            'SIMEI ST 4',
            'LOR 6 TOA PAYOH',
            'KIM KEAT LINK',
            'MARSILING LANE',
            'WOODLANDS ST 82',
            'WOODLANDS DR 60',
            'WOODLANDS AVE 3',
            'WOODLANDS DR 75',
            'WOODLANDS AVE 4',
            'WOODLANDS ST 32',
            'YISHUN AVE 7',
            'ANG MO KIO ST 11',
            'BISHAN ST 11',
            'BT BATOK WEST AVE 2',
            'BT BATOK ST 32',
            'BT BATOK ST 33',
            'BT BATOK ST 22',
            'BT BATOK WEST AVE 7',
            'HOY FATT RD',
            'SILAT AVE',
            'EVERTON PK',
            'BT MERAH CTRL',
            'JELEBU RD',
            'EMPRESS RD',
            'VEERASAMY RD',
            'CHOA CHU KANG ST 64',
            'CHOA CHU KANG ST 53',
            'CHOA CHU KANG NTH 7',
            'CLEMENTI AVE 6',
            'CLEMENTI ST 13',
            'GEYLANG SERAI',
            'JLN TIGA',
            'ALJUNIED RD',
            'YUNG LOH RD',
            'YUNG AN RD',
            "JLN MA'MOR",
            'WHAMPOA RD',
            'LOR 3 GEYLANG',
            'PASIR RIS ST 13',
            "QUEEN'S CL",
            'DOVER CL EAST',
            'SEMBAWANG VISTA',
            'TAMPINES ST 43',
            'SIMEI RD',
            'KIM KEAT AVE',
            'UPP ALJUNIED LANE',
            'POTONG PASIR AVE 2',
            'WOODLANDS DR 72',
            'MARSILING RD',
            'WOODLANDS DR 71',
            'YISHUN AVE 9',
            'YISHUN ST 20',
            'ANG MO KIO ST 21',
            'TIONG BAHRU RD',
            'KLANG LANE',
            'CHOA CHU KANG LOOP',
            'CLEMENTI ST 14',
            'SIMS PL',
            'JURONG EAST ST 31',
            'YUAN CHING RD',
            'CORPORATION DR',
            'YUNG PING RD',
            'WHAMPOA STH',
            'TESSENSOHN RD',
            'JLN DUSUN',
            'QUEENSWAY',
            'FERNVALE LINK',
            'KIM PONG RD',
            'KIM CHENG ST',
            'SAUJANA RD',
            'BUFFALO RD',
            'CLEMENTI ST 12',
            'DAKOTA CRES',
            'JURONG WEST ST 51',
            'FRENCH RD',
            'GLOUCESTER RD',
            'KG ARANG RD',
            'MOULMEIN RD',
            'KENT RD',
            'AH HOOD RD',
            'SERANGOON NTH AVE 2',
            'TAMPINES CTRL 1',
            'TAMPINES AVE 7',
            'LOR 1A TOA PAYOH',
            'WOODLANDS AVE 9',
            'YISHUN CTRL 1',
            'LOWER DELTA RD',
            'JLN DUA',
            'WOODLANDS ST 11',
            'ANG MO KIO AVE 2',
            'SELEGIE RD',
            'SIMS AVE',
            'REDHILL LANE',
            "KING GEORGE'S AVE",
            'PASIR RIS ST 41',
            'PUNGGOL WALK',
            'LIM LIAK ST',
            'JLN BERSEH',
            'SENGKANG WEST WAY',
            'BUANGKOK GREEN',
            'SEMBAWANG WAY',
            'PUNGGOL WAY',
            'YISHUN ST 31',
            'TECK WHYE CRES',
            'KRETA AYER RD',
            'MONTREAL LINK',
            'UPP SERANGOON CRES',
            'SUMANG LINK',
            'SENGKANG EAST AVE',
            'YISHUN AVE 1',
            'ANCHORVALE CRES',
            'YUNG KUANG RD',
            'ANCHORVALE ST',
            'TAMPINES CTRL 8',
            'YISHUN ST 51',
            'UPP SERANGOON VIEW',
            'TAMPINES AVE 1',
            'BEDOK RESERVOIR CRES',
            'ANG MO KIO ST 61',
            'DAWSON RD',
            'FERNVALE ST',
            'SENG POH RD',
            'HOUGANG ST 32',
            'TAMPINES ST 86',
            'HENDERSON RD',
            'SUMANG WALK',
            'CHOA CHU KANG AVE 7',
            'KEAT HONG CL',
            'JURONG WEST CTRL 3',
            'KEAT HONG LINK',
            'ALJUNIED AVE 2',
            'CANBERRA CRES',
            'SUMANG LANE',
            'CANBERRA ST',
            'ANG MO KIO ST 44',
            'ANG MO KIO ST 51',
            'BT BATOK EAST AVE 6',
            'BT BATOK WEST AVE 9',
            'GEYLANG EAST AVE 2',
            'MARINE PARADE CTRL',
            'CANBERRA WALK',
            'WOODLANDS RISE',
            'TAMPINES ST 61',
            'YISHUN ST 43',
            'SENGKANG WEST RD',
            'BIDADARI PK DR',
            'CANBERRA VIEW'], value = "ANG MO KIO AVE 10", label = "Street Name"),
        gr.components.Number(label="Floor", value = 1), 
        gr.components.Dropdown(choices=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'], value='1 ROOM', label="Flat Type"), 
        gr.components.Dropdown(choices=['2-room', 'Improved', 'Simplified', 'Standard', 'Apartment', 'Type S1', 'Type S2', 'Model A', 'Model A2', 'New Generation', 'Adjoined flat', 'Improved-Maisonette', 'Maisonette', 'Model A-Maisonette', 'Multi Generation', 'Premium Apartment', 'Premium Maisonette', 'DBSS', 'Terrace', 'Premium Apartment Loft', '3Gen'], value='Improved', label="Flat Model"),
        gr.components.Number(label="Floor Area (sqm)"),
        gr.components.Number(label="Lease Commence Year")
    ], 
    outputs=[gr.components.Textbox(label="Predicted Resale Price"),
            gr.components.Image(type="filepath", label="Resale Price Trend Timeseries with following year prediction"),
            gr.components.Textbox(label="AI Real Estate Advisor")],
    allow_flagging="never"
)

# Launch the interface
interface.launch()
