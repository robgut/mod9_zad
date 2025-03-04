import streamlit as st
from dotenv import dotenv_values
from dotenv import load_dotenv
import pandas as pd
import os
from io import BytesIO
from PIL import Image
import json
from openai import OpenAI
# from langfuse.decorators import observe
# from langfuse.openai import OpenAI as LangfuseOpenAI
# from langfuse import Langfuse
import boto3
from pycaret.regression import setup, load_model, plot_model, predict_model # type: ignore


load_dotenv()
env = dotenv_values(".env")

# langfuse = Langfuse()
# langfuse.auth_check()

MODELS_PATH = 'halfmarathon/models/'
DATA_PATH = 'halfmarathon/data/'
BUCKET_NAME = "nowy"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def get_digital_ocean_client():
    return boto3.client('s3',)

def get_file_list(prefix):
    s3 = get_digital_ocean_client()
    response = s3.list_objects_v2(Bucket = BUCKET_NAME, Prefix = prefix)
    return response['Contents']

@st.cache_data
def read_data_files():
    file_list = get_file_list(prefix=DATA_PATH)
    csv_names = []

    for item in file_list:
        f = item['Key']
        if str(f).endswith('.csv'):
            csv_names.append(f)

    file1 = download_file(csv_names[0]) 
    file2 = download_file(csv_names[1]) 

    m2023_df = pd.read_csv(file1, sep=';')    
    m2024_df = pd.read_csv(file2, sep=';')

    m2023_df = m2023_df[['Miejsce', 'Nazwisko','Płeć', 'Kategoria wiekowa', '5 km Czas', 'Czas']]
    m2024_df = m2024_df[['Miejsce', 'Nazwisko','Płeć', 'Kategoria wiekowa', '5 km Czas', 'Czas']]

    m2023_df['Czas'] = m2023_df['Czas'].apply(convert_time_to_seconds)
    m2023_df['5 km Czas'] = m2023_df['5 km Czas'].apply(convert_time_to_seconds)
    m2024_df['Czas'] = m2024_df['Czas'].apply(convert_time_to_seconds)
    m2024_df['5 km Czas'] = m2024_df['5 km Czas'].apply(convert_time_to_seconds)

    return m2023_df, m2024_df

def download_file(file_full_name):
    s3 = get_digital_ocean_client()
    base_name = os.path.basename(file_full_name)
    s3.download_file(BUCKET_NAME, file_full_name, base_name)
    return base_name

def upload_file(file_name, prefix):
    s3 = get_digital_ocean_client()
    s3.upload_file(file_name, BUCKET_NAME, prefix + file_name)

@st.cache_resource
def download_models():
    models = []

    try:
        model_list = get_file_list(prefix=MODELS_PATH)
                
        for item in model_list:
            model = item['Key']
            if str(model).endswith('.pkl'):
                models.append(model)
    except:
        return None

    return models

def download_image(img_name):
    try:
        s3 = get_digital_ocean_client()
        image = BytesIO()
        img_path = MODELS_PATH + "img/" + img_name
        data = s3.download_fileobj(BUCKET_NAME, img_path, image)
        image.seek(0)

        return image
    except:
        return None

def validate_response(response: dict = None):
    if dict == None:
        return 'Niepoprawne dane'
    gender = response.get('gender')
    if str(gender).lower() == None:
        return 'Nieznana płeć'
    if str(gender).lower() == 'unknown':
        return 'Nieznana płeć'
    if str(gender).lower() != 'female' and str(gender).lower() != 'male':
        return 'Nieznana płeć'
    if type(response.get('age')) is not int:
        return 'Nieznany wiek'
    if type(response.get('5time')) is not int:
        return 'Nieznany czas na 5 km'
    
    return True

def get_response(response: dict):
    try:
        test_df = pd.DataFrame([
        {
            'sex' : None,
            'age' : None,
            'age_category' : None,
            '5time' : None, # konwertuj do sekund
        }])

        ret_val = validate_response(response)

        if ret_val == True:
            test_df['age'] = response['age']
            test_df['5time'] = response['5time'] * 60 # konwersja do sekund

            if response['gender'] == 'female':
                test_df['sex'] = 'K'
            else:
                test_df['sex'] = 'M' 

            test_df['age_category'] = pd.cut(test_df['age'], 
                                            bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 99],
                                            labels=['10', '20', '30', '40', '50', '60', '70', '80', '90']
                                            )
            test_df['age_category'] = test_df['age_category'].apply(lambda x: test_df['sex'][0] + x)
    except:
        return test_df[['sex', 'age_category', '5time']]

    return test_df[['sex', 'age_category', '5time']]

def convert_seconds_to_time(seconds):
    seconds = int(seconds)
    min, sec = divmod(seconds, 60)
    hrs, min = divmod(min, 60)

    return f'{hrs:02d}:{min:02d}:{sec:02d}'

def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    time = time.split(':')
    return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

def row_color(row):
    if row['Nazwisko'] == "Ty":
        return ['background-color:orange'] * len(row)
    else:
        return [''] * len(row)
    
def get_difference(row, your_time: int):
    return (your_time - row['Czas'])

def get_your_place_df(df_tosearch, your_data : dict, your_time: int):
    try:
        df_tosearch['diff'] = df_tosearch.apply(get_difference, axis=1, your_time=int(your_time))
        df_tosearch['diff'] = df_tosearch['diff'].abs()

        min_val = df_tosearch['diff'].min()
        min_idx = df_tosearch.loc[df_tosearch['diff'] == min_val].index[0]

        # place_df = df_tosearch.loc[(df_tosearch['Czas'] >= int(your_time) - 25) & (df_tosearch['Czas'] <= int(your_time) + 25)]
        low = 0
        high = 0

        if min_idx - 5 <= 0:
            low = 0
            high = 10
        elif min_idx + 5 > df_tosearch.index.max():
            low = df_tosearch.index.max() - 10
            high = df_tosearch.index.max()
        else:
            low = min_idx - 5
            high = min_idx + 5

        place_df = df_tosearch.loc[low:high]
        
        new_row = {'Miejsce':place_df.loc[min_idx, 'Miejsce'], 'Nazwisko':'Ty', 'Płeć':your_data['sex'][0], 'Kategoria wiekowa':your_data['age_category'][0], 
                '5 km Czas': your_data['5time'][0], 'Czas':int(your_time)}
        place_df.loc[min_idx] = new_row
        place_df = place_df.drop(columns=['diff'])
        
        place_df.dropna(inplace=True)
        place_df['Czas'] = place_df['Czas'].apply(convert_seconds_to_time)
        place_df['5 km Czas'] = place_df['5 km Czas'].apply(convert_seconds_to_time)  
        place_df['Miejsce'] = place_df['Miejsce'].astype(int) 
    except Exception as ex:
        st.write(ex)
        miejsce = ""
        
        min_time = df_tosearch['Czas'].min()
        max_time = df_tosearch['Czas'].max()
        print(type(min_time), type(max_time))
        
        if int(your_time) > max_time:
            miejsce = "Jesteś ostatni, Twój czas jest większy od największego zarejestrowanego czasu."
        elif int(your_time) < min_time:
            miejsce = "Wygrałeś ten bieg z ogromną przewagą..."
        else:
            miejsce = 'Brak wiarygodnych danych do porównania'

        new_row = [{'Nazwisko':'Ty', 'Njlepszy czas': convert_seconds_to_time(min_time), 
                    'Najgorszy czas':convert_seconds_to_time(max_time), 'Miejsce': miejsce}]
        return pd.DataFrame(new_row)
    
    return place_df

# @observe
def get_data_from_text(prompt, input_text):
    openai_client = get_openai_client()

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=60,
        response_format={'type':'json_object'},
        # name="get_data_from_text",
        messages=[
            {"role": "system", "content": [ 
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            },
            {'role':'user', 'content':[
                    {
                        'type': 'text',
                        'text' : input_text
                    }
                ]
            }
        ],
    )

    return response.choices[0].message.content      

def get_model():
    try:
        if st.session_state['current_model_name'] != st.session_state['last_model_name']:
            st.session_state['model_file'] = download_file(st.session_state['current_model_name'])
            st.session_state['last_model_name'] = st.session_state['current_model_name']

        model_file = os.path.splitext(st.session_state['model_file'])[0]
        st.session_state.current_model = load_model(model_file)

        return True
    except Exception as ex:
        return False

if 'model_file' not in st.session_state:
    st.session_state['mode_file'] = ""

def finalize_result():
    if get_model():
        if not st.session_state.runner.empty:
            st.session_state['predict'] = predict_model(st.session_state['current_model'], 
                                                        st.session_state['runner']) 

@st.cache_resource
def get_model_names(model_path: list) -> dict:
    models = {os.path.splitext(os.path.basename(model_path[x]))[0] : model_path[x] for x in range(len(model_path))}

    return models

prompt = """
    Jesteś asystentem, który zawsze zwraca odpowiedź w formacie JSON. Otrzymasz dane zawierające informacje o płci, wieku i czasie na 5 km w minutach. Niech odpowiednie zmienne nazywają się: gender, age, 5time. Zwróć je w języku angielskim
"""

model_mapping = 	{
		'gbr':'GradientBoostingRegressor', 
		'catboost':'CatBoostRegressor', 
		'lightgbm':'LGBMRegressor', 
		'en':'ElasticNet', 
		'llar':'LassoLars', 
		'omp':'OrthogonalMatchingPursuit', 
		'br':'BayesianRidge', 
		'lasso':'Lasso', 
		'lr':'LinearRegression', 
		'ridge':'Ridge',
        'llar':'LassoLars',
	}

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

if 'runner' not in st.session_state:
    st.session_state['runner'] = pd.DataFrame().empty

if 'current_model_name' not in st.session_state:
    st.session_state['current_model_name'] = ""

if 'last_model_name' not in st.session_state:
    st.session_state['last_model_name'] = ""

if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None

if 'predict' not in st.session_state:
    st.session_state['predict'] = None

#
# MAIN
#
st.set_page_config(page_title="Półmaraton - Wrocław", layout="centered")
# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

csv_file_list=[]
models = download_models()
models_dict = get_model_names(models)

m2023_df, m2024_df = read_data_files()

with st.sidebar:
    st.write("#### Wytrenowaliśmy kilka modeli")
    st.write("#### Wybierając model, zobaczysz jak różnią się przewidywane czasy ukończenia półmaratonu")
    st.divider()
    model_selected = st.selectbox('Który model wybierasz?', models_dict.keys())
    st.markdown(f"### Wybrany model: **{model_mapping[model_selected[:-2]]}**", unsafe_allow_html=True)
    st.session_state.current_model_name = models_dict[model_selected]
    st.divider()
    st.markdown("### Które cechy modelu najbardziej wpływają na wynik ?")
    img = download_image(model_selected + ".png").getvalue()
    st.image(img)

st.markdown("<center><h3>Podaj dane osoby aby poznać jak szybko przebiegnie półmaraton</h3></center>", unsafe_allow_html=True)  
st.markdown("<center>Nie wahaj się przed wprowadzaniem wartości brzegowych, np. <b>janek 5 2</b></center>", unsafe_allow_html=True)
st.markdown("<center>lub <b>Anna 60 80</b> albo <b>woman1537</b> i zmieniaj modele, <b>wyniki mogą być zaskakujące!</b></center>", unsafe_allow_html=True)
st.markdown("<center>Te zwykłe zapytania też są dobre, np. <b>Rysiek 35 29</b></center>", unsafe_allow_html=True)
st.markdown("<center><h3>Dobrej zabawy</h3></center>", unsafe_allow_html=True)
st.session_state.input_text = st.text_input("Wprowadź dane", placeholder="płeć, wiek, czas na 5 km w minutach")

submit = st.button("Zatwierdź")

if submit:
    st.write("---")
    with st.spinner('Zaczekaj, trwa analiza żądania i przygotowanie odpowiedzi...'):
        response = ""
        response = get_data_from_text(prompt, st.session_state['input_text'])
        response = str(response).replace("```json", "").replace("```", "")

        answer = dict(json.loads(response))

        is_valid = validate_response(answer)

    if is_valid != True:
        st.info(f"{is_valid}. Uzupełnij dane...")
    else:
        st.session_state.runner = get_response(answer)
        finalize_result()
        st.markdown("<center><h3>Twój wynik:</h3></center>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('Płeć', value=st.session_state['runner']['sex'][0])
        with c2:
            st.metric('Kategoria wiekowa', value=st.session_state['runner']['age_category'][0])
        with c3:
            st.metric('Czas na 5 km [min]', value=convert_seconds_to_time(st.session_state['runner']['5time'][0])) 
        with c4:
            st.metric("Twój czas", value=convert_seconds_to_time(int(st.session_state['predict']['prediction_label']))) 
        
        st.write("---")
        st.markdown("<center><h3>Twoje wirtualne miejsca w maratonach Wrocławskich</h3></center>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<center><h3>2023</h3></center>", unsafe_allow_html=True)
            styled_2023 = get_your_place_df(m2023_df, st.session_state['runner'], st.session_state['predict']['prediction_label'])
            styled_2023 = styled_2023.style.apply(row_color, axis=1)
            st.dataframe(styled_2023, hide_index=True)
        with col2:
            st.markdown("<center><h3>2024</h3></center>", unsafe_allow_html=True)
            styled_2024 = get_your_place_df(m2024_df, st.session_state['runner'], st.session_state['predict']['prediction_label'])
            styled_2024 = styled_2024.style.apply(row_color, axis=1)
            st.dataframe(styled_2024, hide_index=True);
