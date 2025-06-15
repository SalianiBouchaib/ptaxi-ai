import streamlit as st
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Données de Taxis Jaunes",
    page_icon="🚕",
    layout="wide"
)

# Imports standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import missingno as msno
import io
from PIL import Image
from datetime import datetime
import time

# Imports pour les modèles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB

# Imports pour l'évaluation et les métriques
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score
)

# Imports pour le prétraitement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Imports pour la cartographie
import folium
from folium.plugins import HeatMap
import requests
from streamlit_folium import folium_static

# ------ FONCTIONS DE PRÉTRAITEMENT ET ANALYSE ------

# Fonction pour créer la variable cible is_generous
def create_target(df):
    # Éviter division par zéro et valeurs négatives
    mask = (df['fare_amount'] > 0)
    df['tip_percentage'] = 0.0
    df.loc[mask, 'tip_percentage'] = (df.loc[mask, 'tip_amount'] / df.loc[mask, 'fare_amount']) * 100
    df['is_generous'] = (df['tip_percentage'] >= 20).astype(int)
    return df

# Fonction pour charger les données
@st.cache_data
def process_uploaded_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Convertir les colonnes datetime
    try:
        data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
        data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])
    except:
        pass  # Si le format n'est pas compatible, on garde les valeurs d'origine
    
    # Créer la variable cible is_generous
    data = create_target(data)
    return data

# Fonction pour générer des données démo
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 5000
    data = pd.DataFrame({
        'VendorID': np.random.choice([1, 2], n),
        'tpep_pickup_datetime': pd.date_range(start='2017-01-01', periods=n, freq='10min'),
        'tpep_dropoff_datetime': pd.date_range(start='2017-01-01 00:15:00', periods=n, freq='10min'),
        'passenger_count': np.random.randint(1, 7, n),
        'trip_distance': np.random.exponential(3, n),
        'RatecodeID': np.random.choice([1, 2, 3, 4, 5], n, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
        'store_and_fwd_flag': np.random.choice(['Y', 'N'], n, p=[0.1, 0.9]),
        'PULocationID': np.random.randint(1, 266, n),
        'DOLocationID': np.random.randint(1, 266, n),
        'payment_type': np.random.choice([1, 2, 3, 4], n, p=[0.6, 0.35, 0.03, 0.02]),
        'fare_amount': np.random.exponential(10, n) + 2.5,
        'extra': np.random.choice([0, 0.5, 1.0], n),
        'mta_tax': np.random.choice([0, 0.5], n, p=[0.05, 0.95]),
        'tip_amount': np.random.exponential(2, n),
        'tolls_amount': np.random.choice([0, 5.76], n, p=[0.9, 0.1]),
        'improvement_surcharge': np.full(n, 0.3),
    })
    
    # Calculer le montant total
    data['total_amount'] = data['fare_amount'] + data['extra'] + data['mta_tax'] + \
                          data['tip_amount'] + data['tolls_amount'] + data['improvement_surcharge']
    
    # Créer la variable cible is_generous
    data = create_target(data)
    
    return data

# Fonctions de nettoyage et prétraitement des données
def nettoyage(df):
    df_clean = df.copy()
    
    # Colonnes à supprimer
    cols_to_drop = ['extra', 'mta_tax', 'Unnamed: 0']
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(cols_to_drop, axis=1)
    
    # Filtrer les lignes selon des conditions
    conditions = []
    
    if "passenger_count" in df_clean.columns:
        conditions.append(df_clean["passenger_count"] != 0)
    
    if "trip_distance" in df_clean.columns:
        conditions.append(df_clean["trip_distance"] > 0)
    
    if "fare_amount" in df_clean.columns:
        conditions.append(df_clean["fare_amount"] > 0)
    
    if "total_amount" in df_clean.columns:
        conditions.append(df_clean["total_amount"] > 0)
    
    if "tpep_pickup_datetime" in df_clean.columns and pd.api.types.is_datetime64_dtype(df_clean["tpep_pickup_datetime"]):
        if "pickup_minute" not in df_clean.columns:
            df_clean["pickup_minute"] = df_clean["tpep_pickup_datetime"].dt.minute
    
    if "pickup_minute" in df_clean.columns:
        conditions.append(df_clean["pickup_minute"] > 0)
    
    # Appliquer les conditions
    if conditions:
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition
        
        df_clean = df_clean[combined_condition].copy()
    
    # Éliminer les pourboires aberrants si présents
    if 'tip_percentage' in df_clean.columns:
        df_clean = df_clean[df_clean['tip_percentage'] <= 100]  # Limite supérieure raisonnable
    
    return df_clean

def encodage(df):
    df_encoded = df.copy()
    
    # Dictionnaire de mapping pour l'encodage
    code1 = {
        'Y': 1, 'N': 0,
        'night': 1, 'morning': 2, 'afternoon': 3, 'evening': 4,
        'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
        'Thursday': 4, 'Friday': 5, 'Saturday': 6,
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].map(code1)
    
    return df_encoded

def extrairedetime(df):
    df_copy = df.copy()
    datetime_cols = [col for col in df_copy.columns if 'datetime' in col.lower()]
    
    for col in datetime_cols:
        df_copy[col] = pd.to_datetime(df_copy[col])
        col_prefix = col.split('_')[0]
        
        if pd.api.types.is_datetime64_dtype(df_copy[col]):
            df_copy[f'{col_prefix}_year'] = df_copy[col].dt.year
            df_copy[f'{col_prefix}_month'] = df_copy[col].dt.month
            df_copy[f'{col_prefix}_day'] = df_copy[col].dt.day
            df_copy[f'{col_prefix}_hour'] = df_copy[col].dt.hour
            df_copy[f'{col_prefix}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[f'{col_prefix}_week'] = df_copy[col].dt.isocalendar().week
    
    return df_copy

def FeatureEngineering(df):
    df_fe = df.copy()
    
    # Créer la colonne trip_duration
    if all(col in df_fe.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
        if pd.api.types.is_datetime64_dtype(df_fe['tpep_pickup_datetime']) and pd.api.types.is_datetime64_dtype(df_fe['tpep_dropoff_datetime']):
            df_fe['trip_duration'] = (df_fe['tpep_dropoff_datetime'] - df_fe['tpep_pickup_datetime']).dt.total_seconds() / 60
            
            # Créer la colonne speed_mph
            if 'trip_distance' in df_fe.columns:
                non_zero_duration = df_fe['trip_duration'].replace(0, np.nan)
                df_fe['speed_mph'] = df_fe['trip_distance'] / (non_zero_duration / 60)
                df_fe['speed_mph'] = df_fe['speed_mph'].fillna(0)
                
                # Filtrer les vitesses aberrantes
                df_fe = df_fe[(df_fe['speed_mph'] >= 0) & (df_fe['speed_mph'] < 100)]
    
    return df_fe

def standardscler(df, cols):
    df_std = df.copy()
    
    # Colonnes à exclure
    exclude_from_scaling = ['Unnamed: 0', 'is_generous', 'VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']
    cols = [col for col in cols if col not in exclude_from_scaling and col in df_std.columns]
    
    if cols:
        scaler = StandardScaler()
        df_std[cols] = scaler.fit_transform(df_std[cols])
    
    return df_std

# Pipeline complet de prétraitement
def preprocess_data(df):
    """Applique l'ensemble des étapes de prétraitement aux données"""
    with st.spinner("Prétraitement des données en cours..."):
        df_processed = df.copy()
        df_processed = extrairedetime(df_processed)
        df_processed = nettoyage(df_processed)
        df_processed = encodage(df_processed)
        df_processed = FeatureEngineering(df_processed)
        
        # Standardisation optionnelle - la désactiver pour la visualisation spatiale
        # numerical_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        # df_processed = standardscler(df_processed, numerical_cols)
        
    return df_processed

# ------ FONCTIONS POUR LA VISUALISATION GÉOGRAPHIQUE ------

# Fonction pour charger le mapping des location IDs vers coordonnées
@st.cache_data
def load_location_mapping():
    """Charge ou génère un mapping entre LocationIDs et coordonnées géographiques"""
    try:
        # Essayer de charger depuis un fichier ou URL
        url = "https://raw.githubusercontent.com/SalianiBouchaib/ptaxi-ai/main/taxi_zone_lookup.csv"
        location_df = pd.read_csv(url)
        
        # Créer un dictionnaire de lookup
        location_mapping = {}
        for _, row in location_df.iterrows():
            location_id = row['LocationID']
            location_mapping[location_id] = {
                'zone': row['Zone'],
                'borough': row['Borough'],
                'lat': row.get('latitude', None),
                'lon': row.get('longitude', None)
            }
        
        return location_mapping, location_df
    
    except Exception as e:
        st.warning(f"Impossible de charger le fichier de mapping: {e}. Utilisation d'un mapping généré.")
        
        # Créer un mapping simplifié pour la démonstration
        # Coordonnées approximatives pour NYC
        base_lat, base_lon = 40.7128, -74.0060
        
        # Générer un mapping arbitraire pour les démonstrations
        location_mapping = {}
        data = []
        
        for i in range(1, 265):  # NYC a environ 263 taxi zones
            # Générer des coordonnées autour de NYC
            lat = base_lat + (np.random.random() - 0.5) * 0.2
            lon = base_lon + (np.random.random() - 0.5) * 0.2
            
            location_mapping[i] = {
                'zone': f'Zone {i}',
                'borough': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'][i % 5],
                'lat': lat,
                'lon': lon
            }
            
            data.append({
                'LocationID': i,
                'Zone': f'Zone {i}',
                'Borough': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'][i % 5],
                'latitude': lat,
                'longitude': lon
            })
        
        location_df = pd.DataFrame(data)
        return location_mapping, location_df

# Calculer taux de générosité par location
def calculate_generosity_by_location(df, location_mapping):
    """Calcule le taux de générosité par lieu de prise en charge et lieu de dépose"""
    # Vérifier les colonnes nécessaires
    if 'PULocationID' not in df.columns or 'is_generous' not in df.columns:
        st.error("Les colonnes 'PULocationID' et/ou 'is_generous' n'existent pas dans le jeu de données.")
        return None, None
    
    # Statistiques par lieu de prise en charge
    pu_generosity = df.groupby('PULocationID')['is_generous'].agg(['mean', 'count']).reset_index()
    pu_generosity.columns = ['LocationID', 'generosity_rate', 'trip_count']
    pu_generosity['location_type'] = 'Pickup'
    
    # Statistiques par lieu de dépose si disponible
    if 'DOLocationID' in df.columns:
        do_generosity = df.groupby('DOLocationID')['is_generous'].agg(['mean', 'count']).reset_index()
        do_generosity.columns = ['LocationID', 'generosity_rate', 'trip_count']
        do_generosity['location_type'] = 'Dropoff'
    else:
        do_generosity = None
    
    # Ajouter les coordonnées géographiques
    for df_loc in [pu_generosity, do_generosity]:
        if df_loc is not None:
            df_loc['latitude'] = df_loc['LocationID'].map(lambda x: location_mapping.get(x, {}).get('lat'))
            df_loc['longitude'] = df_loc['LocationID'].map(lambda x: location_mapping.get(x, {}).get('lon'))
            df_loc['zone'] = df_loc['LocationID'].map(lambda x: location_mapping.get(x, {}).get('zone'))
            df_loc['borough'] = df_loc['LocationID'].map(lambda x: location_mapping.get(x, {}).get('borough'))
            df_loc['generosity_percentage'] = df_loc['generosity_rate'] * 100  # En pourcentage
    
    return pu_generosity, do_generosity

# Créer une carte de générosité
def create_generosity_map(generosity_data, location_type="Pickup"):
    """Crée une carte Folium montrant le taux de générosité par zone"""
    if generosity_data is None or generosity_data.empty:
        st.error(f"Pas de données disponibles pour les lieux de {location_type}.")
        return None
    
    # Filtrer les données avec coordonnées valides
    valid_data = generosity_data.dropna(subset=['latitude', 'longitude'])
    
    if valid_data.empty:
        st.error("Aucune coordonnée valide trouvée pour la cartographie.")
        return None
    
    # Centre de la carte
    center_lat = valid_data['latitude'].mean()
    center_lon = valid_data['longitude'].mean()
    
    # Créer la carte de base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, 
                  tiles='cartodbpositron')
    
    # Ajouter un titre
    title_html = f'''
        <h3 align="center" style="font-size:16px">
            <b>Taux de générosité par lieu de {location_type.lower()}</b>
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Créer une échelle de couleur pour les cercles
    min_rate = valid_data['generosity_rate'].min()
    max_rate = valid_data['generosity_rate'].max()
    
    # Ajouter les marqueurs pour chaque lieu
    for _, row in valid_data.iterrows():
        # Normaliser le taux de générosité pour la couleur
        if max_rate > min_rate:
            normalized_rate = (row['generosity_rate'] - min_rate) / (max_rate - min_rate)
        else:
            normalized_rate = 0.5
        
        # Rouge → vert selon le taux de générosité
        color = f'#{int(255 * (1 - normalized_rate)):02x}{int(255 * normalized_rate):02x}00'
        
        # Rayon basé sur le nombre de courses
        radius = np.sqrt(row['trip_count']) * 20
        radius = min(max(radius, 100), 1000)  # Limiter la taille du cercle
        
        # Texte du popup
        popup_text = f"""
        <b>Zone:</b> {row['zone']}<br>
        <b>Borough:</b> {row['borough']}<br>
        <b>Taux de générosité:</b> {row['generosity_percentage']:.2f}%<br>
        <b>Nombre de courses:</b> {row['trip_count']}
        """
        
        # Ajouter le cercle
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
    
    # Ajouter une légende
    legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 170px; height: 120px; 
            background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
            padding: 10px; border-radius: 5px;">
            <span style="font-weight: bold;">Taux de générosité</span><br>
            <i style="background: #ff0000; width: 15px; height: 15px; display: inline-block;"></i> Bas<br>
            <i style="background: #ffff00; width: 15px; height: 15px; display: inline-block;"></i> Moyen<br>
            <i style="background: #00ff00; width: 15px; height: 15px; display: inline-block;"></i> Élevé<br>
            <span style="font-size: 10px;"><i>Taille des cercles = nombre de courses</i></span>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Afficher l'analyse spatiale (à utiliser après le prétraitement)
def display_spatial_analysis(df_processed):
    st.write("## Analyse Spatiale des Pourboires")
    
    # Vérifier les colonnes nécessaires
    if 'PULocationID' not in df_processed.columns or 'is_generous' not in df_processed.columns:
        st.error("Les colonnes nécessaires pour l'analyse spatiale (PULocationID, is_generous) ne sont pas disponibles.")
        return
    
    # Charger les données de mapping des locations
    with st.spinner("Chargement des données de localisation..."):
        location_mapping, location_df = load_location_mapping()
    
    st.success(f"Données de localisation chargées avec succès! {len(location_mapping)} zones disponibles.")
    
    # Calculer les statistiques de générosité par localisation
    with st.spinner("Calcul des statistiques de générosité par zone..."):
        pu_generosity, do_generosity = calculate_generosity_by_location(df_processed, location_mapping)
    
    # Type de visualisation
    location_type = st.radio("Choisir le type de localisation à analyser:", 
                            ["Pickup (Prise en charge)", "Dropoff (Dépose)"], 
                            horizontal=True)
    
    if location_type.startswith("Pickup"):
        generosity_data = pu_generosity
        map_title = "Lieux de Prise en Charge"
    else:
        generosity_data = do_generosity
        map_title = "Lieux de Dépose"
    
    # Afficher les données dans un tableau et la carte
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.write(f"### Top 10 des zones par taux de générosité - {map_title}")
        
        if generosity_data is not None and not generosity_data.empty:
            # Filtrer pour n'afficher que les zones avec un nombre minimal de courses
            min_trips = st.slider("Nombre minimum de courses par zone:", 
                               min_value=1, max_value=100, value=10)
            filtered_data = generosity_data[generosity_data['trip_count'] >= min_trips]
            
            if not filtered_data.empty:
                # Préparer les données pour l'affichage
                display_data = filtered_data[['zone', 'borough', 'generosity_percentage', 'trip_count']].sort_values(
                    by='generosity_percentage', ascending=False)
                display_data.columns = ['Zone', 'Borough', 'Taux de générosité (%)', 'Nombre de courses']
                
                # Afficher le tableau
                st.dataframe(display_data.head(10))
            else:
                st.warning(f"Aucune zone avec au moins {min_trips} courses.")
        else:
            st.warning("Pas de données disponibles pour cette analyse.")
    
    with col2:
        st.write(f"### Carte de générosité - {map_title}")
        
        if generosity_data is not None and not generosity_data.empty:
            # Créer la carte
            map_obj = create_generosity_map(generosity_data, location_type.split()[0])
            
            if map_obj:
                # Afficher la carte
                folium_static(map_obj)
            else:
                st.error("Impossible de créer la carte. Vérifiez les données.")
        else:
            st.warning("Pas de données disponibles pour la cartographie.")
    
    # Analyse supplémentaire - relation entre borough et générosité
    if generosity_data is not None and not generosity_data.empty and 'borough' in generosity_data.columns:
        st.write("### Analyse par Borough")
        
        # Générosité moyenne par borough
        borough_stats = generosity_data.groupby('borough').agg({
            'generosity_percentage': 'mean', 
            'trip_count': 'sum'
        }).reset_index()
        
        borough_stats.columns = ['Borough', 'Taux de générosité moyen (%)', 'Total des courses']
        borough_stats = borough_stats.sort_values(by='Taux de générosité moyen (%)', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tableau des statistiques par borough
            st.dataframe(borough_stats)
        
        with col2:
            # Graphique des taux de générosité par borough
            fig = plt.figure(figsize=(10, 6))
            bars = plt.bar(borough_stats['Borough'], borough_stats['Taux de générosité moyen (%)'])
            
            # Colorer les barres selon le taux de générosité
            for i, bar in enumerate(bars):
                rate = borough_stats['Taux de générosité moyen (%)'].iloc[i]
                if rate < 15:
                    bar.set_color('red')
                elif rate < 20:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.title('Taux de générosité moyen par Borough')
            plt.ylabel('Taux de générosité (%)')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y', alpha=0.3)
            st.pyplot(fig)

# ------ INTERFACE PRINCIPALE ------

def display_header():
    st.markdown("<h1 style='text-align:center;'>Analyse des Données de Taxis Jaunes</h1>", unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")

# Widget d'upload de fichier
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

# Chargement des données
if uploaded_file is not None:
    with st.spinner('Chargement des données...'):
        df = process_uploaded_data(uploaded_file)
        st.sidebar.success("Données importées avec succès!")
else:
    st.sidebar.info("Aucun fichier importé. Utilisation de données démo.")
    df = generate_demo_data()

# Division train/test pour la modélisation
train_test_ratio = st.sidebar.slider("Pourcentage de données d'entraînement", 50, 90, 80)
datatrainset, datatestset = train_test_split(df, test_size=(100-train_test_ratio)/100, random_state=42)

# Afficher nombre de lignes
st.sidebar.metric("Nombre de lignes", len(df))

# Sélection de page
pages = ["📊 Analyse exploratoire des données", 
         "🛠️ Traitement des données",
         "🌍 Analyse géographique",
         "🤖 Modélisation"]

page = st.sidebar.selectbox("Sélectionnez une page", pages)

# Contexte dans le sidebar
with st.sidebar.expander("Contexte et Objectifs"):
    st.markdown("""
    <h5>Contexte</h5>
    <p>Les pourboires représentent une part significative des revenus des chauffeurs de taxi à New York.</p>
    
    <h5>Objectif Principal</h5>
    <p>Construire un modèle prédictif pour identifier les clients susceptibles de donner un pourboire ≥ 20% du montant total.</p>
    
    <h5>Hypothèses à Tester</h5>
    <ul>
        <li>H1 : Les clients payant par carte sont plus généreux que ceux payant en espèces.</li>
        <li>H2 : Les courses longues distances (> 10 miles) ont des pourboires proportionnellement plus élevés.</li>
        <li>H3 : Les trajets vers les aéroports (RatecodeID=2) ont un taux de générosité différent.</li>
    </ul>
    """, unsafe_allow_html=True)

# ------ PAGES DE L'APPLICATION ------

# Page d'analyse exploratoire des données
if page == "📊 Analyse exploratoire des données":
    display_header()
    
    # Modèle CSV (dans la barre latérale)
    st.sidebar.write("## Télécharger un modèle CSV")
    
    # Fonction pour créer un modèle CSV
    def create_template_csv():
        template_columns = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
            'passenger_count', 'trip_distance', 'RatecodeID', 
            'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 
            'payment_type', 'fare_amount', 'tip_amount', 
            'tolls_amount', 'total_amount'
        ]
        
        template_df = pd.DataFrame(columns=template_columns)
        
        # Ajouter des exemples
        example_data = {
            'VendorID': [1, 2],
            'tpep_pickup_datetime': ['2017-01-01 00:00:00', '2017-01-01 00:15:00'],
            'tpep_dropoff_datetime': ['2017-01-01 00:10:00', '2017-01-01 00:30:00'],
            'passenger_count': [1, 2],
            'trip_distance': [2.5, 3.0],
            'RatecodeID': [1, 1],
            'store_and_fwd_flag': ['N', 'N'],
            'PULocationID': [151, 239],
            'DOLocationID': [239, 246],
            'payment_type': [1, 2],
            'fare_amount': [12.0, 14.5],
            'tip_amount': [3.0, 0.0],
            'tolls_amount': [0.0, 0.0],
            'total_amount': [15.0, 14.5]
        }
        
        example_df = pd.DataFrame(example_data)
        template_df = pd.concat([template_df, example_df], ignore_index=True)
        
        return template_df
    
    # Créer et proposer le téléchargement du modèle
    template_df = create_template_csv()
    csv = template_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📥 Télécharger le modèle CSV",
        data=csv,
        file_name="yellow_taxi_template.csv",
        mime="text/csv"
    )
    
    # Instructions pour l'utilisation du modèle
    with st.sidebar.expander("Instructions d'utilisation du modèle"):
        st.markdown("""
        **Comment utiliser le modèle CSV:**
        1. Téléchargez le modèle CSV en cliquant sur le bouton ci-dessus
        2. Ouvrez-le dans Excel ou un éditeur de texte
        3. Remplissez-le avec vos données (respectez le format des colonnes)
        4. Importez votre fichier complété en utilisant l'option 'Importer un fichier CSV'
        
        **Note sur les formats:**
        - Dates: YYYY-MM-DD HH:MM:SS
        - VendorID: 1 ou 2
        - RatecodeID: 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Négocié, 6=Groupe
        - payment_type: 1=Carte, 2=Espèces, 3=Sans frais, 4=Dispute
        """)
    
    # Documentation sur la structure des données
    with st.sidebar.expander("Documentation des données"):
        st.markdown("""
        **Description des colonnes:**
        - **VendorID**: ID du fournisseur du système de taxi
        - **tpep_pickup_datetime**: Date/heure de prise en charge
        - **tpep_dropoff_datetime**: Date/heure de dépose
        - **passenger_count**: Nombre de passagers
        - **trip_distance**: Distance du trajet en miles
        - **RatecodeID**: Code tarifaire final
        - **store_and_fwd_flag**: Flag indiquant si le trajet a été stocké en mémoire (Y=oui, N=non)
        - **PULocationID**: ID du lieu de prise en charge
        - **DOLocationID**: ID du lieu de dépose
        - **payment_type**: Type de paiement
        - **fare_amount**: Montant du tarif
        - **tip_amount**: Montant du pourboire
        - **tolls_amount**: Montant des péages
        - **total_amount**: Montant total
        """)
    
    # Sections d'analyse
    eda_sections = [
        "Vue d'ensemble des données", 
        "Analyse Univariée", 
        "Analyse Bivariée",
        "Création et Analyse de la Variable Cible"
    ]
    
    eda_section = st.radio("Sélectionnez une section:", eda_sections, horizontal=True)
    
    # Vue d'ensemble des données
    if eda_section == "Vue d'ensemble des données":
        st.write("## Vue d'ensemble des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Aperçu des données")
            st.dataframe(df.head(10))
            
            st.write("### Informations sur les données")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.write("### Statistiques descriptives")
            st.dataframe(df.describe())
            
            st.write("### Types de données")
            dtypes_df = pd.DataFrame(df.dtypes, columns=['Type'])
            st.dataframe(dtypes_df)
            
            # Visualiser la répartition des types de données
            fig, ax = plt.subplots(figsize=(6, 4))
            df.dtypes.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
            ax.set_title("Répartition des types de données")
            ax.set_ylabel("")
            st.pyplot(fig)
        
        # Vérification des données manquantes
        st.write("### Analyse des données manquantes")
        
        missing_data = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Percentage': df.isnull().sum() / len(df) * 100
        })
        
        if missing_data['Missing Values'].sum() > 0:
            st.dataframe(missing_data[missing_data['Missing Values'] > 0])
            
            # Visualisation avec missingno
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.matrix(df, figsize=(10, 6), ax=ax)
            st.pyplot(fig)
        else:
            st.success("✅ Aucune donnée manquante détectée dans le jeu de données!")
    
    # Analyse Univariée
    elif eda_section == "Analyse Univariée":
        st.write("## Analyse Univariée")
        
        # Sélection du type de variable
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        col_type = st.radio("Type de variables:", ["Numériques", "Catégorielles", "Temporelles"], horizontal=True)
        
        # Analyse des variables numériques
        if col_type == "Numériques":
            column = st.selectbox("Sélectionnez une variable numérique:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(df[column].dropna(), kde=True)
                plt.title(f'Distribution de {column}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistiques
                stats_df = pd.DataFrame({
                    'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', '25%', '75%'],
                    'Valeur': [
                        df[column].mean(), 
                        df[column].median(), 
                        df[column].std(), 
                        df[column].min(), 
                        df[column].max(), 
                        df[column].quantile(0.25), 
                        df[column].quantile(0.75)
                    ]
                })
                st.dataframe(stats_df)
            
            with col2:
                # Boxplot
                fig = plt.figure(figsize=(10, 6))
                sns.boxplot(y=df[column].dropna())
                plt.title(f'Boxplot de {column}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Valeurs aberrantes
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
                st.write(f"**Valeurs aberrantes potentielles**: {len(outliers)} lignes")
                if len(outliers) > 0 and len(outliers) < 1000:
                    st.dataframe(outliers[[column]])
        
        # Analyse des variables catégorielles
        elif col_type == "Catégorielles":
            if not categorical_cols:
                st.warning("Aucune variable catégorielle détectée. Conversion de variables numériques...")
                conversion_options = st.multiselect(
                    "Sélectionnez des variables numériques à traiter comme catégorielles:",
                    numeric_cols,
                    default=['VendorID', 'RatecodeID', 'payment_type'] if all(x in numeric_cols for x in ['VendorID', 'RatecodeID', 'payment_type']) else []
                )
                categorical_cols = conversion_options
            
            if categorical_cols:
                column = st.selectbox("Sélectionnez une variable catégorielle:", categorical_cols)
                
                # Compter les occurrences
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = [column, 'Count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tableau de comptage
                    st.write(f"### Comptage des valeurs pour {column}")
                    st.dataframe(value_counts)
                    
                    # Pourcentages
                    value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
                    st.write("### Pourcentages")
                    st.dataframe(value_counts[['Percentage']])
                
                with col2:
                    # Visualisation
                    fig = plt.figure(figsize=(10, 6))
                    sns.countplot(y=df[column], order=df[column].value_counts().index)
                    plt.title(f'Comptage des valeurs pour {column}')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Camembert
                    fig = plt.figure(figsize=(10, 6))
                    plt.pie(value_counts['Count'], labels=value_counts[column], autopct='%1.1f%%')
                    plt.title(f'Répartition des valeurs pour {column}')
                    st.pyplot(fig)
            else:
                st.warning("Aucune variable catégorielle disponible pour l'analyse.")
        
        # Analyse des variables temporelles
        elif col_type == "Temporelles":
            if not datetime_cols:
                st.warning("Aucune variable temporelle détectée dans le jeu de données.")
            else:
                column = st.selectbox("Sélectionnez une variable temporelle:", datetime_cols)
                
                # Vérifier le format
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        time_series = pd.to_datetime(df[column])
                    except:
                        st.error(f"Impossible de convertir la colonne {column} en format date/heure.")
                        time_series = None
                else:
                    time_series = df[column]
                
                if time_series is not None:
                    st.write("### Composantes temporelles")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution par heure
                        fig = plt.figure(figsize=(10, 6))
                        hours = time_series.dt.hour.value_counts().sort_index()
                        plt.bar(hours.index, hours.values)
                        plt.title('Distribution par heure de la journée')
                        plt.xlabel('Heure')
                        plt.ylabel('Nombre de courses')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Distribution par jour de la semaine
                        fig = plt.figure(figsize=(10, 6))
                        days = time_series.dt.dayofweek.value_counts().sort_index()
                        plt.bar(days.index, days.values)
                        plt.title('Distribution par jour de la semaine')
                        plt.xlabel('Jour (0=Lundi, 6=Dimanche)')
                        plt.ylabel('Nombre de courses')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        # Distribution par mois
                        fig = plt.figure(figsize=(10, 6))
                        months = time_series.dt.month.value_counts().sort_index()
                        plt.bar(months.index, months.values)
                        plt.title('Distribution par mois')
                        plt.xlabel('Mois')
                        plt.ylabel('Nombre de courses')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Distribution cumulative
                        fig = plt.figure(figsize=(10, 6))
                        time_series.sort_values().value_counts().sort_index().cumsum().plot()
                        plt.title('Distribution cumulative dans le temps')
                        plt.xlabel('Date')
                        plt.ylabel('Nombre cumulé de courses')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
    # Analyse Bivariée
    elif eda_section == "Analyse Bivariée":
        st.write("## Analyse Bivariée avec la Variable Cible")
        
        # Sélectionner la variable à croiser avec la cible
        all_cols = [col for col in df.columns if col not in ['is_generous', 'tip_percentage']]
        selected_col = st.selectbox("Sélectionnez une variable à croiser avec 'is_generous':", all_cols)
        
        col1, col2 = st.columns(2)
        
        # Analyse différente selon le type de variable
        if df[selected_col].dtype == 'object' or df[selected_col].nunique() < 10:
            # Variable catégorielle
            with col1:
                st.write(f"### Taux de générosité par {selected_col}")
                
                # Taux de générosité par catégorie
                generosity_by_category = df.groupby(selected_col)['is_generous'].mean().reset_index()
                generosity_by_category.columns = [selected_col, 'Taux de générosité']
                generosity_by_category['Taux de générosité'] *= 100
                
                # Trier par taux
                generosity_by_category = generosity_by_category.sort_values('Taux de générosité', ascending=False)
                
                st.dataframe(generosity_by_category)
                
                # Graphique
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(x='Taux de générosité', y=selected_col, data=generosity_by_category)
                plt.title(f'Taux de générosité (%) par {selected_col}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.write(f"### Distribution de {selected_col} par classe")
                
                # Distribution par classe
                category_by_class = pd.crosstab(df[selected_col], df['is_generous'])
                category_by_class.columns = ['Non généreux', 'Généreux']
                
                # Pourcentages
                category_by_class_pct = category_by_class.div(category_by_class.sum(axis=1), axis=0) * 100
                
                st.dataframe(category_by_class)
                
                # Graphique à barres empilées
                fig = plt.figure(figsize=(10, 6))
                category_by_class_pct.plot(kind='barh', stacked=True)
                plt.title(f'Distribution de {selected_col} par classe (en %)')
                plt.legend(title='Classe')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Test Chi²
                contingency_table = pd.crosstab(df[selected_col], df['is_generous'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                
                st.write("### Test statistique (Chi²)")
                st.write(f"Valeur Chi² : {chi2:.4f}")
                st.write(f"p-value : {p:.4f}")
                
                if p < 0.05:
                    st.success(f"✅ Relation significative entre {selected_col} et la générosité (p < 0.05)")
                else:
                    st.info(f"ℹ️ Pas de relation significative détectée (p > 0.05)")
        
        else:
            # Variable numérique
            with col1:
                st.write(f"### Distribution de {selected_col} par classe")
                
                # Boxplot par classe
                fig = plt.figure(figsize=(10, 6))
                sns.boxplot(x='is_generous', y=selected_col, data=df)
                plt.title(f'Distribution de {selected_col} par classe')
                plt.xlabel('Is Generous (0=Non, 1=Oui)')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistiques par classe
                stats_by_class = df.groupby('is_generous')[selected_col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
                stats_by_class['is_generous'] = stats_by_class['is_generous'].map({0: 'Non généreux', 1: 'Généreux'})
                st.dataframe(stats_by_class)
            
            with col2:
                st.write(f"### Histogrammes de {selected_col} par classe")
                
                # Histogrammes superposés par classe
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=selected_col, hue='is_generous', element='step', stat='density', common_norm=False)
                plt.title(f'Distribution de {selected_col} par classe')
                plt.legend(['Non généreux', 'Généreux'])
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Test statistique
                non_generous = df[df['is_generous'] == 0][selected_col].dropna()
                generous = df[df['is_generous'] == 1][selected_col].dropna()
                
                st.write("### Test statistique")
                
                try:
                    t_stat, p_value = stats.ttest_ind(non_generous, generous, equal_var=False)
                    test_name = "t-test"
                except:
                    try:
                        t_stat, p_value = stats.mannwhitneyu(non_generous, generous)
                        test_name = "Mann-Whitney U"
                    except:
                        st.error("Impossible de réaliser le test statistique.")
                        t_stat, p_value = 0, 1
                        test_name = "N/A"
                
                st.write(f"Test utilisé : {test_name}")
                st.write(f"Statistique : {t_stat:.4f}")
                st.write(f"p-value : {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success(f"✅ Différence significative de {selected_col} entre les classes (p < 0.05)")
                else:
                    st.info(f"ℹ️ Pas de différence significative entre les classes (p > 0.05)")
        
        # Test des hypothèses
        st.write("## Test des hypothèses")
        hypothesis_tabs = st.tabs(["H1: Mode de paiement", "H2: Distance", "H3: Aéroports"])
        
        # H1: Mode de paiement
        with hypothesis_tabs[0]:
            st.write("### H1: Les clients payant par carte sont plus généreux")
            
            if 'payment_type' in df.columns:
                # Taux de générosité par type de paiement
                payment_generosity = df.groupby('payment_type')['is_generous'].mean().reset_index()
                payment_generosity.columns = ['Type de paiement', 'Taux de générosité']
                payment_generosity['Taux de générosité'] *= 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(payment_generosity)
                    
                    st.info("""
                    Types de paiement:
                    1 = Carte de crédit
                    2 = Espèces
                    3 = Sans frais
                    4 = Contestation
                    """)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Type de paiement', y='Taux de générosité', data=payment_generosity)
                    plt.title('Taux de générosité par type de paiement')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Test statistique
                card = df[df['payment_type'] == 1]['is_generous']
                cash = df[df['payment_type'] == 2]['is_generous']
                
                st.write("### Statistiques:")
                st.write(f"Nombre de paiements par carte: {len(card)}")
                st.write(f"Clients généreux (carte): {card.sum()} ({card.mean()*100:.2f}%)")
                st.write(f"Nombre de paiements en espèces: {len(cash)}")
                st.write(f"Clients généreux (espèces): {cash.sum()} ({cash.mean()*100:.2f}%)")
                
                # Test Chi²
                if len(card) > 0 and len(cash) > 0:
                    try:
                        contingency = pd.crosstab(
                            df[df['payment_type'].isin([1, 2])]['payment_type'],
                            df[df['payment_type'].isin([1, 2])]['is_generous']
                        )
                        
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Carte vs Espèces (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                if df[df['payment_type'] == 1]['is_generous'].mean() > df[df['payment_type'] == 2]['is_generous'].mean():
                                    st.success("✅ H1 CONFIRMÉE: Les clients payant par carte sont significativement plus généreux")
                                else:
                                    st.error("❌ H1 REJETÉE: Les clients payant par carte sont significativement moins généreux")
                            else:
                                st.info("ℹ️ H1 INDÉTERMINÉE: Pas de différence significative entre carte et espèces")
                    except Exception as e:
                        st.error(f"Erreur lors du test Chi²: {str(e)}")
            else:
                st.error("La colonne 'payment_type' n'existe pas dans le jeu de données.")
        
        # H2: Distance
        with hypothesis_tabs[1]:
            st.write("### H2: Les courses longues distances ont des pourboires proportionnellement plus élevés")
            
            if 'trip_distance' in df.columns:
                # Seuil de longue distance (10 miles)
                long_distance_threshold = 10
                
                # Catégorie de distance
                df['distance_cat'] = df['trip_distance'].apply(
                    lambda x: 'Long (>10 miles)' if x > long_distance_threshold else 'Court (≤10 miles)'
                )
                
                # Taux de générosité par catégorie
                distance_generosity = df.groupby('distance_cat')['is_generous'].agg(['mean', 'count']).reset_index()
                distance_generosity.columns = ['Catégorie de distance', 'Taux de générosité', 'Nombre de courses']
                distance_generosity['Taux de générosité'] *= 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(distance_generosity)
                    
                    tip_stats_by_distance = df.groupby('distance_cat')['tip_percentage'].agg(['mean', 'median']).reset_index()
                    tip_stats_by_distance.columns = ['Catégorie de distance', 'Pourboire moyen (%)', 'Pourboire médian (%)']
                    st.dataframe(tip_stats_by_distance)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Catégorie de distance', y='Taux de générosité', data=distance_generosity)
                    plt.title('Taux de générosité par catégorie de distance')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Boxplot du pourcentage de pourboire
                    fig = plt.figure(figsize=(10, 6))
                    sns.boxplot(x='distance_cat', y='tip_percentage', data=df)
                    plt.title('Distribution du pourcentage de pourboire par catégorie de distance')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Statistiques et test
                long_distance = df[df['trip_distance'] > long_distance_threshold]['is_generous']
                short_distance = df[df['trip_distance'] <= long_distance_threshold]['is_generous']
                
                st.write("### Statistiques:")
                st.write(f"Courses longue distance: {len(long_distance)}")
                st.write(f"Clients généreux (longue): {long_distance.sum()} ({long_distance.mean()*100:.2f}%)")
                st.write(f"Courses courte distance: {len(short_distance)}")
                st.write(f"Clients généreux (courte): {short_distance.sum()} ({short_distance.mean()*100:.2f}%)")
                
                # Test statistique
                if len(long_distance) > 0 and len(short_distance) > 0:
                    try:
                        contingency = pd.crosstab(df['distance_cat'], df['is_generous'])
                        
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Longue vs Courte distance (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                if long_distance.mean() > short_distance.mean():
                                    st.success("✅ H2 CONFIRMÉE: Les courses longues distances ont significativement plus de pourboires généreux")
                                else:
                                    st.error("❌ H2 REJETÉE: Les courses longues distances ont significativement moins de pourboires généreux")
                            else:
                                st.info("ℹ️ H2 INDÉTERMINÉE: Pas de différence significative entre longue et courte distance")
                        else:
                            st.warning("La table de contingence n'a pas la bonne forme")
                    except Exception as e:
                        st.error(f"Erreur lors du test: {str(e)}")
            else:
                st.error("La colonne 'trip_distance' n'existe pas dans le jeu de données.")
        
        # H3: Aéroports
        with hypothesis_tabs[2]:
            st.write("### H3: Les trajets vers les aéroports ont un taux de générosité différent")
            
            if 'RatecodeID' in df.columns:
                # Trajets vers aéroports: RatecodeID=2
                df['airport_trip'] = df['RatecodeID'].apply(lambda x: 'Aéroport' if x == 2 else 'Non-aéroport')
                
                # Taux de générosité
                airport_generosity = df.groupby('airport_trip')['is_generous'].agg(['mean', 'count']).reset_index()
                airport_generosity.columns = ['Type de trajet', 'Taux de générosité', 'Nombre de courses']
                airport_generosity['Taux de générosité'] *= 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(airport_generosity)
                    
                    tip_stats_by_airport = df.groupby('airport_trip')['tip_percentage'].agg(['mean', 'median']).reset_index()
                    tip_stats_by_airport.columns = ['Type de trajet', 'Pourboire moyen (%)', 'Pourboire médian (%)']
                    st.dataframe(tip_stats_by_airport)
                    
                    st.info("""
                    Codes RatecodeID:
                    1 = Standard
                    2 = JFK
                    3 = Newark
                    4 = Nassau ou Westchester
                    5 = Négocié
                    6 = Trajet groupé
                    """)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Type de trajet', y='Taux de générosité', data=airport_generosity)
                    plt.title('Taux de générosité par type de trajet')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    fig = plt.figure(figsize=(10, 6))
                    sns.boxplot(x='airport_trip', y='tip_percentage', data=df)
                    plt.title('Distribution du pourcentage de pourboire par type de trajet')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Statistiques et test
                airport = df[df['RatecodeID'] == 2]['is_generous']
                non_airport = df[df['RatecodeID'] != 2]['is_generous']
                
                st.write("### Statistiques:")
                st.write(f"Trajets vers l'aéroport: {len(airport)}")
                st.write(f"Clients généreux (aéroport): {airport.sum()} ({airport.mean()*100:.2f}%" if len(airport) > 0 else "Pourcentage: N/A")
                st.write(f"Trajets hors aéroport: {len(non_airport)}")
                st.write(f"Clients généreux (non-aéroport): {non_airport.sum()} ({non_airport.mean()*100:.2f}%)")
                
                # Test statistique
                if len(airport) > 0 and len(non_airport) > 0:
                    try:
                        contingency = pd.crosstab(df['airport_trip'], df['is_generous'])
                        
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Aéroport vs Non-aéroport (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                if airport.mean() > non_airport.mean():
                                    st.success("✅ H3 CONFIRMÉE: Les trajets vers les aéroports ont significativement plus de pourboires généreux")
                                else:
                                    st.error("❌ H3 REJETÉE: Les trajets vers les aéroports ont significativement moins de pourboires généreux")
                            else:
                                st.info("ℹ️ H3 INDÉTERMINÉE: Pas de différence significative entre les trajets")
                        else:
                            st.warning("La table de contingence n'a pas la bonne forme")
                    except Exception as e:
                        st.error(f"Erreur lors du test: {str(e)}")
            else:
                st.error("La colonne 'RatecodeID' n'existe pas dans le jeu de données.")
    
    # Création et Analyse de la Variable Cible
    elif eda_section == "Création et Analyse de la Variable Cible":
        st.write("## Création et Analyse de la Variable Cible")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Distribution du pourcentage de pourboire")
            
            # Histogramme du pourcentage de pourboire
            fig = plt.figure(figsize=(10, 6))
            plt.hist(df['tip_percentage'].clip(0, 50), bins=50)  # Clip pour limiter l'effet des outliers
            plt.axvline(x=20, color='red', linestyle='--', label='Seuil 20%')
            plt.title('Distribution du pourcentage de pourboire')
            plt.xlabel('Pourcentage de pourboire')
            plt.ylabel('Nombre de courses')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Statistiques
            tip_stats = pd.DataFrame({
                'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', '% ≥ 20%'],
                'Valeur': [
                    df['tip_percentage'].mean(), 
                    df['tip_percentage'].median(), 
                    df['tip_percentage'].std(), 
                    df['tip_percentage'].min(), 
                    df['tip_percentage'].max(),
                    (df['is_generous'].sum() / len(df)) * 100
                ]
            })
            st.dataframe(tip_stats)
        
        with col2:
            st.write("### Distribution de la variable cible 'is_generous'")
            
            # Camembert
            target_counts = df['is_generous'].value_counts()
            fig = plt.figure(figsize=(10, 6))
            plt.pie(target_counts, labels=['Non généreux', 'Généreux'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
            plt.title('Distribution de la variable cible is_generous')
            st.pyplot(fig)
            
            # Équilibre des classes
            st.write("### Équilibre des classes")
            class_ratio = target_counts[1] / target_counts[0] if 0 in target_counts and 1 in target_counts else 0
            
            st.metric("Ratio Généreux / Non généreux", f"{class_ratio:.4f}")
            
            if class_ratio < 0.2:
                st.warning("⚠️ Le jeu de données est fortement déséquilibré. Envisagez des techniques de rééquilibrage pour la modélisation.")
            elif class_ratio < 0.5:
                st.info("ℹ️ Le jeu de données présente un déséquilibre modéré.")
            else:
                st.success("✅ Le jeu de données est relativement équilibré.")

# Page de traitement des données
elif page == "🛠️ Traitement des données":
    display_header()
    
    # Sous-sections
    processing_sections = ["Analyse des outliers", 
                           "Analyse des Corrélations", 
                           "Pré-Traitement"]
    
    processing_section = st.radio("Sélectionnez une section:", processing_sections, horizontal=True)
    
    # Analyse des outliers
    if processing_section == "Analyse des outliers":
        st.write("## Analyse des outliers")
        
        # Variables numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclure certaines colonnes
        cols_to_exclude = ['Unnamed: 0', 'is_generous', 'tip_percentage']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        selected_col = st.selectbox("Sélectionnez une variable pour l'analyse des outliers:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[selected_col])
            plt.title(f'Boxplot de {selected_col} - Détection des outliers')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Statistiques
            st.write("### Statistiques descriptives")
            
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            stats_df = pd.DataFrame({
                'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 
                               'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Limite inf.', 'Limite sup.'],
                'Valeur': [
                    df[selected_col].mean(), 
                    df[selected_col].median(), 
                    df[selected_col].std(), 
                    df[selected_col].min(), 
                    df[selected_col].max(),
                    Q1, Q3, IQR, lower_bound, upper_bound
                ]
            })
            st.dataframe(stats_df)
            
            # Détection des outliers
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_percent = (outliers_count / len(df)) * 100
            
            st.write(f"### Outliers détectés: {outliers_count} ({outliers_percent:.2f}%)")
            
            # Distribution avec limites
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(df[selected_col], kde=True)
            plt.axvline(x=lower_bound, color='r', linestyle='--', label=f'Limite inf. ({lower_bound:.2f})')
            plt.axvline(x=upper_bound, color='r', linestyle='--', label=f'Limite sup. ({upper_bound:.2f})')
            plt.title(f'Distribution de {selected_col} avec limites d\'outliers')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Analyse des Corrélations
    elif processing_section == "Analyse des Corrélations":
        st.write("## Analyse des Corrélations")
        
        # Variables numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclure certaines colonnes
        cols_to_exclude = ['Unnamed: 0']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        # Matrice de corrélation
        correlation_matrix = df[numeric_cols].corr(method='pearson')
        
        # Heatmap
        fig = plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matrice de corrélation (Pearson)')
        st.pyplot(fig)
        
        # Valeurs de corrélation
        st.write("### Valeurs de toutes les corrélations")
        
        # Transformer la matrice en liste
        corr_list = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_list.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Corrélation': correlation_matrix.iloc[i, j]
                })
        
        # Trier par valeur absolue
        corr_df = pd.DataFrame(corr_list)
        corr_df['Corrélation absolue'] = corr_df['Corrélation'].abs()
        corr_df = corr_df.sort_values('Corrélation absolue', ascending=False)
        
        st.dataframe(corr_df)
        
        # Corrélations avec la cible
        if 'is_generous' in numeric_cols:
            st.write("### Corrélations avec la variable cible 'is_generous'")
            
            target_corr = correlation_matrix['is_generous'].sort_values(ascending=False).drop('is_generous')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(target_corr)
            
            with col2:
                fig = plt.figure(figsize=(10, 8))
                target_corr.plot(kind='bar')
                plt.title('Corrélations avec is_generous')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        # PCA pour visualiser les relations
        st.write("### Analyse en Composantes Principales (ACP)")
        
        # Sélection des variables pour l'ACP
        pca_cols = [col for col in numeric_cols if col not in ['is_generous', 'tip_percentage']]
        
        if len(pca_cols) >= 2:
            # Standardiser
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[pca_cols])
            
            # ACP
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(df_scaled)
            
            # DataFrame pour visualisation
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            
            # Ajouter la cible si disponible
            if 'is_generous' in df.columns:
                pca_df['is_generous'] = df['is_generous'].values
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plt.figure(figsize=(10, 8))
                if 'is_generous' in pca_df.columns:
                    sns.scatterplot(x='PC1', y='PC2', hue='is_generous', data=pca_df, alpha=0.7)
                else:
                    sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
                plt.title('Visualisation PCA')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Variance expliquée
                explained_variance = pca.explained_variance_ratio_
                st.write("### Variance expliquée par les composantes")
                
                variance_df = pd.DataFrame({
                    'Composante': ['PC1', 'PC2'],
                    'Variance expliquée (%)': explained_variance * 100
                })
                st.dataframe(variance_df)
                
                # Graphique variance
                fig = plt.figure(figsize=(10, 6))
                plt.bar(['PC1', 'PC2'], explained_variance)
                plt.axhline(y=0.7, color='r', linestyle='-', label='Seuil 70%')
                plt.title('Variance expliquée par composante')
                plt.ylabel('Proportion de variance expliquée')
                plt.grid(True, alpha=0.3)
                plt.legend()
                st.pyplot(fig)
            
            # Contributions des variables
            st.write("### Contribution des variables aux composantes principales")
            
            loadings = pca.components_.T
            loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=pca_cols)
            
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Contribution des variables aux composantes principales')
            st.pyplot(fig)
        else:
            st.warning("Pas assez de variables numériques pour réaliser une ACP.")
    
    # Pré-Traitement
    elif processing_section == "Pré-Traitement":
        st.write("## Pré-Traitement des données")
        
        # Options de prétraitement
        preprocessing_options = st.multiselect(
            "Sélectionnez les opérations de pré-traitement à appliquer:",
            ["Nettoyage des colonnes inutiles",
             "Encodage des variables catégorielles", 
             "Feature Engineering", 
             "Standardisation"]
        )
        
        # DataFrame pour le prétraitement
        df_processed = df.copy()
        transformations_applied = []
        
        # 1. Nettoyage
        if "Nettoyage des colonnes inutiles" in preprocessing_options:
            st.write("### Nettoyage des colonnes inutiles")
            
            n_before = len(df_processed)
            df_processed = nettoyage(df_processed)
            n_after = len(df_processed)
            
            # Résumé du nettoyage
            st.write(f"Taille du DataFrame avant nettoyage: {n_before}")
            st.write(f"Taille du DataFrame après nettoyage: {n_after}")
            st.write(f"Lignes supprimées: {n_before - n_after}")
            
            # Liste des transformations
            transformations_applied.append(f"Nettoyage: {n_before - n_after} lignes supprimées")
            
            # Aperçu des données nettoyées
            st.write("#### Aperçu des données nettoyées")
            st.dataframe(df_processed.head())
        
        # 2. Encodage
        if "Encodage des variables catégorielles" in preprocessing_options:
            st.write("### Encodage des variables catégorielles")
            
            # Compter les variables catégorielles avant
            cat_cols_before = df_processed.select_dtypes(include='object').columns.tolist()
            
            # Appliquer l'encodage
            df_processed = encodage(df_processed)
            
            # Compter les variables catégorielles après
            cat_cols_after = df_processed.select_dtypes(include='object').columns.tolist()
            
            # Liste des transformations
            transformations_applied.append(f"Encodage: {len(cat_cols_before) - len(cat_cols_after)} colonnes encodées")
            
            # Aperçu des données encodées
            st.write("#### Aperçu des données encodées")
            st.dataframe(df_processed.head())
            
            # Informations du DataFrame
            buffer = io.StringIO()
            df_processed.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # 3. Feature Engineering
        if "Feature Engineering" in preprocessing_options:
            st.write("### Feature Engineering")
            
            n_before = len(df_processed)
            df_processed = FeatureEngineering(df_processed)
            n_after = len(df_processed)
            
            # Liste des transformations
            new_cols = []
            if 'trip_duration' in df_processed.columns and 'trip_duration' not in df.columns:
                new_cols.append('trip_duration')
            if 'speed_mph' in df_processed.columns and 'speed_mph' not in df.columns:
                new_cols.append('speed_mph')
            
            if new_cols:
                transformations_applied.append(f"Feature Engineering: Création des colonnes {', '.join(new_cols)}")
            
            if n_before > n_after:
                transformations_applied.append(f"Feature Engineering: {n_before - n_after} lignes filtrées (vitesses aberrantes)")
            
            # Aperçu des nouvelles features
            st.write("#### Nouvelles features créées")
            
            if 'trip_duration' in df_processed.columns and 'speed_mph' in df_processed.columns:
                st.dataframe(df_processed[['trip_duration', 'speed_mph']].describe())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plt.figure(figsize=(10, 6))
                    sns.histplot(df_processed['trip_duration'].clip(0, 60), kde=True)
                    plt.title('Distribution de la durée de trajet (minutes)')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.histplot(df_processed['speed_mph'].clip(0, 50), kde=True)
                    plt.title('Distribution de la vitesse (mph)')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
            else:
                st.warning("Les nouvelles features n'ont pas été créées.")
        
        # 4. Standardisation
        if "Standardisation" in preprocessing_options:
            st.write("### Standardisation des données numériques")
            
            # Variables numériques
            numeric_columns = df_processed.select_dtypes(include=['float', 'int']).columns.tolist()
            
            # Colonnes à exclure
            exclude_from_scaling = ['Unnamed: 0', 'is_generous', 'VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']
            numeric_columns = [col for col in numeric_columns if col not in exclude_from_scaling and col in df_processed.columns]
            
            st.write("#### Colonnes à standardiser:")
            st.write(", ".join(numeric_columns))
            
            if numeric_columns:
                # Sauvegarde pour comparaison
                df_before = df_processed[numeric_columns].head().copy()
                
                # Standardisation
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
                
                transformations_applied.append(f"Standardisation: {len(numeric_columns)} colonnes numériques")
                
                # Résultats
                st.write("#### Comparaison avant/après standardisation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Avant standardisation")
                    st.dataframe(df_before)
                
                with col2:
                    st.write("Après standardisation")
                    st.dataframe(df_processed[numeric_columns].head())
                
                # Visualisation
                st.write("#### Visualisation de la standardisation")
                
                # Afficher quelques colonnes standardisées
                num_cols_to_show = min(4, len(numeric_columns))
                cols_to_show = numeric_columns[:num_cols_to_show]
                
                col1, col2 = st.columns(2)
                
                for i, col_name in enumerate(cols_to_show):
                    with col1 if i % 2 == 0 else col2:
                        fig = plt.figure(figsize=(10, 6))
                        sns.histplot(df_processed[col_name], kde=True)
                        plt.title(f'Distribution de {col_name} après standardisation')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
            else:
                st.warning("Aucune colonne numérique disponible pour la standardisation.")
        
        # Résumé des transformations
        if transformations_applied:
            st.write("## Résumé des transformations appliquées")
            
            for i, transformation in enumerate(transformations_applied, 1):
                st.write(f"{i}. {transformation}")
            
            # Aperçu final
            st.write("## Aperçu des données transformées")
            st.dataframe(df_processed.head())
            
            # Informations
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Dimensions")
                st.write(f"Lignes: {df_processed.shape[0]}")
                st.write(f"Colonnes: {df_processed.shape[1]}")
            
            with col2:
                st.write("### Types de données")
                dtypes_count = df_processed.dtypes.value_counts()
                
                fig = plt.figure(figsize=(8, 6))
                plt.pie(dtypes_count, labels=dtypes_count.index, autopct='%1.1f%%')
                plt.title('Répartition des types de données après transformation')
                st.pyplot(fig)
            
            # Téléchargement
            st.download_button(
                label="Télécharger les données transformées (CSV)",
                data=df_processed.to_csv(index=False).encode('utf-8'),
                file_name="yellow_taxi_processed.csv",
                mime="text/csv"
            )
            
            # Pour conserver les données prétraitées en mémoire
            st.session_state['preprocessed_data'] = df_processed
        else:
            st.info("Aucune transformation n'a été appliquée aux données.")

# Page d'analyse géographique (APRÈS le traitement des données)
elif page == "🌍 Analyse géographique":
    display_header()
    
    # Vérifier si des données prétraitées sont disponibles
    if 'preprocessed_data' in st.session_state:
        df_processed = st.session_state['preprocessed_data']
        st.success("Utilisation des données prétraitées!")
    else:
        # Si non, prétraiter les données avant l'analyse spatiale
        st.info("Prétraitement des données avant l'analyse géographique...")
        df_processed = preprocess_data(df)
        st.success("Prétraitement terminé!")
    
    # Afficher l'analyse spatiale avec les données prétraitées
    display_spatial_analysis(df_processed)

# Page de modélisation
elif page == "🤖 Modélisation":
    display_header()
    
    # Vérifier si des données prétraitées sont disponibles, sinon prétraiter
    if 'preprocessed_data' in st.session_state:
        datatrainset_clean = st.session_state['preprocessed_data']
        st.success("Utilisation des données prétraitées de la section 'Traitement des données'!")
    else:
        # Appliquer le prétraitement
        st.write("### Prétraitement des données")
        datatrainset_clean = preprocess_data(df)
        st.success("Prétraitement terminé avec succès!")
    
    # Vérifier la présence de la variable cible
    if 'is_generous' not in datatrainset_clean.columns:
        st.error("La variable cible 'is_generous' n'a pas été créée. Vérifiez que le DataFrame contient les colonnes tip_amount et fare_amount.")
    else:
        # Fonctions utilitaires pour la modélisation
        def safe_default_features(df):
            """Retourne une liste de features disponibles dans le dataframe"""
            suggested_features = ['pickup_week', 'pickup_month', 'fare_amount',
                                 'trip_distance', 'RatecodeID', 'payment_type', 'speed_mph']
            
            # Features existantes
            available_features = [f for f in suggested_features if f in df.columns]
            
            # Si pas assez de features disponibles, ajouter d'autres colonnes numériques
            if len(available_features) < 3:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cols_to_exclude = ['is_generous', 'tip_percentage', 'tip_amount', 'Unnamed: 0']
                additional_features = [col for col in numeric_cols if col not in cols_to_exclude and col not in available_features]
                available_features.extend(additional_features[:min(5, len(additional_features))])
            
            return available_features[:min(7, len(available_features))]
        
        # Sections de modélisation
        modeling_sections = ["Sélection de features", 
                            "Algorithmes de classification", 
                            "Optimisation d'Hyperparamètres",
                            "Comparaison des Modèles"]
        
        modeling_section = st.radio("Sélectionnez une section:", modeling_sections, horizontal=True)
        
        # Features par défaut
        default_features = safe_default_features(datatrainset_clean)
        st.info(f"Features disponibles par défaut : {', '.join(default_features)}")
        
        # Section 1: Sélection de features
        if modeling_section == "Sélection de features":
            st.write("## Sélection des features pour la modélisation")
            
            # Variables numériques et catégorielles
            numeric_cols = datatrainset_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = datatrainset_clean.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Exclure certaines colonnes
            cols_to_exclude = ['is_generous', 'tip_percentage', 'tip_amount', 'Unnamed: 0']
            numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
            
            # Interface de sélection
            st.write("### Sélection manuelle des features")
            
            selected_numeric_features = st.multiselect(
                "Sélectionnez les features numériques:",
                numeric_cols,
                default=[f for f in default_features if f in numeric_cols]
            )
            
            selected_categorical_features = st.multiselect(
                "Sélectionnez les features catégorielles:",
                categorical_cols,
                default=[f for f in default_features if f in categorical_cols]
            )
            
            # Combiner les features sélectionnées
            selected_features = selected_numeric_features + selected_categorical_features
            
            if len(selected_features) == 0:
                st.warning("Veuillez sélectionner au moins une feature pour la modélisation.")
            else:
                st.success(f"{len(selected_features)} features sélectionnées.")
                
                # Sélection automatique
                st.write("### Sélection automatique des features")
                
                # Préparer les données
                X = datatrainset_clean[selected_features].copy()
                y = datatrainset_clean['is_generous'].copy()
                
                # RFE
                if st.checkbox("Utiliser la Recursive Feature Elimination (RFE)"):
                    st.write("#### Recursive Feature Elimination")
                    
                    n_features_to_select = st.slider(
                        "Nombre de features à sélectionner:", 
                        min_value=1, 
                        max_value=min(20, len(X.columns)), 
                        value=min(5, len(X.columns))
                    )
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # RFE
                    base_model = RandomForestClassifier(random_state=42)
                    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
                    rfe.fit(X_train, y_train)
                    
                    # Features sélectionnées
                    selected_features_rfe = X.columns[rfe.support_]
                    st.write(f"Features sélectionnées par RFE: {', '.join(selected_features_rfe)}")
                    
                    # Tableau de ranking
                    feature_ranking = pd.DataFrame({
                        'Feature': X.columns,
                        'Ranking': rfe.ranking_
                    }).sort_values('Ranking')
                    st.dataframe(feature_ranking)
                    
                    # Plot d'importance
                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(feature_ranking['Feature'][:10], 1/feature_ranking['Ranking'][:10])
                    plt.title('Importance des features selon RFE')
                    plt.xlabel('Importance relative (1/Ranking)')
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)
                
                # Random Forest Importance
                if st.checkbox("Utiliser l'importance des features avec Random Forest"):
                    st.write("#### Importance des features avec Random Forest")
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # Entrainement
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)
                    
                    # Importance des features
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.dataframe(feature_importance)
                    
                    # Plot d'importance
                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
                    plt.title('Importance des features selon Random Forest')
                    plt.xlabel('Importance')
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)
                    
                    # Sélection par seuil
                    importance_threshold = st.slider(
                        "Seuil d'importance pour la sélection:", 
                        min_value=0.0, 
                        max_value=float(feature_importance['Importance'].max()), 
                        value=0.01,
                        step=0.01
                    )
                    
                    selected_features_rf = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
                    st.write(f"Features sélectionnées (importance > {importance_threshold}): {', '.join(selected_features_rf)}")
                
                # Enregistrer les features
                if st.button("Utiliser ces features pour la modélisation"):
                    st.session_state['selected_features'] = selected_features
                    st.success(f"Les {len(selected_features)} features sélectionnées ont été enregistrées!")
                    
                    # Aperçu des données avec les features
                    st.write("### Aperçu des données avec les features sélectionnées")
                    st.dataframe(datatrainset_clean[selected_features + ['is_generous']].head())
        
        # Section 2: Algorithmes de classification
        elif modeling_section == "Algorithmes de classification":
            st.write("## Algorithmes de classification")
            
            # Vérifier si des features ont été sélectionnées
            if 'selected_features' not in st.session_state:
                st.info("Aucune feature sélectionnée. Utilisation des features par défaut.")
                selected_features = safe_default_features(datatrainset_clean)
                st.session_state['selected_features'] = selected_features
            else:
                selected_features = st.session_state['selected_features']
                # Vérifier les features existantes
                missing_features = [f for f in selected_features if f not in datatrainset_clean.columns]
                if missing_features:
                    st.warning(f"Features manquantes: {', '.join(missing_features)}")
                    selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                    if not selected_features:
                        selected_features = safe_default_features(datatrainset_clean)
                    st.session_state['selected_features'] = selected_features
            
            st.write(f"Features utilisées: {', '.join(selected_features)}")
            
            # Préparation des données
            X = datatrainset_clean[selected_features].copy()
            y = datatrainset_clean['is_generous'].copy()
            
            # Split
            test_size = st.slider("Proportion du jeu de test:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.write(f"Dimensions: X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Choix de l'algorithme
            algorithm = st.selectbox(
                "Choisissez un algorithme de classification:",
                ["Régression Logistique", "Random Forest", "Support Vector Machine (SVM)", 
                 "K-Nearest Neighbors (KNN)", "Réseau de Neurones", "Arbre de Décision", 
                 "Naive Bayes"]
            )
            
            # Entraînement
            if st.button("Entraîner le modèle"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialisation
                metrics = {}
                model = None
                
                # Algorithmes
                if algorithm == "Régression Logistique":
                    status_text.text("Entraînement de la Régression Logistique...")
                    progress_bar.progress(25)
                    
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': pd.DataFrame({
                            'Feature': X.columns,
                            'Coefficient': model.coef_[0]
                        }).sort_values('Coefficient', ascending=False),
                        'cmap': 'Blues'
                    }
                
                elif algorithm == "Random Forest":
                    status_text.text("Entraînement du Random Forest...")
                    progress_bar.progress(25)
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False),
                        'cmap': 'Greens'
                    }
                
                elif algorithm == "Support Vector Machine (SVM)":
                    status_text.text("Entraînement du SVM...")
                    progress_bar.progress(25)
                    
                    model = svm.SVC(probability=True, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': None,  # SVM n'a pas d'importance directe
                        'cmap': 'Blues'
                    }
                
                elif algorithm == "K-Nearest Neighbors (KNN)":
                    status_text.text("Entraînement du KNN...")
                    progress_bar.progress(25)
                    
                    model = KNeighborsClassifier(n_neighbors=5)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': None,  # KNN n'a pas d'importance directe
                        'cmap': 'Oranges'
                    }
                
                elif algorithm == "Réseau de Neurones":
                    status_text.text("Entraînement du Réseau de Neurones...")
                    progress_bar.progress(25)
                    
                    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': None,  # MLP n'a pas d'importance directe
                        'cmap': 'Purples'
                    }
                
                elif algorithm == "Arbre de Décision":
                    status_text.text("Entraînement de l'Arbre de Décision...")
                    progress_bar.progress(25)
                    
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Règles de l'arbre
                    tree_rules = export_text(model, feature_names=list(X.columns))
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False),
                        'cmap': 'YlGnBu',
                        'tree_rules': tree_rules
                    }
                
                elif algorithm == "Naive Bayes":
                    status_text.text("Entraînement du Naive Bayes...")
                    progress_bar.progress(25)
                    
                    model = GaussianNB()
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Métriques
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred),
                        'feature_importance': None,  # NB n'a pas d'importance directe
                        'cmap': 'OrRd'
                    }
                    
                    # Parameters du modèle
                    if hasattr(model, 'theta_') and hasattr(model, 'var_'):
                        metrics['theta'] = pd.DataFrame(model.theta_, columns=X.columns, index=['Classe 0', 'Classe 1'])
                        metrics['var'] = pd.DataFrame(model.var_, columns=X.columns, index=['Classe 0', 'Classe 1'])
                
                # Enregistrer dans la session
                st.session_state['current_model'] = model
                st.session_state['current_model_name'] = algorithm
                st.session_state['current_metrics'] = metrics
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                # Enregistrer dans la liste des modèles
                if 'models_results' not in st.session_state:
                    st.session_state['models_results'] = []
                
                # Vérifier si le modèle existe déjà
                existing_model_idx = None
                for i, result in enumerate(st.session_state['models_results']):
                    if result['Model'] == algorithm:
                        existing_model_idx = i
                        break
                
                model_result = {
                    'Model': algorithm,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC AUC': metrics['roc_auc'],
                    'Model Object': model
                }
                
                if existing_model_idx is not None:
                    st.session_state['models_results'][existing_model_idx] = model_result
                else:
                    st.session_state['models_results'].append(model_result)
                
                progress_bar.progress(100)
                status_text.text("Modèle entraîné avec succès!")
            
            # Afficher les résultats si disponibles
            if 'current_model' in st.session_state and 'current_metrics' in st.session_state:
                model = st.session_state['current_model']
                model_name = st.session_state['current_model_name']
                metrics = st.session_state['current_metrics']
                
                st.write(f"## Résultats du modèle {model_name}")
                
                # Box de métriques
                st.markdown(f"""
                ```
                ╔══════════════════════════════╗
                ║       MÉTRIQUES DU MODÈLE    ║
                ╠══════════════════════════════╣
                ║ Accuracy: {metrics['accuracy']:.4f}             ║
                ║ Precision: {metrics['precision']:.4f}            ║
                ║ Recall: {metrics['recall']:.4f}               ║
                ║ F1-score: {metrics['f1']:.4f}             ║
                ║ ROC AUC: {metrics['roc_auc']:.4f}              ║
                ╚══════════════════════════════╝
                ```
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tableau des métriques
                    metrics_df = pd.DataFrame({
                        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                        'Valeur': [
                            metrics['accuracy'],
                            metrics['precision'],
                            metrics['recall'],
                            metrics['f1'],
                            metrics['roc_auc']
                        ]
                    })
                    st.dataframe(metrics_df)
                
                with col2:
                    # Graphique des métriques
                    fig = plt.figure(figsize=(10, 6))
                    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc']]
                    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
                    plt.bar(metrics_names, metrics_values, color='skyblue')
                    plt.ylim(0, 1)
                    plt.title(f'Métriques de performance - {model_name}')
                    plt.xticks(rotation=45)
                    plt.grid(True, axis='y', alpha=0.3)
                    st.pyplot(fig)
                
                # Classification report
                st.write("### Rapport de classification détaillé")
                st.text(metrics['classification_report'])
                
                # Matrice de confusion
                st.write("### Matrice de confusion")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'])
                disp.plot(cmap=metrics['cmap'], ax=ax)
                plt.title(f"Matrice de confusion - {model_name}")
                st.pyplot(fig)
                
                # Feature importance (si disponible)
                if metrics['feature_importance'] is not None:
                    st.write("### Importance des features")
                    
                    st.dataframe(metrics['feature_importance'])
                    
                    # Graphique d'importance
                    fig = plt.figure(figsize=(10, 6))
                    importance_df = metrics['feature_importance']
                    importance_col = 'Importance' if 'Importance' in importance_df.columns else 'Coefficient'
                    plt.barh(importance_df['Feature'].head(10), importance_df[importance_col].head(10))
                    plt.title(f"Top 10 features - {model_name}")
                    plt.xlabel(importance_col)
                    plt.ylabel('Feature')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x', alpha=0.3)
                    st.pyplot(fig)
                
                # Spécifique à l'arbre de décision
                if model_name == "Arbre de Décision" and 'tree_rules' in metrics:
                    st.write("### Structure de l'arbre de décision")
                    st.text(metrics['tree_rules'])
                    
                    # Visualisation de l'arbre
                    fig = plt.figure(figsize=(20, 10))
                    plot_tree(model, 
                              feature_names=list(X_test.columns),
                              class_names=['Non Généreux', 'Généreux'],
                              filled=True,
                              rounded=True,
                              proportion=True)
                    plt.title("Structure de l'arbre de décision")
                    st.pyplot(fig)
                
                # Prédicition interactive
                st.write("### Tester avec vos propres données")
                st.write("Entrez les valeurs pour obtenir une prédiction:")
                
                user_inputs = {}
                for feature in X_train.columns:
                    min_val = float(X_train[feature].min())
                    max_val = float(X_train[feature].max())
                    mean_val = float(X_train[feature].mean())
                    user_inputs[feature] = st.slider(
                        f"{feature}:", 
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val-min_val)/100
                    )
                
                if st.button("Prédire"):
                    user_df = pd.DataFrame([user_inputs])
                    
                    user_pred = model.predict(user_df)[0]
                    user_proba = model.predict_proba(user_df)[0, 1]
                    
                    st.write("### Résultat de la prédiction")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if user_pred == 1:
                            st.success("Client généreux ✓")
                        else:
                            st.error("Client non généreux ✗")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh([0], [user_proba], color='green', height=0.4)
                        ax.barh([0], [1-user_proba], left=[user_proba], color='red', height=0.4)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(-0.5, 0.5)
                        ax.set_yticks([])
                        ax.set_xlabel('Probabilité d\'être généreux')
                        ax.text(user_proba, 0, f"{user_proba:.2%}", 
                               ha='center', va='center',
                               color='white' if user_proba > 0.3 else 'black',
                               fontweight='bold')
                        st.pyplot(fig)
        
        # Section 3: Optimisation d'hyperparamètres
        elif modeling_section == "Optimisation d'Hyperparamètres":
            st.write("## Optimisation d'Hyperparamètres")
            
            # Vérifier les features
            if 'selected_features' not in st.session_state:
                st.info("Aucune feature sélectionnée. Utilisation des features par défaut.")
                selected_features = safe_default_features(datatrainset_clean)
                st.session_state['selected_features'] = selected_features
            else:
                selected_features = st.session_state['selected_features']
                # Vérifier les features existantes
                missing_features = [f for f in selected_features if f not in datatrainset_clean.columns]
                if missing_features:
                    st.warning(f"Features manquantes: {', '.join(missing_features)}")
                    selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                    if not selected_features:
                        selected_features = safe_default_features(datatrainset_clean)
                    st.session_state['selected_features'] = selected_features
            
            st.write(f"Features utilisées: {', '.join(selected_features)}")
            
            # Préparation des données
            X = datatrainset_clean[selected_features].copy()
            y = datatrainset_clean['is_generous'].copy()
            
            # Split
            test_size = st.slider("Proportion du jeu de test:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.write(f"Dimensions: X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Choix de l'algorithme
            algorithm = st.selectbox(
                "Choisissez un algorithme pour l'optimisation:",
                ["Random Forest", "Régression Logistique", "Support Vector Machine (SVM)", 
                 "K-Nearest Neighbors (KNN)", "Arbre de Décision"]
            )
            
            # Méthode de recherche
            search_method = st.radio("Méthode de recherche:", ["GridSearchCV", "RandomizedSearchCV"])
            
            # Paramètres communs
            cv_folds = st.slider("Nombre de folds pour la validation croisée:", min_value=2, max_value=10, value=5)
            scoring = st.selectbox("Métrique d'optimisation:", ["accuracy", "precision", "recall", "f1", "roc_auc"])
            
            # Grilles d'hyperparamètres par algorithme
            if algorithm == "Random Forest":
                st.write("### Hyperparamètres pour Random Forest")
                
                n_estimators_values = st.multiselect(
                    "Nombre d'arbres (n_estimators):",
                    [50, 100, 200, 300],
                    default=[100, 200]
                )
                
                max_depth_options = st.multiselect(
                    "Profondeur maximale (max_depth):",
                    ["None", "5", "10", "20"],
                    default=["None", "10"]
                )
                max_depth_values = [None if x == "None" else int(x) for x in max_depth_options]
                
                min_samples_split_values = st.multiselect(
                    "Minimum d'échantillons pour diviser (min_samples_split):",
                    [2, 5, 10],
                    default=[2, 5]
                )
                
                param_grid = {
                    'n_estimators': n_estimators_values,
                    'max_depth': max_depth_values,
                    'min_samples_split': min_samples_split_values
                }
                
                base_model = RandomForestClassifier(random_state=42)
            
            elif algorithm == "Régression Logistique":
                st.write("### Hyperparamètres pour la Régression Logistique")
                
                C_values = st.multiselect(
                    "Valeurs de C (inverse de la régularisation):",
                    [0.01, 0.1, 1, 10, 100],
                    default=[0.1, 1, 10]
                )
                
                penalty_options = st.multiselect(
                    "Types de pénalité:",
                    ["l1", "l2", "none"],
                    default=["l2", "none"]
                )
                
                solver_options = st.multiselect(
                    "Solveurs:",
                    ["liblinear", "lbfgs", "saga"],
                    default=["liblinear", "lbfgs"]
                )
                
                param_grid = {
                    'C': C_values,
                    'penalty': ['l1', 'l2', None],
                    'solver': solver_options
                }
                
                base_model = LogisticRegression(max_iter=1000)
            
            elif algorithm == "Support Vector Machine (SVM)":
                st.write("### Hyperparamètres pour SVM")
                
                C_values = st.multiselect(
                    "Valeurs de C (pénalité):",
                    [0.1, 1, 10, 100],
                    default=[0.1, 1, 10]
                )
                
                kernel_options = st.multiselect(
                    "Types de kernel:",
                    ["linear", "rbf", "poly", "sigmoid"],
                    default=["linear", "rbf"]
                )
                
                gamma_options = st.multiselect(
                    "Valeurs de gamma:",
                    ["scale", "auto", 0.1, 0.01, 0.001],
                    default=["scale", "auto"]
                )
                
                param_grid = {
                    'C': C_values,
                    'kernel': kernel_options,
                    'gamma': gamma_options
                }
                
                base_model = svm.SVC(probability=True, random_state=42)
            
            elif algorithm == "K-Nearest Neighbors (KNN)":
                st.write("### Hyperparamètres pour KNN")
                
                n_neighbors_values = st.multiselect(
                    "Nombre de voisins:",
                    [3, 5, 7, 9, 11, 15],
                    default=[3, 5, 7]
                )
                
                weights_options = st.multiselect(
                    "Métriques de distance:",
                    ["euclidean", "manhattan", "chebyshev", "minkowski"],
                    default=["euclidean", "manhattan"]
                )
                
                param_grid = {
                    'n_neighbors': n_neighbors_values,
                    'weights': weights_options,
                    'metric': metric_options
                }
                
                base_model = KNeighborsClassifier()
            
            elif algorithm == "Arbre de Décision":
                st.write("### Hyperparamètres pour l'Arbre de Décision")
                
                max_depth_options = st.multiselect(
                    "Profondeur maximale de l'arbre (max_depth):",
                    ["None", "3", "5", "10"],
                    default=["None", "3", "5"]
                )
                max_depth_values = [None if x == "None" else int(x) for x in max_depth_options]
                
                min_samples_split_values = st.multiselect(
                    "Nombre minimum d'échantillons pour diviser un nœud (min_samples_split):",
                    [2, 5, 10],
                    default=[2, 5]
                )
                
                criterion_options = st.multiselect(
                    "Critère de division:",
                    ["gini", "entropy"],
                    default=["gini", "entropy"]
                )
                
                param_grid = {
                    'max_depth': max_depth_values,
                    'min_samples_split': min_samples_split_values,
                    'criterion': criterion_options
                }
                
                base_model = DecisionTreeClassifier(random_state=42)
            
            # Lancement de l'optimisation
            if st.button("Lancer l'optimisation des hyperparamètres"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Préparation de la recherche d'hyperparamètres...")
                progress_bar.progress(10)
                
                # Choix de la méthode
                if search_method == "GridSearchCV":
                    search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv_folds,
                        n_jobs=-1,
                        verbose=1
                    )
                else:  # RandomizedSearchCV
                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring=scoring,
                        cv=cv_folds,
                        n_jobs=-1,
                        random_state=42,
                        verbose=1
                    )
                
                status_text.text("Recherche des meilleurs hyperparamètres en cours...")
                progress_bar.progress(20)
                
                # Entraînement
                search.fit(X_train, y_train)
                
                status_text.text("Traitement des résultats...")
                progress_bar.progress(80)
                
                # Résultats
                best_params = search.best_params_
                best_score = search.best_score_
                best_model = search.best_estimator_
                
                # Évaluation sur le jeu de test
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                # Résultats de l'optimisation
                optimization_results = {
                    'algorithm': algorithm,
                    'best_params': best_params,
                    'best_cv_score': best_score,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_roc_auc': roc_auc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred),
                    'model': best_model
                }
                
                st.session_state['optimization_results'] = optimization_results
                
                # Enregistrer dans les modèles
                optimized_model_name = f"{algorithm} (Optimisé)"
                
                # Vérifier si un modèle existe
                if 'models_results' not in st.session_state:
                    st.session_state['models_results'] = []
                
                existing_model_idx = None
                for i, result in enumerate(st.session_state['models_results']):
                    if result['Model'] == optimized_model_name:
                        existing_model_idx = i
                        break
                
                optimized_model_result = {
                    'Model': optimized_model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'ROC AUC': roc_auc,
                    'Model Object': best_model
                }
                
                if existing_model_idx is not None:
                    st.session_state['models_results'][existing_model_idx] = optimized_model_result
                else:
                    st.session_state['models_results'].append(optimized_model_result)
                
                progress_bar.progress(100)
                status_text.text("Optimisation terminée!")
            
            # Afficher les résultats d'optimisation
            if 'optimization_results' in st.session_state:
                results = st.session_state['optimization_results']
                
                st.write(f"## Résultats de l'optimisation pour {results['algorithm']}")
                
                # Meilleurs hyperparamètres
                st.write("### Meilleurs hyperparamètres")
                st.json(results['best_params'])
                
                # Score de validation croisée
                st.write(f"### Meilleur score de validation croisée ({scoring}): {results['best_cv_score']:.4f}")
                
                # Métriques stylisées
                st.markdown(f"""
                ```
                ╔══════════════════════════════╗
                ║       MÉTRIQUES DU MODÈLE    ║
                ╠══════════════════════════════╣
                ║ Accuracy: {results['test_accuracy']:.4f}             ║
                ║ Precision: {results['test_precision']:.4f}            ║
                ║ Recall: {results['test_recall']:.4f}               ║
                ║ F1-score: {results['test_f1']:.4f}             ║
                ║ ROC AUC: {results['test_roc_auc']:.4f}              ║
                ╚══════════════════════════════╝
                ```
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tableau des métriques
                    metrics_df = pd.DataFrame({
                        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                        'Valeur': [
                            results['test_accuracy'],
                            results['test_precision'],
                            results['test_recall'],
                            results['test_f1'],
                            results['test_roc_auc']
                        ]
                    })
                    st.dataframe(metrics_df)
                
                with col2:
                    # Matrice de confusion
                    fig, ax = plt.subplots(figsize=(8, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=results['confusion_matrix'])
                    disp.plot(cmap='Blues', ax=ax)
                    plt.title(f"Matrice de confusion - {results['algorithm']} (Optimisé)")
                    st.pyplot(fig)
                
                # Classification report
                st.write("### Rapport de classification détaillé")
                st.text(results['classification_report'])
                
                # Importance des features (si disponible)
                if hasattr(results['model'], 'feature_importances_'):
                    st.write("### Importance des features")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': results['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.dataframe(feature_importance)
                    
                    # Graphique d'importance
                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
                    plt.title(f"Top 10 des features - {results['algorithm']} (Optimisé)")
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x', alpha=0.3)
                    st.pyplot(fig)
                
                # Arbre de décision optimisé
                if results['algorithm'] == "Arbre de Décision":
                    st.write("### Structure de l'arbre optimisé")
                    
                    tree_rules = export_text(results['model'], feature_names=list(X.columns))
                    st.text(tree_rules)
                    
                    # Visualisation de l'arbre
                    fig = plt.figure(figsize=(20, 10))
                    plot_tree(results['model'], 
                              feature_names=list(X.columns),
                              class_names=['Non Généreux', 'Généreux'],
                              filled=True,
                              rounded=True,
                              proportion=True)
                    plt.title("Structure de l'arbre de décision optimisé")
                    st.pyplot(fig)
                
                # Prédiction avec le modèle optimisé
                st.write("### Tester avec vos propres données")
                st.write("Entrez les valeurs pour obtenir une prédiction:")
                
                user_inputs = {}
                for feature in X.columns:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    user_inputs[feature] = st.slider(
                        f"{feature}:", 
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val-min_val)/100
                    )
                
                if st.button("Prédire avec le modèle optimisé"):
                    user_df = pd.DataFrame([user_inputs])
                    
                    user_pred = results['model'].predict(user_df)[0]
                    user_proba = results['model'].predict_proba(user_df)[0, 1]
                    
                    st.write("### Résultat de la prédiction")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if user_pred == 1:
                            st.success("Client généreux ✓")
                        else:
                            st.error("Client non généreux ✗")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh([0], [user_proba], color='green', height=0.4)
                        ax.barh([0], [1-user_proba], left=[user_proba], color='red', height=0.4)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(-0.5, 0.5)
                        ax.set_yticks([])
                        ax.set_xlabel('Probabilité d\'être généreux')
                        ax.text(user_proba, 0, f"{user_proba:.2%}", 
                               ha='center', va='center',
                               color='white' if user_proba > 0.3 else 'black',
                               fontweight='bold')
                        st.pyplot(fig)
        
        # Section 4: Comparaison des modèles
        elif modeling_section == "Comparaison des Modèles":
            st.write("## Comparaison des Modèles")
            
            if 'models_results' not in st.session_state or not st.session_state['models_results']:
                st.warning("Aucun modèle n'a été entraîné. Veuillez d'abord entraîner des modèles dans les sections précédentes.")
            else:
                # Tableau comparatif
                models_results = st.session_state['models_results']
                models_df = pd.DataFrame(models_results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']]
                
                st.write("### Tableau comparatif des modèles")
                st.dataframe(models_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']))
                
                # Graphiques de comparaison
                st.write("### Comparaison graphique des modèles")
                
                metrics_to_compare = st.multiselect(
                    "Sélectionnez les métriques à comparer:",
                    ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                    default=['Accuracy', 'F1-Score', 'ROC AUC']
                )
                
                if metrics_to_compare:
                    # Préparation du graphique
                    models_names = models_df['Model'].tolist()
                    metrics_data = []
                    
                    for metric in metrics_to_compare:
                        metrics_data.append(models_df[metric].tolist())
                    
                    # Graphique à barres groupées
                    fig, ax = plt.subplots(figsize=(12, 8))
                    x = np.arange(len(models_names))
                    width = 0.8 / len(metrics_to_compare)
                    
                    for i, metric in enumerate(metrics_to_compare):
                        ax.bar(x + i * width - width * len(metrics_to_compare) / 2 + width / 2, 
                              models_df[metric], 
                              width, 
                              label=metric)
                    
                    ax.set_ylim(0, 1)
                    ax.set_xlabel('Modèles')
                    ax.set_ylabel('Score')
                    ax.set_title('Comparaison des modèles')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_names, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Graphique radar
                    if len(metrics_to_compare) >= 3:
                        st.write("### Graphique radar des performances")
                        
                        models_to_compare = st.multiselect(
                            "Sélectionnez les modèles à comparer dans le graphique radar:",
                            models_names,
                            default=models_names[:min(3, len(models_names))]
                        )
                        
                        if models_to_compare:
                            filtered_models_df = models_df[models_df['Model'].isin(models_to_compare)]
                            
                            fig = plt.figure(figsize=(10, 10))
                            ax = fig.add_subplot(111, polar=True)
                            
                            # Nombre de variables
                            N = len(metrics_to_compare)
                            
                            # Angle pour chaque variable
                            angles = [n / float(N) * 2 * np.pi for n in range(N)]
                            angles += angles[:1]  # Fermer le graphique
                            
                            # Labels
                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(metrics_to_compare)
                            
                            # Traçage pour chaque modèle
                            for idx, model in filtered_models_df.iterrows():
                                values = [model[metric] for metric in metrics_to_compare]
                                values += values[:1]  # Fermer le graphique
                                
                                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model['Model'])
                                ax.fill(angles, values, alpha=0.25)
                            
                            plt.legend(loc='upper right')
                            st.pyplot(fig)
                
                # Sélection du meilleur modèle
                st.write("### Sélection du meilleur modèle")
                
                best_model_criterion = st.selectbox(
                    "Critère pour sélectionner le meilleur modèle:",
                    ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                    index=3  # F1-Score par défaut
                )
                
                # Trouver le meilleur modèle
                best_model_idx = models_df[best_model_criterion].idxmax()
                best_model_row = models_df.iloc[best_model_idx]
                
                st.success(f"Le meilleur modèle selon {best_model_criterion} est **{best_model_row['Model']}** avec un score de {best_model_row[best_model_criterion]:.4f}.")
                
                # Exporter les résultats
                st.write("### Exportation des résultats")
                
                csv = models_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger la comparaison (CSV)",
                    data=csv,
                    file_name="comparaison_modeles.csv",
                    mime="text/csv"
                )

# Activer l'application
# Pour exécuter: streamlit run app.py