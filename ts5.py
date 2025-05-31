import streamlit as st
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Données de Taxis Jaunes",
    page_icon="🚕",
    layout="wide"
)


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
# Imports nécessaires pour les modèles
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

# Imports pour la visualisation et manipulation de données
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Fonction corrigée pour créer la variable cible is_generous (pourboire >= 20% du fare_amount)
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

# Fonction pour afficher le header commun
def display_header():
    st.markdown("<h1 style='text-align:center;'>Analyse des Données de Taxis Jaunes </h1>", unsafe_allow_html=True)

# Sidebar pour la navigation et le chargement des données
st.sidebar.title("Navigation")

# Widget d'upload de fichier
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

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
         "🤖 Modélisation"]

page = st.sidebar.selectbox("Sélectionnez une page", pages)

# Contexte et objectifs dans le sidebar
with st.sidebar.expander("Contexte et Objectifs"):
    st.markdown("""
    <h5>Contexte</h5>
    <p>Les pourboires représentent une part significative des revenus des chauffeurs de taxi à New York.</p>
    
    <h5>Objectif Principal</h5>
    <p>Construire un modèle prédictif pour identifier les clients susceptibles de donner un pourboire ≥ 20% du montant total.</p>
    
    <h5>Hypothèses à Tester</h5>
    <ul>
        <li>H1 : Les clients payant par carte sont plus généreux que ceux payant en espèces.</li>
        <li>H2 : Les courses longues distances (> 10 miles) ont des pourboires proportionnellement plus élevés</li>
        <li>H3 : Les trajets vers les aéroports (RatecodeID=2) ont un taux de générosité différent.</li>
    </ul>
    """, unsafe_allow_html=True)

# Afficher la page sélectionnée
if page == "📊 Analyse exploratoire des données":
    display_header()
    # Ajouter le téléchargement du modèle CSV dans la barre latérale
    st.sidebar.write("## Télécharger un modèle CSV")
    st.sidebar.write("Téléchargez un modèle CSV vierge à remplir avec vos données.")
    
    # Fonction pour créer un modèle CSV
    def create_template_csv():
        template_columns = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
            'passenger_count', 'trip_distance', 'RatecodeID', 
            'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 
            'payment_type', 'fare_amount', 'tip_amount', 
            'tolls_amount', 'total_amount'
        ]
        
        # Créer un DataFrame vide avec les colonnes requises
        template_df = pd.DataFrame(columns=template_columns)
        
        # Ajouter quelques lignes d'exemple
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
    
    # Créer le template
    template_df = create_template_csv()
    
    # Convertir le DataFrame en CSV
    csv = template_df.to_csv(index=False)
    
    # Bouton de téléchargement pour le modèle
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
        4. Importez votre fichier complété en utilisant l'option 'Importer un fichier CSV' dans la barre latérale principale
        
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
        - **store_and_fwd_flag**: Flag indiquant si le trajet a été stocké dans la mémoire avant envoi (Y=oui, N=non)
        - **PULocationID**: ID du lieu de prise en charge
        - **DOLocationID**: ID du lieu de dépose
        - **payment_type**: Type de paiement
        - **fare_amount**: Montant du tarif
        - **tip_amount**: Montant du pourboire
        - **tolls_amount**: Montant des péages
        - **total_amount**: Montant total
        """)
    
    # Sous-sections de la page EDA (Le reste de votre code reste identique)
    eda_sections = ["Vue d'ensemble des données", 
                    "Analyse Univariée", 
                    "Analyse Bivariée",
                    "Création et Analyse de la Variable Cible"]
    
    eda_section = st.radio("Sélectionnez une section:", eda_sections, horizontal=True)
    
    # Le reste du code pour cette page reste inchangé...
    
    # Vue d'ensemble des données
    if eda_section == "Vue d'ensemble des données":
        st.write("## Vue d'ensemble des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Aperçu des données")
            st.dataframe(df.head(10))
            
            st.write("### Informations sur les données")
            # Recréer l'affichage de df.info() car il ne fonctionne pas directement dans streamlit
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.write("### Statistiques descriptives")
            st.dataframe(df.describe())
            
            st.write("### Types de données")
            # Afficher les types de données sous forme de tableau
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
        
        # Calculer et afficher le pourcentage de valeurs manquantes
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
        
        # Sélectionner une colonne à analyser
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        col_type = st.radio("Type de variables:", ["Numériques", "Catégorielles", "Temporelles"], horizontal=True)
        
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
                
                # Afficher les valeurs aberrantes potentielles
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
                st.write(f"**Valeurs aberrantes potentielles**: {len(outliers)} lignes")
                if len(outliers) > 0 and len(outliers) < 1000:
                    st.dataframe(outliers[[column]])
        
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
                    
                    # Calculer les pourcentages
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
        
        elif col_type == "Temporelles":
            if not datetime_cols:
                st.warning("Aucune variable temporelle détectée dans le jeu de données.")
            else:
                column = st.selectbox("Sélectionnez une variable temporelle:", datetime_cols)
                
                # Vérifier si la colonne est déjà au format datetime
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        # Tenter de convertir en datetime
                        time_series = pd.to_datetime(df[column])
                    except:
                        st.error(f"Impossible de convertir la colonne {column} en format date/heure.")
                        time_series = None
                else:
                    time_series = df[column]
                
                if time_series is not None:
                    # Extraire composantes de date
                    st.write("### Composantes temporelles")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution par heure de la journée
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
                        
                        # Distribution cumulative dans le temps
                        fig = plt.figure(figsize=(10, 6))
                        time_series.sort_values().value_counts().sort_index().cumsum().plot()
                        plt.title('Distribution cumulative dans le temps')
                        plt.xlabel('Date')
                        plt.ylabel('Nombre cumulé de courses')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
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
            
            # Statistiques sur le pourcentage de pourboire
            st.write("### Statistiques sur le pourcentage de pourboire")
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
            
            # Camembert de la distribution de is_generous
            target_counts = df['is_generous'].value_counts()
            fig = plt.figure(figsize=(10, 6))
            plt.pie(target_counts, labels=['Non généreux', 'Généreux'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
            plt.title('Distribution de la variable cible is_generous')
            st.pyplot(fig)
            
            # Afficher les déséquilibres de classes
            st.write("### Équilibre des classes")
            class_ratio = target_counts[1] / target_counts[0] if 0 in target_counts and 1 in target_counts else 0
            
            st.metric("Ratio Généreux / Non généreux", f"{class_ratio:.4f}")
            
            if class_ratio < 0.2:
                st.warning("⚠️ Le jeu de données est fortement déséquilibré. Envisagez des techniques de rééquilibrage pour la modélisation.")
            elif class_ratio < 0.5:
                st.info("ℹ️ Le jeu de données présente un déséquilibre modéré.")
            else:
                st.success("✅ Le jeu de données est relativement équilibré.")
    
    # Analyse Bivariée
    elif eda_section == "Analyse Bivariée":
        st.write("## Analyse Bivariée avec la Variable Cible")
        
        # Sélection de la variable à croiser avec la cible
        all_cols = [col for col in df.columns if col not in ['is_generous', 'tip_percentage']]
        selected_col = st.selectbox("Sélectionnez une variable à croiser avec 'is_generous':", all_cols)
        
        col1, col2 = st.columns(2)
        
        # Déterminer si la variable est catégorielle ou numérique
        if df[selected_col].dtype == 'object' or df[selected_col].nunique() < 10:
            # Variable catégorielle ou discrète avec peu de valeurs
            with col1:
                st.write(f"### Taux de générosité par {selected_col}")
                
                # Calculer le taux de générosité par catégorie
                generosity_by_category = df.groupby(selected_col)['is_generous'].mean().reset_index()
                generosity_by_category.columns = [selected_col, 'Taux de générosité']
                generosity_by_category['Taux de générosité'] *= 100  # Convertir en pourcentage
                
                # Trier par taux de générosité
                generosity_by_category = generosity_by_category.sort_values('Taux de générosité', ascending=False)
                
                # Afficher le tableau
                st.dataframe(generosity_by_category)
                
                # Afficher un graphique à barres
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(x='Taux de générosité', y=selected_col, data=generosity_by_category)
                plt.title(f'Taux de générosité (%) par {selected_col}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.write(f"### Distribution de {selected_col} par classe")
                
                # Compter les occurrences de chaque catégorie par classe
                category_by_class = pd.crosstab(df[selected_col], df['is_generous'])
                category_by_class.columns = ['Non généreux', 'Généreux']
                
                # Calculer les pourcentages
                category_by_class_pct = category_by_class.div(category_by_class.sum(axis=1), axis=0) * 100
                
                # Afficher le tableau
                st.dataframe(category_by_class)
                
                # Afficher un graphique à barres empilées
                fig = plt.figure(figsize=(10, 6))
                category_by_class_pct.plot(kind='barh', stacked=True)
                plt.title(f'Distribution de {selected_col} par classe (en %)')
                plt.legend(title='Classe')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Test de chi2 pour les variables catégorielles
                contingency_table = pd.crosstab(df[selected_col], df['is_generous'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                
                st.write("### Test statistique (Chi²)")
                st.write(f"Valeur Chi² : {chi2:.4f}")
                st.write(f"p-value : {p:.4f}")
                
                if p < 0.05:
                    st.success(f"✅ Il existe une relation significative entre {selected_col} et la générosité (p < 0.05)")
                else:
                    st.info(f"ℹ️ Pas de relation significative détectée entre {selected_col} et la générosité (p > 0.05)")
        
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
                
                # Test t de Student ou Mann-Whitney selon la normalité
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
                    st.info(f"ℹ️ Pas de différence significative de {selected_col} entre les classes (p > 0.05)")
        
        # Test des hypothèses spécifiques - VERSION CORRIGÉE
        st.write("## Test des hypothèses")

        hypothesis_tabs = st.tabs(["H1: Mode de paiement", "H2: Distance", "H3: Aéroports"])

        with hypothesis_tabs[0]:
            st.write("### H1: Les clients payant par carte sont plus généreux")
            
            # Vérifier que payment_type existe
            if 'payment_type' in df.columns:
                # Calculer le taux de générosité par type de paiement
                payment_generosity = df.groupby('payment_type')['is_generous'].mean().reset_index()
                payment_generosity.columns = ['Type de paiement', 'Taux de générosité']
                payment_generosity['Taux de générosité'] *= 100  # Convertir en pourcentage
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(payment_generosity)
                    
                    # Expliquer les codes payment_type
                    st.info("""
                    Types de paiement:
                    1 = Carte de crédit
                    2 = Espèces
                    3 = Pas de frais
                    4 = Contestation
                    """)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Type de paiement', y='Taux de générosité', data=payment_generosity)
                    plt.title('Taux de générosité par type de paiement')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Test statistique (comparaison carte vs espèces)
                card = df[df['payment_type'] == 1]['is_generous']
                cash = df[df['payment_type'] == 2]['is_generous']
                
                # Afficher des statistiques de débogage
                st.write("### Statistiques de débogage:")
                st.write(f"Nombre de paiements par carte: {len(card)}")
                st.write(f"Nombre de clients généreux parmi les paiements par carte: {card.sum()}")
                st.write(f"Pourcentage: {card.mean()*100:.2f}%")
                st.write(f"Nombre de paiements en espèces: {len(cash)}")
                st.write(f"Nombre de clients généreux parmi les paiements en espèces: {cash.sum()}")
                st.write(f"Pourcentage: {cash.mean()*100:.2f}%")
                
                # Approche améliorée avec chi2_contingency
                if len(card) > 0 and len(cash) > 0:
                    try:
                        # Créer une table de contingence
                        contingency = pd.crosstab(
                            df[df['payment_type'].isin([1, 2])]['payment_type'],
                            df[df['payment_type'].isin([1, 2])]['is_generous']
                        )
                        
                        # Afficher la table pour vérification
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        # Effectuer le test si la table a la bonne forme
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Carte vs Espèces (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                # Vérifier quelle proportion est plus élevée
                                if df[df['payment_type'] == 1]['is_generous'].mean() > df[df['payment_type'] == 2]['is_generous'].mean():
                                    st.success("✅ H1 CONFIRMÉE: Les clients payant par carte sont significativement plus généreux (p < 0.05)")
                                else:
                                    st.error("❌ H1 REJETÉE: Les clients payant par carte sont significativement moins généreux (p < 0.05)")
                            else:
                                st.info("ℹ️ H1 INDÉTERMINÉE: Pas de différence significative entre carte et espèces (p > 0.05)")
                        else:
                            st.warning("La table de contingence n'a pas la bonne forme pour effectuer le test chi2. Vérifiez vos données.")
                    except Exception as e:
                        st.error(f"Erreur lors du test statistique Chi²: {str(e)}")
                        
                        # Essayer l'approche alternative avec proportions_ztest
                        st.write("### Tentative alternative avec test Z des proportions")
                        try:
                            if card.sum() > 0 and cash.sum() > 0 and len(card) - card.sum() > 0 and len(cash) - cash.sum() > 0:
                                z_stat, p_value = stats.proportions_ztest(
                                    [card.sum(), cash.sum()], 
                                    [len(card), len(cash)]
                                )
                                
                                st.write(f"Z-statistique : {z_stat:.4f}")
                                st.write(f"p-value : {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    if card.mean() > cash.mean():
                                        st.success("✅ H1 CONFIRMÉE: Les clients payant par carte sont significativement plus généreux (p < 0.05)")
                                    else:
                                        st.error("❌ H1 REJETÉE: Les clients payant par carte sont significativement moins généreux (p < 0.05)")
                                else:
                                    st.info("ℹ️ H1 INDÉTERMINÉE: Pas de différence significative entre carte et espèces (p > 0.05)")
                            else:
                                st.warning("Au moins l'un des groupes n'a pas suffisamment d'observations de chaque type pour le test Z")
                        except Exception as e:
                            st.error(f"Erreur lors du test Z de proportions: {str(e)}")
            else:
                st.error("La colonne 'payment_type' n'existe pas dans le jeu de données.")

        with hypothesis_tabs[1]:
            st.write("### H2: Les courses longues distances ont des pourboires proportionnellement plus élevés")
            
            # Vérifier que trip_distance existe
            if 'trip_distance' in df.columns:
                # Définir le seuil de longue distance (10 miles)
                long_distance_threshold = 10
                
                # Créer une variable catégorielle pour la distance
                df['distance_cat'] = df['trip_distance'].apply(lambda x: 'Long (>10 miles)' if x > long_distance_threshold else 'Court (≤10 miles)')
                
                # Calculer le taux de générosité par catégorie de distance
                distance_generosity = df.groupby('distance_cat')['is_generous'].agg(['mean', 'count']).reset_index()
                distance_generosity.columns = ['Catégorie de distance', 'Taux de générosité', 'Nombre de courses']
                distance_generosity['Taux de générosité'] *= 100  # Convertir en pourcentage
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(distance_generosity)
                    
                    # Statistiques sur les pourboires par catégorie
                    tip_stats_by_distance = df.groupby('distance_cat')['tip_percentage'].agg(['mean', 'median']).reset_index()
                    tip_stats_by_distance.columns = ['Catégorie de distance', 'Pourboire moyen (%)', 'Pourboire médian (%)']
                    st.dataframe(tip_stats_by_distance)
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Catégorie de distance', y='Taux de générosité', data=distance_generosity)
                    plt.title('Taux de générosité par catégorie de distance')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Boxplot du pourcentage de pourboire par catégorie
                    fig = plt.figure(figsize=(10, 6))
                    sns.boxplot(x='distance_cat', y='tip_percentage', data=df)
                    plt.title('Distribution du pourcentage de pourboire par catégorie de distance')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Test statistique
                long_distance = df[df['trip_distance'] > long_distance_threshold]['is_generous']
                short_distance = df[df['trip_distance'] <= long_distance_threshold]['is_generous']
                
                # Afficher des statistiques de débogage
                st.write("### Statistiques de débogage:")
                st.write(f"Nombre de courses longue distance: {len(long_distance)}")
                st.write(f"Nombre de clients généreux parmi les courses longue distance: {long_distance.sum()}")
                st.write(f"Pourcentage: {long_distance.mean()*100:.2f}%")
                st.write(f"Nombre de courses courte distance: {len(short_distance)}")
                st.write(f"Nombre de clients généreux parmi les courses courte distance: {short_distance.sum()}")
                st.write(f"Pourcentage: {short_distance.mean()*100:.2f}%")
                
                # Approche avec chi2_contingency
                if len(long_distance) > 0 and len(short_distance) > 0:
                    try:
                        # Créer une table de contingence
                        contingency = pd.crosstab(
                            df['distance_cat'],
                            df['is_generous']
                        )
                        
                        # Afficher la table pour vérification
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        # Effectuer le test si la table a la bonne forme
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Longue vs Courte distance (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                # Vérifier quelle proportion est plus élevée
                                if long_distance.mean() > short_distance.mean():
                                    st.success("✅ H2 CONFIRMÉE: Les courses longues distances ont significativement plus de pourboires généreux (p < 0.05)")
                                else:
                                    st.error("❌ H2 REJETÉE: Les courses longues distances ont significativement moins de pourboires généreux (p < 0.05)")
                            else:
                                st.info("ℹ️ H2 INDÉTERMINÉE: Pas de différence significative entre longue et courte distance (p > 0.05)")
                        else:
                            st.warning("La table de contingence n'a pas la bonne forme pour effectuer le test chi2. Vérifiez vos données.")
                    except Exception as e:
                        st.error(f"Erreur lors du test statistique Chi²: {str(e)}")
                        
                        # Essayer l'approche alternative avec proportions_ztest
                        st.write("### Tentative alternative avec test Z des proportions")
                        try:
                            if long_distance.sum() > 0 and short_distance.sum() > 0 and len(long_distance) - long_distance.sum() > 0 and len(short_distance) - short_distance.sum() > 0:
                                z_stat, p_value = stats.proportions_ztest(
                                    [long_distance.sum(), short_distance.sum()], 
                                    [len(long_distance), len(short_distance)]
                                )
                                
                                st.write(f"Z-statistique : {z_stat:.4f}")
                                st.write(f"p-value : {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    if long_distance.mean() > short_distance.mean():
                                        st.success("✅ H2 CONFIRMÉE: Les courses longues distances ont significativement plus de pourboires généreux (p < 0.05)")
                                    else:
                                        st.error("❌ H2 REJETÉE: Les courses longues distances ont significativement moins de pourboires généreux (p < 0.05)")
                                else:
                                    st.info("ℹ️ H2 INDÉTERMINÉE: Pas de différence significative entre longue et courte distance (p > 0.05)")
                            else:
                                st.warning("Au moins l'un des groupes n'a pas suffisamment d'observations de chaque type pour le test Z")
                        except Exception as e:
                            st.error(f"Erreur lors du test Z de proportions: {str(e)}")
            else:
                st.error("La colonne 'trip_distance' n'existe pas dans le jeu de données.")

        with hypothesis_tabs[2]:
            st.write("### H3: Les trajets vers les aéroports ont un taux de générosité différent")
            
            # Vérifier que RatecodeID existe
            if 'RatecodeID' in df.columns:
                # Trajets vers aéroports: RatecodeID=2
                df['airport_trip'] = df['RatecodeID'].apply(lambda x: 'Aéroport' if x == 2 else 'Non-aéroport')
                
                # Calculer le taux de générosité pour les trajets aéroport vs non-aéroport
                airport_generosity = df.groupby('airport_trip')['is_generous'].agg(['mean', 'count']).reset_index()
                airport_generosity.columns = ['Type de trajet', 'Taux de générosité', 'Nombre de courses']
                airport_generosity['Taux de générosité'] *= 100  # Convertir en pourcentage
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(airport_generosity)
                    
                    # Statistiques sur les pourboires par type de trajet
                    tip_stats_by_airport = df.groupby('airport_trip')['tip_percentage'].agg(['mean', 'median']).reset_index()
                    tip_stats_by_airport.columns = ['Type de trajet', 'Pourboire moyen (%)', 'Pourboire médian (%)']
                    st.dataframe(tip_stats_by_airport)
                    
                    # Expliquer les codes RatecodeID
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
                    
                    # Boxplot du pourcentage de pourboire par type de trajet
                    fig = plt.figure(figsize=(10, 6))
                    sns.boxplot(x='airport_trip', y='tip_percentage', data=df)
                    plt.title('Distribution du pourcentage de pourboire par type de trajet')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Test statistique
                airport = df[df['RatecodeID'] == 2]['is_generous']
                non_airport = df[df['RatecodeID'] != 2]['is_generous']
                
                # Afficher des statistiques de débogage
                st.write("### Statistiques de débogage:")
                st.write(f"Nombre de trajets vers l'aéroport: {len(airport)}")
                st.write(f"Nombre de clients généreux parmi les trajets vers l'aéroport: {airport.sum()}")
                st.write(f"Pourcentage: {airport.mean()*100:.2f}%" if len(airport) > 0 else "Pourcentage: N/A")
                st.write(f"Nombre de trajets hors aéroport: {len(non_airport)}")
                st.write(f"Nombre de clients généreux parmi les trajets hors aéroport: {non_airport.sum()}")
                st.write(f"Pourcentage: {non_airport.mean()*100:.2f}%")
                
                # Approche avec chi2_contingency
                if len(airport) > 0 and len(non_airport) > 0:
                    try:
                        # Créer une table de contingence
                        contingency = pd.crosstab(
                            df['airport_trip'],
                            df['is_generous']
                        )
                        
                        # Afficher la table pour vérification
                        st.write("### Table de contingence:")
                        st.write(contingency)
                        
                        # Effectuer le test si la table a la bonne forme
                        if contingency.shape == (2, 2):
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            
                            st.write("### Comparaison Aéroport vs Non-aéroport (Chi²)")
                            st.write(f"Chi² : {chi2:.4f}")
                            st.write(f"p-value : {p_value:.4f}")
                            
                            if p_value < 0.05:
                                # Vérifier quelle proportion est plus élevée
                                if airport.mean() > non_airport.mean():
                                    st.success("✅ H3 CONFIRMÉE: Les trajets vers les aéroports ont significativement plus de pourboires généreux (p < 0.05)")
                                else:
                                    st.error("❌ H3 REJETÉE: Les trajets vers les aéroports ont significativement moins de pourboires généreux (p < 0.05)")
                            else:
                                st.info("ℹ️ H3 INDÉTERMINÉE: Pas de différence significative entre les trajets aéroport et non-aéroport (p > 0.05)")
                        else:
                            st.warning("La table de contingence n'a pas la bonne forme pour effectuer le test chi2. Vérifiez vos données.")
                    except Exception as e:
                        st.error(f"Erreur lors du test statistique Chi²: {str(e)}")
                        
                        # Essayer l'approche alternative avec proportions_ztest
                        st.write("### Tentative alternative avec test Z des proportions")
                        try:
                            if len(airport) > 0 and airport.sum() > 0 and len(airport) - airport.sum() > 0 and non_airport.sum() > 0 and len(non_airport) - non_airport.sum() > 0:
                                z_stat, p_value = stats.proportions_ztest(
                                    [airport.sum(), non_airport.sum()], 
                                    [len(airport), len(non_airport)]
                                )
                                
                                st.write(f"Z-statistique : {z_stat:.4f}")
                                st.write(f"p-value : {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    if airport.mean() > non_airport.mean():
                                        st.success("✅ H3 CONFIRMÉE: Les trajets vers les aéroports ont significativement plus de pourboires généreux (p < 0.05)")
                                    else:
                                        st.error("❌ H3 REJETÉE: Les trajets vers les aéroports ont significativement moins de pourboires généreux (p < 0.05)")
                                else:
                                    st.info("ℹ️ H3 INDÉTERMINÉE: Pas de différence significative entre les trajets aéroport et non-aéroport (p > 0.05)")
                            else:
                                st.warning("Au moins l'un des groupes n'a pas suffisamment d'observations de chaque type pour le test Z")
                        except Exception as e:
                            st.error(f"Erreur lors du test Z de proportions: {str(e)}")
            else:
                st.error("La colonne 'RatecodeID' n'existe pas dans le jeu de données.")

elif page == "🛠️ Traitement des données":
    display_header()
    
    # Sous-sections de la page de traitement des données
    processing_sections = ["Analyse des outliers", 
                           "Analyse des Corrélations", 
                           "Pré-Traitement"]
    
    processing_section = st.radio("Sélectionnez une section:", processing_sections, horizontal=True)
    
    # Analyse des outliers
    if processing_section == "Analyse des outliers":
        st.write("## Analyse des outliers")
        
        # Sélectionner les variables numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclure certaines colonnes comme les identifiants et la cible
        cols_to_exclude = ['Unnamed: 0', 'is_generous', 'tip_percentage']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        selected_col = st.selectbox("Sélectionnez une variable pour l'analyse des outliers:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot pour visualiser les outliers
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[selected_col])
            plt.title(f'Boxplot de {selected_col} - Détection des outliers')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Statistiques de la variable
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
                    Q1,
                    Q3,
                    IQR,
                    lower_bound,
                    upper_bound
                ]
            })
            st.dataframe(stats_df)
            
            # Détection des outliers
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_percent = (outliers_count / len(df)) * 100
            
            st.write(f"### Outliers détectés: {outliers_count} ({outliers_percent:.2f}%)")
            
            # Histogramme avec KDE
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
        
        # Sélectionner les variables numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclure certaines colonnes comme les identifiants
        cols_to_exclude = ['Unnamed: 0']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        # Calculer la matrice de corrélation avec Pearson uniquement
        correlation_matrix = df[numeric_cols].corr(method='pearson')
        
        # Afficher la matrice de corrélation
        fig = plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matrice de corrélation (Pearson)')
        st.pyplot(fig)
        
        # Afficher les valeurs de toutes les corrélations
        st.write("### Valeurs de toutes les corrélations")
        
        # Transformer la matrice de corrélation en un format plus lisible
        corr_list = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_list.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Corrélation': correlation_matrix.iloc[i, j]
                })
        
        # Créer un DataFrame et trier par valeur absolue de corrélation décroissante
        corr_df = pd.DataFrame(corr_list)
        corr_df['Corrélation absolue'] = corr_df['Corrélation'].abs()
        corr_df = corr_df.sort_values('Corrélation absolue', ascending=False)
        
        # Afficher le tableau
        st.dataframe(corr_df)
        
        # Analyse des corrélations avec la variable cible
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
        
        # PCA pour visualiser les relations entre les variables
        st.write("### Analyse en Composantes Principales (ACP)")
        
        # Sélection des variables pour l'ACP (exclure la cible)
        pca_cols = [col for col in numeric_cols if col not in ['is_generous', 'tip_percentage']]
        
        if len(pca_cols) >= 2:
            # Standardiser les données
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[pca_cols])
            
            # Appliquer PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(df_scaled)
            
            # Créer un DataFrame avec les composantes principales
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            
            # Ajouter la cible si disponible
            if 'is_generous' in df.columns:
                pca_df['is_generous'] = df['is_generous'].values
            
            # Afficher le graphique PCA
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
                # Afficher la variance expliquée
                explained_variance = pca.explained_variance_ratio_
                st.write("### Variance expliquée par les composantes")
                
                variance_df = pd.DataFrame({
                    'Composante': ['PC1', 'PC2'],
                    'Variance expliquée (%)': explained_variance * 100
                })
                st.dataframe(variance_df)
                
                # Graphique de variance cumulée
                fig = plt.figure(figsize=(10, 6))
                plt.bar(['PC1', 'PC2'], explained_variance)
                plt.axhline(y=0.7, color='r', linestyle='-', label='Seuil 70%')
                plt.title('Variance expliquée par composante')
                plt.ylabel('Proportion de variance expliquée')
                plt.grid(True, alpha=0.3)
                plt.legend()
                st.pyplot(fig)
            
            # Afficher les contributions des variables aux composantes principales
            st.write("### Contribution des variables aux composantes principales")
            
            loadings = pca.components_.T
            loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=pca_cols)
            
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Contribution des variables aux composantes principales')
            st.pyplot(fig)
            
            # Section Biplot supprimée comme demandé
        else:
            st.warning("Pas assez de variables numériques pour réaliser une ACP.")
    
    # Pré-Traitement
    elif processing_section == "Pré-Traitement":
        st.write("## Pré-Traitement des données")
        
        # Afficher les étapes de pré-traitement disponibles
        preprocessing_options = st.multiselect(
            "Sélectionnez les opérations de pré-traitement à appliquer:",
            ["Nettoyage des colonnes inutiles",
             "Encodage des variables catégorielles", 
             "Feature Engineering", 
             "Standardisation"]
        )
        
        # Initialiser dataframe traité
        df_processed = df.copy()
        transformations_applied = []
        
        # 1. Nettoyage des colonnes inutiles
        if "Nettoyage des colonnes inutiles" in preprocessing_options:
            st.write("### Nettoyage des colonnes inutiles")
            
            # Fonction de nettoyage
            def nettoyage(df):
                # Colonnes à supprimer
                cols_to_drop = ['extra', 'mta_tax','Unnamed: 0']
                
                # Vérifier si ces colonnes existent avant de les supprimer
                cols_to_drop = [col for col in cols_to_drop if col in df.columns]
                if cols_to_drop:
                    df = df.drop(cols_to_drop, axis=1)
                    transformations_applied.append(f"Suppression des colonnes: {', '.join(cols_to_drop)}")
                
                # Filtrer les lignes selon les conditions
                n_before = len(df)
                
                # Vérifier que les colonnes existent avant de filtrer
                conditions = []
                
                if "passenger_count" in df.columns:
                    conditions.append(df["passenger_count"] != 0)
                
                if "trip_distance" in df.columns:
                    conditions.append(df["trip_distance"] > 0)
                
                if "fare_amount" in df.columns:
                    conditions.append(df["fare_amount"] > 0)
                
                if "total_amount" in df.columns:
                    conditions.append(df["total_amount"] > 0)
                
                # Gérer pickup_minute s'il existe ou le créer si tpep_pickup_datetime existe
                if "tpep_pickup_datetime" in df.columns and pd.api.types.is_datetime64_dtype(df["tpep_pickup_datetime"]):
                    if "pickup_minute" not in df.columns:
                        df["pickup_minute"] = df["tpep_pickup_datetime"].dt.minute
                        transformations_applied.append("Création de la colonne pickup_minute")
                
                if "pickup_minute" in df.columns:
                    conditions.append(df["pickup_minute"] > 0)
                
                # Appliquer les conditions de filtrage si elles existent
                if conditions:
                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition = combined_condition & condition
                    
                    df = df[combined_condition].copy()
                    n_after = len(df)
                    
                    if n_before > n_after:
                        transformations_applied.append(f"Filtrage des données: {n_before - n_after} lignes supprimées")
                
                return df
            
            # Appliquer le nettoyage
            df_processed = nettoyage(df_processed)
            
            # Afficher le résultat du nettoyage
            st.write(f"Taille du DataFrame avant nettoyage: {len(df)}")
            st.write(f"Taille du DataFrame après nettoyage: {len(df_processed)}")
            st.write(f"Lignes supprimées: {len(df) - len(df_processed)}")
            
            # Afficher un aperçu des données nettoyées
            st.write("#### Aperçu des données nettoyées")
            st.dataframe(df_processed.head())
        
        # 2. Encodage des variables catégorielles
        if "Encodage des variables catégorielles" in preprocessing_options:
            st.write("### Encodage des variables catégorielles")
            
            # Fonction d'encodage
            def encodage(df):
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
                
                # Appliquer l'encodage uniquement sur les colonnes object/text
                for col in df.select_dtypes(include='object').columns:
                    df[col] = df[col].map(code1)
                    transformations_applied.append(f"Encodage appliqué à la colonne {col}")
                
                return df
            
            # Appliquer l'encodage
            df_processed = encodage(df_processed)
            
            # Afficher un aperçu des données encodées
            st.write("#### Aperçu des données encodées")
            st.dataframe(df_processed.head())
            
            # Afficher les informations sur le DataFrame
            buffer = io.StringIO()
            df_processed.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # 3. Feature Engineering
        if "Feature Engineering" in preprocessing_options:
            st.write("### Feature Engineering")
            
            # Fonction de Feature Engineering
            def FeatureEngineering(df):
                n_before = len(df)
                
                # Créer la colonne trip_duration si les colonnes de datetime existent
                if all(col in df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
                    if pd.api.types.is_datetime64_dtype(df['tpep_pickup_datetime']) and pd.api.types.is_datetime64_dtype(df['tpep_dropoff_datetime']):
                        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
                        transformations_applied.append("Création de la colonne trip_duration (en minutes)")
                        
                        # Créer la colonne speed_mph
                        if 'trip_distance' in df.columns:
                            df['speed_mph'] = df['trip_distance'] / (df['trip_duration'] / 60)
                            transformations_applied.append("Création de la colonne speed_mph")
                            
                            # Filtrer les vitesses aberrantes
                            df = df[(df['speed_mph'] > 0) & (df['speed_mph'] < 100)]
                            n_after = len(df)
                            if n_before > n_after:
                                transformations_applied.append(f"Filtrage des vitesses aberrantes: {n_before - n_after} lignes supprimées")
                
                return df
            
            # Appliquer le Feature Engineering
            df_processed = FeatureEngineering(df_processed)
            
            # Afficher un aperçu des nouvelles features
            st.write("#### Nouvelles features créées")
            
            if 'trip_duration' in df_processed.columns and 'speed_mph' in df_processed.columns:
                st.dataframe(df_processed[['trip_duration', 'speed_mph']].describe())
                
                # Visualiser les distributions des nouvelles features
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
            
            # Identifier les colonnes numériques
            numeric_columns = df_processed.select_dtypes(include=['float', 'int']).columns.tolist()
            
            # Exclure certaines colonnes comme les identifiants et la cible
            exclude_from_scaling = ['Unnamed: 0', 'is_generous', 'VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']
            numeric_columns = [col for col in numeric_columns if col not in exclude_from_scaling and col in df_processed.columns]
            
            st.write("#### Colonnes à standardiser:")
            st.write(", ".join(numeric_columns))
            
            if numeric_columns:
                # Appliquer la standardisation
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
                # Sauvegarder les données avant standardisation pour comparaison
                df_before = df_processed[numeric_columns].head().copy()
                
                # Appliquer la standardisation
                df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
                transformations_applied.append(f"Standardisation appliquée à {len(numeric_columns)} colonnes numériques")
                
                # Afficher les résultats de la standardisation
                st.write("#### Comparaison avant/après standardisation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Avant standardisation")
                    st.dataframe(df_before)
                
                with col2:
                    st.write("Après standardisation")
                    st.dataframe(df_processed[numeric_columns].head())
                
                # Visualiser les distributions pour TOUTES les colonnes standardisées
                st.write("#### Visualisation de la standardisation pour toutes les colonnes")
                
                # Déterminer le nombre de colonnes pour organiser les graphiques
                num_cols = len(numeric_columns)
                
                # Créer une grille de sous-graphiques
                if num_cols > 0:
                    # Calculer le nombre de lignes nécessaires (2 graphiques par ligne)
                    num_rows = (num_cols + 1) // 2
                    
                    # Créer des paires de colonnes pour les graphiques
                    for i in range(0, num_cols, 2):
                        col1, col2 = st.columns(2)
                        
                        # Premier graphique de la paire
                        with col1:
                            fig = plt.figure(figsize=(10, 6))
                            sns.histplot(df_processed[numeric_columns[i]], kde=True)
                            plt.title(f'Distribution de {numeric_columns[i]} après standardisation')
                            plt.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # Second graphique de la paire (s'il existe)
                        if i + 1 < num_cols:
                            with col2:
                                fig = plt.figure(figsize=(10, 6))
                                sns.histplot(df_processed[numeric_columns[i+1]], kde=True)
                                plt.title(f'Distribution de {numeric_columns[i+1]} après standardisation')
                                plt.grid(True, alpha=0.3)
                                st.pyplot(fig)
            else:
                st.warning("Aucune colonne numérique disponible pour la standardisation.")
        
        # Résumé des transformations
        if transformations_applied:
            st.write("## Résumé des transformations appliquées")
            
            for i, transformation in enumerate(transformations_applied, 1):
                st.write(f"{i}. {transformation}")
            
            # Aperçu des données transformées
            st.write("## Aperçu des données transformées")
            st.dataframe(df_processed.head())
            
            # Informations sur les données transformées
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Dimensions")
                st.write(f"Lignes: {df_processed.shape[0]}")
                st.write(f"Colonnes: {df_processed.shape[1]}")
            
            with col2:
                st.write("### Types de données")
                dtypes_count = df_processed.dtypes.value_counts()
                
                # Afficher sous forme de diagramme
                fig = plt.figure(figsize=(8, 6))
                plt.pie(dtypes_count, labels=dtypes_count.index, autopct='%1.1f%%')
                plt.title('Répartition des types de données après transformation')
                st.pyplot(fig)
            
            # Option de téléchargement
            st.download_button(
                label="Télécharger les données transformées (CSV)",
                data=df_processed.to_csv(index=False).encode('utf-8'),
                file_name="yellow_taxi_processed.csv",
                mime="text/csv"
            )
        
        else:
            st.info("Aucune transformation n'a été appliquée aux données.")

elif page == "🤖 Modélisation":
    display_header()
    
    # Importations nécessaires pour les modèles
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    from sklearn.naive_bayes import GaussianNB
    
    # Importations pour l'évaluation des modèles
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report, 
        ConfusionMatrixDisplay
    )
    
    # Importations pour le prétraitement et la sélection de features
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Définition des fonctions pour le prétraitement exact comme demandé
    def nettoyage(df):
        df_clean = df.copy()
        
        # Colonnes à supprimer
        cols_to_drop = ['extra', 'mta_tax','Unnamed: 0']
        
        # Vérifier si ces colonnes existent avant de les supprimer
        cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(cols_to_drop, axis=1)
        
        # Filtrer les lignes selon les conditions
        n_before = len(df_clean)
        
        # Vérifier que les colonnes existent avant de filtrer
        conditions = []
        
        if "passenger_count" in df_clean.columns:
            conditions.append(df_clean["passenger_count"] != 0)
        
        if "trip_distance" in df_clean.columns:
            conditions.append(df_clean["trip_distance"] > 0)
        
        if "fare_amount" in df_clean.columns:
            conditions.append(df_clean["fare_amount"] > 0)
        
        if "total_amount" in df_clean.columns:
            conditions.append(df_clean["total_amount"] > 0)
        
        # Gérer pickup_minute s'il existe ou le créer si tpep_pickup_datetime existe
        if "tpep_pickup_datetime" in df_clean.columns and pd.api.types.is_datetime64_dtype(df_clean["tpep_pickup_datetime"]):
            if "pickup_minute" not in df_clean.columns:
                df_clean["pickup_minute"] = df_clean["tpep_pickup_datetime"].dt.minute
        
        if "pickup_minute" in df_clean.columns:
            conditions.append(df_clean["pickup_minute"] > 0)
        
        # Appliquer les conditions de filtrage si elles existent
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
            
            df_clean = df_clean[combined_condition].copy()
        
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
        
        # Appliquer l'encodage uniquement sur les colonnes object/text
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = df_encoded[col].map(code1)
        
        return df_encoded
    
    def extrairedetime(df):
        """Extrait les variables temporelles à partir des colonnes datetime"""
        df_copy = df.copy()
        datetime_cols = [col for col in df_copy.columns if 'datetime' in col.lower()]
        
        for col in datetime_cols:
            df_copy[col] = pd.to_datetime(df_copy[col])
            col_prefix = col.split('_')[0]
            
            # Extraire les composantes temporelles
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
        n_before = len(df_fe)
        
        # Créer la colonne trip_duration si les colonnes de datetime existent
        if all(col in df_fe.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
            if pd.api.types.is_datetime64_dtype(df_fe['tpep_pickup_datetime']) and pd.api.types.is_datetime64_dtype(df_fe['tpep_dropoff_datetime']):
                df_fe['trip_duration'] = (df_fe['tpep_dropoff_datetime'] - df_fe['tpep_pickup_datetime']).dt.total_seconds() / 60
                
                # Créer la colonne speed_mph
                if 'trip_distance' in df_fe.columns:
                    # Éviter la division par zéro
                    non_zero_duration = df_fe['trip_duration'].replace(0, np.nan)
                    df_fe['speed_mph'] = df_fe['trip_distance'] / (non_zero_duration / 60)
                    df_fe['speed_mph'] = df_fe['speed_mph'].fillna(0)
                    
                    # Filtrer les vitesses aberrantes
                    df_fe = df_fe[(df_fe['speed_mph'] > 0) & (df_fe['speed_mph'] < 100)]
        
        # Créer la variable cible si possible
        if all(col in df_fe.columns for col in ['tip_amount', 'fare_amount']):
            # Éviter la division par zéro
            non_zero_fare = df_fe['fare_amount'].replace(0, np.nan)
            df_fe['tip_percentage'] = (df_fe['tip_amount'] / non_zero_fare) * 100
            df_fe['tip_percentage'] = df_fe['tip_percentage'].fillna(0)
            df_fe['is_generous'] = (df_fe['tip_percentage'] > 20).astype(int)
        
        return df_fe
    
    def standardscler(df, cols):
        """Standardise les colonnes numériques spécifiées"""
        df_std = df.copy()
        
        # Exclure certaines colonnes comme les identifiants et la cible
        exclude_from_scaling = ['Unnamed: 0', 'is_generous', 'VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']
        cols = [col for col in cols if col not in exclude_from_scaling and col in df_std.columns]
        
        if cols:
            scaler = StandardScaler()
            df_std[cols] = scaler.fit_transform(df_std[cols])
        
        return df_std
    
    # Fonction pour déterminer les features par défaut disponibles
    def safe_default_features(df):
        """Retourne une liste de features qui existent dans le dataframe"""
        suggested_features = ['pickup_week', 'pickup_month', 'fare_amount',
                            'trip_distance', 'RatecodeID', 'payment_type', 'speed_mph']
        
        # Vérifier quelles features existent réellement dans le dataframe
        available_features = [f for f in suggested_features if f in df.columns]
        
        # S'il n'y a pas assez de features disponibles, ajouter d'autres colonnes numériques
        if len(available_features) < 3:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cols_to_exclude = ['is_generous', 'tip_percentage', 'tip_amount', 'Unnamed: 0']
            additional_features = [col for col in numeric_cols if col not in cols_to_exclude and col not in available_features]
            available_features.extend(additional_features[:min(5, len(additional_features))])
        
        return available_features[:min(7, len(available_features))]
    
    modeling_sections = ["Sélection de features", 
                        "Algorithmes de classification", 
                        "Optimisation d'Hyperparamètres",
                        "Comparaison des Modèles",
                        "Test des Modèles"]
    
    modeling_section = st.radio("Sélectionnez une section:", modeling_sections, horizontal=True)
    
    # Vérifier si les données ont été chargées
    if df is None or df.empty:
        st.error("Aucun jeu de données n'a été chargé. Veuillez importer des données avant de continuer.")
    else:
        # Appliquer le prétraitement complet
        st.write("### Prétraitement des données")
        datatrainset_clean = df.copy()
        
        # Application du pipeline de prétraitement
        with st.spinner("Prétraitement des données en cours..."):
            datatrainset_clean = extrairedetime(datatrainset_clean)
            datatrainset_clean = nettoyage(datatrainset_clean)
            datatrainset_clean = encodage(datatrainset_clean)
            datatrainset_clean = FeatureEngineering(datatrainset_clean)
            
            # Standardisation des colonnes numériques
            numerical_cols = datatrainset_clean.select_dtypes(include=['float64', 'int64']).columns
            datatrainset_clean = standardscler(datatrainset_clean, numerical_cols)
            
            st.success("Prétraitement terminé avec succès!")
        
        # Vérifier que la variable cible existe
        if 'is_generous' not in datatrainset_clean.columns:
            st.error("La variable cible 'is_generous' n'a pas été créée. Vérifiez que le DataFrame contient les colonnes tip_amount et fare_amount.")
        else:
            # Définir les features par défaut en fonction des colonnes disponibles
            default_features = safe_default_features(datatrainset_clean)
            st.info(f"Features disponibles par défaut : {', '.join(default_features)}")
            
            # Sélection des features
            if modeling_section == "Sélection de features":
                st.write("## Sélection des features pour la modélisation")
                
                # Identifier les colonnes numériques et catégorielles
                numeric_cols = datatrainset_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = datatrainset_clean.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Exclure certaines colonnes
                cols_to_exclude = ['is_generous', 'tip_percentage', 'tip_amount', 'Unnamed: 0']
                numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
                
                # Interface utilisateur pour sélectionner les features
                st.write("### Sélection manuelle des features")
                
                # Sélection des features numériques avec vérification
                selected_numeric_features = st.multiselect(
                    "Sélectionnez les features numériques:",
                    numeric_cols,
                    default=[f for f in default_features if f in numeric_cols]
                )
                
                # Sélection des features catégorielles avec vérification
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
                    
                    # Méthodes de sélection automatique de features
                    st.write("### Sélection automatique des features")
                    
                    # Préparation des données pour la sélection de features
                    X = datatrainset_clean[selected_features].copy()
                    y = datatrainset_clean['is_generous'].copy()
                    
                    # Sélection de features avec Recursive Feature Elimination
                    if st.checkbox("Utiliser la Recursive Feature Elimination (RFE)"):
                        st.write("#### Recursive Feature Elimination")
                        
                        n_features_to_select = st.slider(
                            "Nombre de features à sélectionner:", 
                            min_value=1, 
                            max_value=min(20, len(X.columns)), 
                            value=min(5, len(X.columns))
                        )
                        
                        # Split des données
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        # Modèle de base pour RFE
                        base_model = RandomForestClassifier(random_state=42)
                        
                        # Appliquer RFE
                        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
                        rfe.fit(X_train, y_train)
                        
                        # Récupérer les features sélectionnées
                        selected_features_rfe = X.columns[rfe.support_]
                        
                        st.write(f"Features sélectionnées par RFE: {', '.join(selected_features_rfe)}")
                        
                        # Afficher l'importance des features
                        feature_ranking = pd.DataFrame({
                            'Feature': X.columns,
                            'Ranking': rfe.ranking_
                        }).sort_values('Ranking')
                        
                        st.dataframe(feature_ranking)
                        
                        # Plot des features les plus importantes
                        fig = plt.figure(figsize=(10, 6))
                        plt.barh(feature_ranking['Feature'][:10], 1/feature_ranking['Ranking'][:10])
                        plt.title('Importance des features selon RFE')
                        plt.xlabel('Importance relative (1/Ranking)')
                        plt.gca().invert_yaxis()  # Pour afficher la feature la plus importante en haut
                        st.pyplot(fig)
                    
                    # Sélection de features avec Feature Importance de Random Forest
                    if st.checkbox("Utiliser l'importance des features avec Random Forest"):
                        st.write("#### Importance des features avec Random Forest")
                        
                        # Split des données
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        # Entraîner un Random Forest
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        
                        # Récupérer l'importance des features
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.dataframe(feature_importance)
                        
                        # Plot des features les plus importantes
                        fig = plt.figure(figsize=(10, 6))
                        plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
                        plt.title('Importance des features selon Random Forest')
                        plt.xlabel('Importance')
                        plt.gca().invert_yaxis()  # Pour afficher la feature la plus importante en haut
                        st.pyplot(fig)
                        
                        # Sélection basée sur l'importance
                        importance_threshold = st.slider(
                            "Seuil d'importance pour la sélection:", 
                            min_value=0.0, 
                            max_value=float(feature_importance['Importance'].max()), 
                            value=0.01,
                            step=0.01
                        )
                        
                        selected_features_rf = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
                        
                        st.write(f"Features sélectionnées (importance > {importance_threshold}): {', '.join(selected_features_rf)}")
                    
                    # Stockage des features sélectionnées dans la session
                    if st.button("Utiliser ces features pour la modélisation"):
                        # Stocker les features sélectionnées
                        st.session_state['selected_features'] = selected_features
                        st.success(f"Les {len(selected_features)} features sélectionnées ont été enregistrées pour la modélisation!")
                        
                        # Afficher un aperçu des données avec les features sélectionnées
                        st.write("### Aperçu des données avec les features sélectionnées")
                        st.dataframe(datatrainset_clean[selected_features + ['is_generous']].head())
            
            elif modeling_section == "Algorithmes de classification":
                st.write("## Algorithmes de classification")
                
                # Vérifier si des features ont été sélectionnées
                if 'selected_features' not in st.session_state:
                    st.info("Aucune feature n'a été sélectionnée. Les features par défaut seront utilisées.")
                    selected_features = safe_default_features(datatrainset_clean)
                    st.session_state['selected_features'] = selected_features
                else:
                    selected_features = st.session_state['selected_features']
                    # Vérifier que toutes les features sélectionnées existent
                    missing_features = [f for f in selected_features if f not in datatrainset_clean.columns]
                    if missing_features:
                        st.warning(f"Attention : Les features suivantes sont manquantes dans les données : {', '.join(missing_features)}")
                        # Filtrer pour n'utiliser que les features disponibles
                        selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                        if not selected_features:  # Si aucune feature valide ne reste
                            selected_features = safe_default_features(datatrainset_clean)
                        st.session_state['selected_features'] = selected_features
                
                st.write(f"Features utilisées: {', '.join(selected_features)}")
                
                # Préparation des données
                X = datatrainset_clean[selected_features].copy()
                y = datatrainset_clean['is_generous'].copy()
                
                # Split des données
                test_size = st.slider("Proportion du jeu de test:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                st.write(f"Dimensions des jeux de données: X_train: {X_train.shape}, X_test: {X_test.shape}")
                
                # Sélection de l'algorithme
                algorithm = st.selectbox(
                    "Choisissez un algorithme de classification:",
                    ["Régression Logistique", "Random Forest", "Support Vector Machine (SVM)", 
                     "K-Nearest Neighbors (KNN)", "Réseau de Neurones", "Arbre de Décision", 
                     "Naive Bayes"]
                )
                
                # Entraînement et évaluation du modèle
                if st.button("Entraîner le modèle"):
                    # Afficher la progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialisation des métriques
                    metrics = {}
                    model = None
                    
                    # Logique pour chaque algorithme basée sur le code Jupyter
                    if algorithm == "Support Vector Machine (SVM)":
                        status_text.text("Entraînement du modèle SVM...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans le code Jupyter
                        model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Pas d'importance de features directement disponible pour SVM
                        metrics['feature_importance'] = None
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'Blues'
                        
                    elif algorithm == "Régression Logistique":
                        status_text.text("Entraînement du modèle de Régression Logistique...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans le code Jupyter
                        model = LogisticRegression(max_iter=1000, penalty=None, solver='lbfgs')
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Coefficients du modèle
                        coef_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Coefficient': model.coef_[0]
                        }).sort_values('Coefficient', ascending=False)
                        
                        metrics['feature_importance'] = coef_df
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'Blues'
                        
                    elif algorithm == "Arbre de Décision":
                        status_text.text("Entraînement du modèle Arbre de Décision...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans le code Jupyter
                        model = DecisionTreeClassifier(
                            max_depth=3,
                            min_samples_split=20,
                            criterion='gini',
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        # Récupérer les règles de l'arbre en texte
                        tree_rules = export_text(model, feature_names=list(X.columns))
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred),
                            'tree_rules': tree_rules
                        }
                        
                        # Importance des features pour l'arbre de décision
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        metrics['feature_importance'] = feature_importance
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'YlGnBu'
                        
                    elif algorithm == "Naive Bayes":
                        status_text.text("Entraînement du modèle Naive Bayes...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans le code Jupyter
                        model = GaussianNB()
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Paramètres du modèle (moyennes et variances)
                        if hasattr(model, 'theta_') and hasattr(model, 'var_'):
                            metrics['theta'] = pd.DataFrame(model.theta_, columns=X.columns, index=['Classe 0', 'Classe 1'])
                            metrics['var'] = pd.DataFrame(model.var_, columns=X.columns, index=['Classe 0', 'Classe 1'])
                        
                        # Pas d'importance de features directement disponible pour Naive Bayes
                        metrics['feature_importance'] = None
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'OrRd'
                        
                    elif algorithm == "Random Forest":
                        status_text.text("Entraînement du modèle Random Forest...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans votre code
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Importance des features
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        metrics['feature_importance'] = feature_importance
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'Greens'
                        
                    elif algorithm == "K-Nearest Neighbors (KNN)":
                        status_text.text("Entraînement du modèle KNN...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans votre code
                        model = KNeighborsClassifier(n_neighbors=5)
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Pas d'importance de features directement disponible pour KNN
                        metrics['feature_importance'] = None
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'Oranges'
                        
                    elif algorithm == "Réseau de Neurones":
                        status_text.text("Entraînement du réseau de neurones...")
                        progress_bar.progress(25)
                        
                        # Entraînement - EXACTEMENT comme dans votre code
                        model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Ajouter l'option pour les prédictions personnalisées
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Créer un formulaire pour les inputs utilisateur
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
                        
                        # Bouton pour prédire
                        if st.button("Prédire avec mes valeurs"):
                            # Créer un DataFrame avec les valeurs de l'utilisateur
                            user_df = pd.DataFrame([user_inputs])
                            
                            # Effectuer la prédiction
                            user_pred = model.predict(user_df)[0]
                            user_proba = model.predict_proba(user_df)[0, 1]
                            
                            # Afficher le résultat
                            if user_pred == 1:
                                st.success(f"Prédiction: Client généreux ✓ (Probabilité: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error(f"Prédiction: Client non généreux ✗ (Probabilité d'être généreux: {user_proba:.2%})")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        progress_bar.progress(50)
                        status_text.text("Évaluation du modèle...")
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calcul des métriques
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        # Pas d'importance de features directement disponible pour MLP
                        metrics['feature_importance'] = None
                        
                        # Couleur pour la matrice de confusion
                        metrics['cmap'] = 'Purples'
                    
                    progress_bar.progress(75)
                    status_text.text("Génération des visualisations...")
                    
                    # Sauvegarder le modèle et les métriques
                    st.session_state['current_model'] = model
                    st.session_state['current_model_name'] = algorithm
                    st.session_state['current_metrics'] = metrics
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    
                    progress_bar.progress(100)
                    status_text.text("Modèle entraîné avec succès!")
                    
                    # Ajouter le modèle à la liste des modèles entraînés
                    if 'models_results' not in st.session_state:
                        st.session_state['models_results'] = []
                        
                    # Vérifier si un modèle avec le même nom existe déjà
                    existing_model_idx = None
                    for i, result in enumerate(st.session_state['models_results']):
                        if result['Model'] == algorithm:
                            existing_model_idx = i
                            break
                    
                    model_result = {
                        'Model': algorithm,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1,
                        'ROC AUC': roc_auc,
                        'Model Object': model
                    }
                    
                    if existing_model_idx is not None:
                        # Mettre à jour le modèle existant
                        st.session_state['models_results'][existing_model_idx] = model_result
                    else:
                        # Ajouter un nouveau modèle
                        st.session_state['models_results'].append(model_result)
                
                # Affichage des résultats si un modèle a été entraîné
                if 'current_model' in st.session_state and 'current_metrics' in st.session_state:
                    model = st.session_state['current_model']
                    model_name = st.session_state['current_model_name']
                    metrics = st.session_state['current_metrics']
                    
                    st.write(f"## Résultats du modèle {model_name}")
                    
                    # Métriques principales
                    st.write("### Métriques de performance")
                    
                    # Box de métriques stylisée comme dans le notebook
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
                    
                    # Affichage de l'importance des features si disponible
                    if metrics['feature_importance'] is not None:
                        st.write("### Importance des features")
                        
                        # Afficher le tableau d'importance
                        st.dataframe(metrics['feature_importance'])
                        
                        # Graphique d'importance des features
                        fig = plt.figure(figsize=(10, 8))
                        importance_df = metrics['feature_importance'].sort_values('Importance' if 'Importance' in metrics['feature_importance'].columns else 'Coefficient', ascending=False).head(10)
                        
                        if 'Importance' in importance_df.columns:
                            plt.barh(importance_df['Feature'], importance_df['Importance'])
                            plt.title(f"Top 10 des features les plus importantes - {model_name}")
                            plt.xlabel('Importance')
                        else:
                            plt.barh(importance_df['Feature'], importance_df['Coefficient'])
                            plt.title(f"Top 10 des coefficients - {model_name}")
                            plt.xlabel('Coefficient')
                        
                        plt.ylabel('Feature')
                        plt.gca().invert_yaxis()  # Pour afficher la feature la plus importante en haut
                        plt.grid(True, axis='x', alpha=0.3)
                        st.pyplot(fig)
                    
                    # Visualisation spécifique pour Arbre de Décision
                    if model_name == "Arbre de Décision":
                        st.write("### Structure de l'arbre de décision")
                        
                        # Afficher la structure de l'arbre sous forme de texte
                        st.write("#### Règles de l'arbre (version texte)")
                        st.text(metrics['tree_rules'])
                        
                        # Visualisation graphique de l'arbre
                        st.write("#### Visualisation graphique de l'arbre")
                        
                        fig = plt.figure(figsize=(20, 10))
                        plot_tree(model, 
                                 feature_names=list(X_test.columns),
                                 class_names=['Non Généreux', 'Généreux'],
                                 filled=True,
                                 rounded=True,
                                 proportion=True)
                        plt.title("Structure de l'arbre de décision")
                        st.pyplot(fig)
                    
                    # Visualisation des paramètres du modèle Naive Bayes
                    if model_name == "Naive Bayes" and 'theta' in metrics and 'var' in metrics:
                        st.write("### Paramètres du modèle (GaussianNB)")
                        
                        st.write("#### Moyennes par classe:")
                        st.dataframe(metrics['theta'])
                        
                        st.write("#### Variances par classe:")
                        st.dataframe(metrics['var'])
                    
                    # Visualisation avec PCA pour les modèles
                    st.write("### Visualisation avec PCA")
                    
                    X_train = st.session_state['X_train']
                    y_train = st.session_state['y_train']
                    
                    # Réduire à 2 dimensions avec PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)
                    
                    # Créer une grille pour les frontières
                    try:
                        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                          np.linspace(y_min, y_max, 100))
                        
                        # Prédictions sur la grille
                        Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
                        Z = Z.reshape(xx.shape)
                        
                        # Tracé
                        fig = plt.figure(figsize=(10, 7))
                        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
                        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
                                           edgecolors='k', cmap=plt.cm.coolwarm)
                        plt.xlabel('Composante Principale 1')
                        plt.ylabel('Composante Principale 2')
                        plt.title(f'Frontières de Décision (Réduction PCA) - {model_name}')
                        plt.colorbar(scatter, label='is_generous')
                        st.pyplot(fig)
                        
                        # Variance expliquée par les composantes principales
                        st.write(f"Variance expliquée par les deux premières composantes principales: {pca.explained_variance_ratio_.sum():.2%}")
                    except Exception as e:
                        st.warning(f"La visualisation PCA n'a pas pu être générée pour ce modèle: {str(e)}")
                        
                    # Prédictions sur des exemples du jeu de test
                    st.write("### Exemples de prédictions")
                    
                    # Sélectionner quelques exemples du jeu de test
                    n_samples = min(5, len(st.session_state['X_test']))
                    sample_indices = np.random.choice(len(st.session_state['X_test']), n_samples, replace=False)
                    
                    X_samples = st.session_state['X_test'].iloc[sample_indices]
                    y_samples = st.session_state['y_test'].iloc[sample_indices]
                    
                    # Prédire
                    y_pred_samples = model.predict(X_samples)
                    y_proba_samples = model.predict_proba(X_samples)[:, 1]
                    
                    # Créer un DataFrame pour afficher les résultats
                    sample_results = pd.DataFrame({
                        'Example': range(1, n_samples + 1),
                        'Vraie classe': y_samples.values,
                        'Prédiction': y_pred_samples,
                        'Probabilité': y_proba_samples,
                        'Correct': y_samples.values == y_pred_samples
                    })
                    
                    # Coloriser le DataFrame
                    def highlight_correct(val):
                        color = 'lightgreen' if val else 'lightcoral'
                        return f'background-color: {color}'
                    
                    st.dataframe(sample_results.style.applymap(highlight_correct, subset=['Correct']))
            
            elif modeling_section == "Optimisation d'Hyperparamètres":
                st.write("## Optimisation d'Hyperparamètres")
                
                # Vérifier si des features ont été sélectionnées
                if 'selected_features' not in st.session_state:
                    st.info("Aucune feature n'a été sélectionnée. Les features par défaut seront utilisées.")
                    selected_features = safe_default_features(datatrainset_clean)
                    st.session_state['selected_features'] = selected_features
                else:
                    selected_features = st.session_state['selected_features']
                    # Vérifier que toutes les features sélectionnées existent
                    missing_features = [f for f in selected_features if f not in datatrainset_clean.columns]
                    if missing_features:
                        st.warning(f"Attention : Les features suivantes sont manquantes dans les données : {', '.join(missing_features)}")
                        # Filtrer pour n'utiliser que les features disponibles
                        selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                        if not selected_features:  # Si aucune feature valide ne reste
                            selected_features = safe_default_features(datatrainset_clean)
                        st.session_state['selected_features'] = selected_features
                
                st.write(f"Features utilisées: {', '.join(selected_features)}")
                
                # Préparation des données
                X = datatrainset_clean[selected_features].copy()
                y = datatrainset_clean['is_generous'].copy()
                
                # Split des données
                test_size = st.slider("Proportion du jeu de test:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                st.write(f"Dimensions des jeux de données: X_train: {X_train.shape}, X_test: {X_test.shape}")
                
                # Sélection de l'algorithme pour l'optimisation
                algorithm = st.selectbox(
                    "Choisissez un algorithme pour l'optimisation des hyperparamètres:",
                    ["Régression Logistique", "Random Forest", "Support Vector Machine (SVM)", 
                     "K-Nearest Neighbors (KNN)", "Réseau de Neurones", "Arbre de Décision", 
                     "Naive Bayes"]
                )
                
                # Options de recherche
                search_method = st.radio(
                    "Méthode de recherche d'hyperparamètres:",
                    ["GridSearchCV", "RandomizedSearchCV"]
                )
                
                cv_folds = st.slider("Nombre de folds pour la validation croisée:", min_value=2, max_value=10, value=5)
                
                scoring = st.selectbox(
                    "Métrique d'optimisation:",
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]
                )
                
                # Définition des hyperparamètres à optimiser selon l'algorithme
                if algorithm == "Régression Logistique":
                    st.write("### Hyperparamètres pour la Régression Logistique")
                    
                    C_values = st.multiselect(
                        "Valeurs de C (inverse de la régularisation):",
                        [0.01, 0.1, 1, 10, 100],
                        default=[0.1, 1, 10]
                    )
                    
                    penalty_options = st.multiselect(
                        "Types de pénalité:",
                        ["l1", "l2", "none"],
                        default=["l2"]
                    )
                    
                    solver_options = st.multiselect(
                        "Solveurs:",
                        ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
                        default=["liblinear"]
                    )
                    
                    param_grid = {
                        'C': C_values,
                        'penalty': [p for p in penalty_options if p != "none"] or [None],
                        'solver': solver_options
                    }
                    
                    base_model = LogisticRegression(max_iter=1000)
                
                elif algorithm == "Random Forest":
                    st.write("### Hyperparamètres pour Random Forest")
                    
                    n_estimators_values = st.multiselect(
                        "Nombre d'arbres (n_estimators):",
                        [50, 100, 200, 300],
                        default=[50, 100, 200]
                    )
                    
                    max_depth_options = st.multiselect(
                        "Profondeur maximale des arbres (max_depth):",
                        ["None", "5", "10", "20"],
                        default=["None", "10"]
                    )
                    max_depth_values = [None if x == "None" else int(x) for x in max_depth_options]
                    
                    min_samples_split_values = st.multiselect(
                        "Nombre minimum d'échantillons pour diviser un nœud (min_samples_split):",
                        [2, 5, 10],
                        default=[2, 5]
                    )
                    
                    min_samples_leaf_values = st.multiselect(
                        "Nombre minimum d'échantillons dans une feuille (min_samples_leaf):",
                        [1, 2, 4],
                        default=[1, 2]
                    )
                    
                    param_grid = {
                        'n_estimators': n_estimators_values,
                        'max_depth': max_depth_values,
                        'min_samples_split': min_samples_split_values,
                        'min_samples_leaf': min_samples_leaf_values
                    }
                    
                    base_model = RandomForestClassifier(random_state=42)
                
                elif algorithm == "Support Vector Machine (SVM)":
                    st.write("### Hyperparamètres pour SVM")
                    
                    C_values = st.multiselect(
                        "Valeurs de C (pénalité d'erreur):",
                        [0.1, 1, 10, 100],
                        default=[0.1, 1, 10]
                    )
                    
                    kernel_options = st.multiselect(
                        "Types de kernel:",
                        ["linear", "poly", "rbf", "sigmoid"],
                        default=["linear", "rbf"]
                    )
                    
                    gamma_options = st.multiselect(
                        "Valeurs de gamma (pour kernels non linéaires):",
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
                        "Nombre de voisins (n_neighbors):",
                        [3, 5, 7, 9, 11, 15, 19],
                        default=[3, 5, 7, 9, 11, 15, 19]
                    )
                    
                    weights_options = st.multiselect(
                        "Pondération:",
                        ["uniform", "distance"],
                        default=["uniform", "distance"]
                    )
                    
                    metric_options = st.multiselect(
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
                
                elif algorithm == "Réseau de Neurones":
                    st.write("### Hyperparamètres pour le Réseau de Neurones")
                    
                    hidden_layer_sizes_options = st.multiselect(
                        "Architecture des couches cachées:",
                        ["(50,)", "(100,)", "(50, 50)", "(100, 50)", "(100, 100)"],
                        default=["(50,)", "(100,)"]
                    )
                    hidden_layer_sizes_values = [eval(x) for x in hidden_layer_sizes_options]
                    
                    activation_options = st.multiselect(
                        "Fonction d'activation:",
                        ["relu", "tanh", "logistic"],
                        default=["relu", "tanh"]
                    )
                    
                    solver_options = st.multiselect(
                        "Solveur:",
                        ["adam", "sgd", "lbfgs"],
                        default=["adam"]
                    )
                    
                    alpha_values = st.multiselect(
                        "Paramètre de régularisation (alpha):",
                        [0.0001, 0.001, 0.01],
                        default=[0.0001, 0.001]
                    )
                    
                    param_grid = {
                        'hidden_layer_sizes': hidden_layer_sizes_values,
                        'activation': activation_options,
                        'solver': solver_options,
                        'alpha': alpha_values
                    }
                    
                    base_model = MLPClassifier(max_iter=300, random_state=42)
                
                elif algorithm == "Arbre de Décision":
                    st.write("### Hyperparamètres pour l'Arbre de Décision")
                    
                    max_depth_options = st.multiselect(
                        "Profondeur maximale de l'arbre (max_depth):",
                        ["None", "3", "5", "10"],
                        default=["None", "3", "5", "10"]
                    )
                    max_depth_values = [None if x == "None" else int(x) for x in max_depth_options]
                    
                    min_samples_split_values = st.multiselect(
                        "Nombre minimum d'échantillons pour diviser un nœud (min_samples_split):",
                        [2, 5, 10],
                        default=[2, 5, 10]
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
                
                elif algorithm == "Naive Bayes":
                    st.write("### Hyperparamètres pour Naive Bayes")
                    
                    # GaussianNB n'a que quelques hyperparamètres
                    var_smoothing_values = st.multiselect(
                        "Valeurs de var_smoothing (lissage des variances):",
                        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                        default=[1e-9, 1e-8, 1e-7]
                    )
                    
                    param_grid = {
                        'var_smoothing': var_smoothing_values
                    }
                    
                    base_model = GaussianNB()
                
                # Lancement de l'optimisation
                if st.button("Lancer l'optimisation des hyperparamètres"):
                    st.write("### Optimisation en cours...")
                    
                    # Barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Préparation de la recherche d'hyperparamètres...")
                    progress_bar.progress(10)
                    
                    # Choix de la méthode de recherche
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
                            n_iter=10,  # Nombre d'itérations
                            scoring=scoring,
                            cv=cv_folds,
                            n_jobs=-1,
                            random_state=42,
                            verbose=1
                        )
                    
                    status_text.text("Lancement de la recherche d'hyperparamètres...")
                    progress_bar.progress(20)
                    
                    # Entraînement
                    search.fit(X_train, y_train)
                    
                    status_text.text("Traitement des résultats...")
                    progress_bar.progress(80)
                    
                    # Récupérer les meilleurs hyperparamètres
                    best_params = search.best_params_
                    best_score = search.best_score_
                    
                    # Entraîner le modèle final avec les meilleurs hyperparamètres
                    best_model = search.best_estimator_
                    
                    # Évaluer sur le jeu de test
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    # Calculer les métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_proba)
                    
                    # Stocker les résultats
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
                    
                    # Ajouter l'option pour les prédictions personnalisées
                    st.write("### Tester avec vos propres données")
                    st.write("Entrez les valeurs pour obtenir une prédiction:")
                    
                    # Créer un formulaire pour les inputs utilisateur
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
                    
                    # Bouton pour prédire
                    if st.button("Prédire avec le modèle optimisé"):
                        # Créer un DataFrame avec les valeurs de l'utilisateur
                        user_df = pd.DataFrame([user_inputs])
                        
                        # Effectuer la prédiction
                        user_pred = best_model.predict(user_df)[0]
                        user_proba = best_model.predict_proba(user_df)[0, 1]
                        
                        # Afficher le résultat avec style
                        st.write("### Résultat de la prédiction")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if user_pred == 1:
                                st.success("Client généreux ✓")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                            else:
                                st.error("Client non généreux ✗")
                                print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                        
                        with col2:
                            # Jauge de probabilité
                            fig, ax = plt.subplots(figsize=(8, 2))
                            ax.barh([0], [user_proba], color='green', height=0.4)
                            ax.barh([0], [1-user_proba], left=[user_proba], color='red', height=0.4)
                            ax.set_xlim(0, 1)
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            ax.set_xlabel('Probabilité d\'être généreux')
                            
                            # Ajouter la valeur de probabilité
                            ax.text(user_proba, 0, f"{user_proba:.2%}", 
                                   ha='center', va='center',
                                   color='white' if user_proba > 0.3 else 'black',
                                   fontweight='bold')
                            
                            st.pyplot(fig)
                    
                    progress_bar.progress(100)
                    status_text.text("Optimisation terminée!")
                    
                    # Stocker le modèle optimisé
                    if 'models_results' not in st.session_state:
                        st.session_state['models_results'] = []
                        
                    optimized_model_name = f"{algorithm} (Optimisé)"
                    
                    # Vérifier si un modèle optimisé avec le même nom existe déjà
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
                        # Mettre à jour le modèle existant
                        st.session_state['models_results'][existing_model_idx] = optimized_model_result
                    else:
                        # Ajouter un nouveau modèle
                        st.session_state['models_results'].append(optimized_model_result)
                
                # Affichage des résultats d'optimisation
                if 'optimization_results' in st.session_state:
                    results = st.session_state['optimization_results']
                    
                    st.write(f"## Résultats de l'optimisation pour {results['algorithm']}")
                    
                    # Afficher les meilleurs hyperparamètres
                    st.write("### Meilleurs hyperparamètres")
                    st.json(results['best_params'])
                    
                    # Afficher le score de validation croisée
                    st.write(f"### Meilleur score de validation croisée ({scoring}): {results['best_cv_score']:.4f}")
                    
                    # Afficher les métriques sur le jeu de test
                    st.write("### Performance sur le jeu de test")
                    
                    # Box de métriques stylisée comme dans le notebook
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
                        # Graphique des métriques
                        fig = plt.figure(figsize=(10, 6))
                        metrics_values = [
                            results['test_accuracy'],
                            results['test_precision'],
                            results['test_recall'],
                            results['test_f1'],
                            results['test_roc_auc']
                        ]
                        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
                        plt.bar(metrics_names, metrics_values, color='skyblue')
                        plt.ylim(0, 1)
                        plt.title(f'Métriques de performance - {results["algorithm"]} (Optimisé)')
                        plt.xticks(rotation=45)
                        plt.grid(True, axis='y', alpha=0.3)
                        st.pyplot(fig)
                    
                    # Classification report
                    st.write("### Rapport de classification détaillé")
                    st.text(results['classification_report'])
                    
                    # Matrice de confusion
                    st.write("### Matrice de confusion")
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=results['confusion_matrix'])
                    disp.plot(cmap='Blues', ax=ax)
                    plt.title(f"Matrice de confusion - {results['algorithm']} (Optimisé)")
                    st.pyplot(fig)
                    
                    # Importance des features (si disponible)
                    if hasattr(results['model'], 'feature_importances_'):
                        st.write("### Importance des features")
                        
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': results['model'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.dataframe(feature_importance)
                        
                        # Graphique d'importance des features
                        fig = plt.figure(figsize=(10, 8))
                        plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
                        plt.title(f"Top 10 des features les plus importantes - {results['algorithm']} (Optimisé)")
                        plt.xlabel('Importance')
                        plt.ylabel('Feature')
                        plt.gca().invert_yaxis()  # Pour afficher la feature la plus importante en haut
                        plt.grid(True, axis='x', alpha=0.3)
                        st.pyplot(fig)
                    elif hasattr(results['model'], 'coef_') and results['algorithm'] != "Support Vector Machine (SVM)":
                        st.write("### Coefficients des features")
                        
                        coef_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Coefficient': results['model'].coef_[0]
                        }).sort_values('Coefficient', ascending=False)
                        
                        st.dataframe(coef_df)
                        
                        # Graphique des coefficients
                        fig = plt.figure(figsize=(10, 8))
                        plt.barh(coef_df['Feature'][:10], coef_df['Coefficient'][:10])
                        plt.title(f"Top 10 des coefficients - {results['algorithm']} (Optimisé)")
                        plt.xlabel('Coefficient')
                        plt.ylabel('Feature')
                        plt.gca().invert_yaxis()  # Pour afficher la feature avec le coefficient le plus élevé en haut
                        plt.grid(True, axis='x', alpha=0.3)
                        st.pyplot(fig)
                        
                    # Visualisation spécifique pour Arbre de Décision optimisé
                    if results['algorithm'] == "Arbre de Décision":
                        st.write("### Structure de l'arbre de décision optimisé")
                        
                        # Afficher la structure de l'arbre sous forme de texte
                        tree_rules = export_text(results['model'], feature_names=list(X.columns))
                        st.write("#### Règles de l'arbre (version texte)")
                        st.text(tree_rules)
                        
                        # Visualisation graphique de l'arbre
                        st.write("#### Visualisation graphique de l'arbre")
                        
                        fig = plt.figure(figsize=(20, 10))
                        plot_tree(results['model'], 
                                 feature_names=list(X.columns),
                                 class_names=['Non Généreux', 'Généreux'],
                                 filled=True,
                                 rounded=True,
                                 proportion=True)
                        plt.title("Structure de l'arbre de décision optimisé")
                        st.pyplot(fig)
            
            elif modeling_section == "Comparaison des Modèles":
                st.write("## Comparaison des Modèles")
                
                # Vérifier si des modèles ont été entraînés
                if 'models_results' not in st.session_state or not st.session_state['models_results']:
                    st.warning("Aucun modèle n'a été entraîné. Veuillez d'abord entraîner des modèles dans la section 'Algorithmes de classification'.")
                else:
                    # Récupérer les résultats des modèles
                    models_results = st.session_state['models_results']
                    
                    # Créer un DataFrame pour la comparaison
                    models_df = pd.DataFrame(models_results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']]
                    
                    # Afficher le tableau de comparaison
                    st.write("### Tableau comparatif des modèles")
                    st.dataframe(models_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']))
                    
                    # Graphiques de comparaison
                    st.write("### Comparaison graphique des modèles")
                    
                    # Choisir les métriques à comparer
                    metrics_to_compare = st.multiselect(
                        "Sélectionnez les métriques à comparer:",
                        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                        default=['Accuracy', 'F1-Score', 'ROC AUC']
                    )
                    
                    if metrics_to_compare:
                        # Préparer les données pour le graphique
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
                        
                        # Graphique radar pour une comparaison plus visuelle
                        if len(metrics_to_compare) >= 3:
                            st.write("### Graphique radar des performances")
                            
                            # Sélectionner les modèles à inclure dans le graphique radar
                            models_to_compare = st.multiselect(
                                "Sélectionnez les modèles à comparer dans le graphique radar:",
                                models_names,
                                default=models_names[:min(3, len(models_names))]
                            )
                            
                            if models_to_compare:
                                # Préparer les données pour le graphique radar
                                filtered_models_df = models_df[models_df['Model'].isin(models_to_compare)]
                                
                                # Créer le graphique radar
                                fig = plt.figure(figsize=(10, 10))
                                ax = fig.add_subplot(111, polar=True)
                                
                                # Nombre de variables
                                N = len(metrics_to_compare)
                                
                                # Angle pour chaque variable
                                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                                angles += angles[:1]  # Fermer le graphique
                                
                                # Labels pour chaque axe
                                ax.set_xticks(angles[:-1])
                                ax.set_xticklabels(metrics_to_compare)
                                
                                # Plotting for each model
                                for idx, model in filtered_models_df.iterrows():
                                    values = [model[metric] for metric in metrics_to_compare]
                                    values += values[:1]  # Fermer le graphique
                                    
                                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model['Model'])
                                    ax.fill(angles, values, alpha=0.25)
                                
                                plt.legend(loc='upper right')
                                st.pyplot(fig)
                    
                    # Analyse détaillée d'un modèle spécifique
                    st.write("### Analyse détaillée d'un modèle")
                    
                    selected_model_name = st.selectbox(
                        "Sélectionnez un modèle pour l'analyse détaillée:",
                        [model['Model'] for model in models_results]
                    )
                    
                    # Trouver le modèle sélectionné
                    selected_model_result = next((model for model in models_results if model['Model'] == selected_model_name), None)
                    
                    if selected_model_result:
                        # Box de métriques stylisée comme dans le notebook
                        st.markdown(f"""
                        ```
                        ╔══════════════════════════════╗
                        ║       MÉTRIQUES DU MODÈLE    ║
                        ╠══════════════════════════════╣
                        ║ Accuracy: {selected_model_result['Accuracy']:.4f}             ║
                        ║ Precision: {selected_model_result['Precision']:.4f}            ║
                        ║ Recall: {selected_model_result['Recall']:.4f}               ║
                        ║ F1-score: {selected_model_result['F1-Score']:.4f}             ║
                        ║ ROC AUC: {selected_model_result['ROC AUC']:.4f}              ║
                        ╚══════════════════════════════╝
                        ```
                        """)

                        # Option pour tester avec ses propres valeurs
                        st.write("### Tester avec vos propres données")
                        st.write("Entrez les valeurs pour obtenir une prédiction:")
                        
                        # Accéder au modèle entraîné
                        model = selected_model_result['Model Object']
                        
                        # Vérifier si on a les features
                        if 'selected_features' in st.session_state:
                            selected_features = st.session_state['selected_features']
                            # Vérifier que les features existent dans le dataframe
                            available_features = [f for f in selected_features if f in datatrainset_clean.columns]
                            
                            # Créer un formulaire pour les inputs utilisateur
                            user_inputs = {}
                            for feature in available_features:
                                min_val = float(datatrainset_clean[feature].min())
                                max_val = float(datatrainset_clean[feature].max())
                                mean_val = float(datatrainset_clean[feature].mean())
                                user_inputs[feature] = st.slider(
                                    f"{feature}:", 
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    step=(max_val-min_val)/100
                                )
                            
                            # Bouton pour prédire
                            if st.button(f"Prédire avec {selected_model_name}"):
                                # Créer un DataFrame avec les valeurs de l'utilisateur
                                user_df = pd.DataFrame([user_inputs])
                                
                                # S'assurer que toutes les colonnes nécessaires sont présentes
                                if 'X_train' in st.session_state:
                                    train_cols = st.session_state['X_train'].columns
                                    
                                    # Ajouter les colonnes manquantes avec des valeurs par défaut
                                    for col in train_cols:
                                        if col not in user_df.columns:
                                            user_df[col] = 0
                                    
                                    # Supprimer les colonnes supplémentaires
                                    extra_cols = [col for col in user_df.columns if col not in train_cols]
                                    if extra_cols:
                                        user_df = user_df.drop(extra_cols, axis=1)
                                    
                                    # Réorganiser les colonnes pour qu'elles correspondent à l'ordre d'apprentissage
                                    user_df = user_df[train_cols]
                                
                                try:
                                    # Effectuer la prédiction
                                    user_pred = model.predict(user_df)[0]
                                    user_proba = model.predict_proba(user_df)[0, 1]
                                    
                                    # Afficher le résultat avec style
                                    st.write("### Résultat de la prédiction")
                                    
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        if user_pred == 1:
                                            st.success("Client généreux ✓")
                                            print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                                        else:
                                            st.error("Client non généreux ✗")
                                            print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                                    
                                    with col2:
                                        # Jauge de probabilité
                                        fig, ax = plt.subplots(figsize=(8, 2))
                                        ax.barh([0], [user_proba], color='green', height=0.4)
                                        ax.barh([0], [1-user_proba], left=[user_proba], color='red', height=0.4)
                                        ax.set_xlim(0, 1)
                                        ax.set_ylim(-0.5, 0.5)
                                        ax.set_yticks([])
                                        ax.set_xlabel('Probabilité d\'être généreux')
                                        
                                        # Ajouter la valeur de probabilité
                                        ax.text(user_proba, 0, f"{user_proba:.2%}", 
                                              ha='center', va='center',
                                              color='white' if user_proba > 0.3 else 'black',
                                              fontweight='bold')
                                        
                                        st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Erreur lors de la prédiction: {str(e)}")
                        
                        # S'il s'agit d'un modèle avec des features importantes
                        if hasattr(model, 'feature_importances_'):
                            # Vérifier si des features sont disponibles
                            if 'selected_features' in st.session_state:
                                selected_features = st.session_state['selected_features']
                                # Vérifier que les features existent dans le dataframe
                                selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                                
                                if selected_features:
                                    # Afficher l'importance des features
                                    st.write("### Importance des features")
                                    
                                    feature_importance = pd.DataFrame({
                                        'Feature': selected_features,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                                    
                                    st.dataframe(feature_importance)
                                    
                                    # Graphique d'importance des features
                                    fig = plt.figure(figsize=(10, 8))
                                    top_features = feature_importance.head(10)
                                    plt.barh(top_features['Feature'], top_features['Importance'])
                                    plt.title(f"Top 10 des features les plus importantes - {selected_model_name}")
                                    plt.xlabel('Importance')
                                    plt.ylabel('Feature')
                                    plt.gca().invert_yaxis()
                                    plt.grid(True, axis='x', alpha=0.3)
                                    st.pyplot(fig)
                        
                        # S'il s'agit d'un modèle avec des coefficients
                        elif hasattr(model, 'coef_'):
                            # Vérifier si des features sont disponibles
                            if 'selected_features' in st.session_state:
                                selected_features = st.session_state['selected_features']
                                # Vérifier que les features existent dans le dataframe
                                selected_features = [f for f in selected_features if f in datatrainset_clean.columns]
                                
                                if selected_features:
                                    # Afficher les coefficients
                                    st.write("### Coefficients des features")
                                    
                                    coef_df = pd.DataFrame({
                                        'Feature': selected_features,
                                        'Coefficient': model.coef_[0][:len(selected_features)]
                                    }).sort_values('Coefficient', ascending=False)
                                    
                                    st.dataframe(coef_df)
                                    
                                    # Graphique des coefficients positifs
                                    fig = plt.figure(figsize=(10, 8))
                                    top_coefs = coef_df.head(10)
                                    plt.barh(top_coefs['Feature'], top_coefs['Coefficient'])
                                    plt.title(f"Top 10 des coefficients positifs - {selected_model_name}")
                                    plt.xlabel('Coefficient')
                                    plt.ylabel('Feature')
                                    plt.gca().invert_yaxis()
                                    plt.grid(True, axis='x', alpha=0.3)
                                    st.pyplot(fig)
                        
                        # Obtenir les données de validation (si disponibles)
                        if 'X_test' in st.session_state and 'y_test' in st.session_state:
                            X_test = st.session_state['X_test']
                            y_test = st.session_state['y_test']
                            
                            # Recréer la prédiction
                            try:
                                y_pred = model.predict(X_test)
                                y_proba = model.predict_proba(X_test)[:, 1]
                                
                                # Matrice de confusion
                                st.write("### Matrice de confusion")
                                
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                disp.plot(cmap='Blues', ax=ax)
                                plt.title(f"Matrice de confusion - {selected_model_name}")
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Impossible de générer les visualisations: {str(e)}")
                    
                    # Sélection du meilleur modèle
                    st.write("### Sélection du meilleur modèle")
                    
                    # Choisir le critère de sélection
                    best_model_criterion = st.selectbox(
                        "Choisissez le critère pour sélectionner le meilleur modèle:",
                        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                        index=3  # F1-Score par défaut
                    )
                    
                    # Trouver le meilleur modèle selon le critère
                    best_model_idx = models_df[best_model_criterion].idxmax()
                    best_model_row = models_df.iloc[best_model_idx]
                    best_model_result = models_results[best_model_idx]
                    
                    st.success(f"Le meilleur modèle selon {best_model_criterion} est **{best_model_row['Model']}** avec un score de {best_model_row[best_model_criterion]:.4f}.")
                    
                    # Option pour sauvegarder le modèle
                    if st.button("Définir comme modèle principal"):
                        st.session_state['best_model'] = best_model_result['Model Object']
                        st.session_state['best_model_name'] = best_model_row['Model']
                        st.session_state['best_model_metrics'] = {
                            'accuracy': best_model_row['Accuracy'],
                            'precision': best_model_row['Precision'],
                            'recall': best_model_row['Recall'],
                            'f1': best_model_row['F1-Score'],
                            'roc_auc': best_model_row['ROC AUC']
                        }
                        
                        st.success(f"Le modèle **{best_model_row['Model']}** a été défini comme modèle principal!")
                    
                    # Exportation des résultats de comparaison
                    st.write("### Exportation des résultats de comparaison")
                    
                    if st.button("Exporter la comparaison des modèles (CSV)"):
                        csv = models_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Télécharger la comparaison des modèles (CSV)",
                            data=csv,
                            file_name="comparaison_modeles.csv",
                            mime="text/csv"
                        )
            
            elif modeling_section == "Test des Modèles":
                st.write("## Test des modèles sur un nouveau jeu de données")
                
                # Vérifier si un modèle a été entraîné
                if 'models_results' not in st.session_state or not st.session_state['models_results']:
                    st.warning("Aucun modèle n'a été entraîné. Veuillez d'abord entraîner des modèles dans la section 'Algorithmes de classification'.")
                else:
                    # Options de test
                    test_option = st.radio(
                        "Méthode de test:",
                        ["Tester avec de nouvelles données", "Tester avec mes propres valeurs"]
                    )
                    
                    if test_option == "Tester avec mes propres valeurs":
                        st.write("### Prédiction avec vos propres valeurs")
                        
                        # Sélection du modèle
                        model_names = [model_result['Model'] for model_result in st.session_state['models_results']]
                        selected_model_name = st.selectbox("Choisir un modèle:", model_names)
                        
                        # Récupérer le modèle sélectionné
                        selected_model = None
                        for model_result in st.session_state['models_results']:
                            if model_result['Model'] == selected_model_name:
                                selected_model = model_result['Model Object']
                                break
                        
                        if selected_model and 'selected_features' in st.session_state:
                            features = st.session_state['selected_features']
                            
                            # Si nous avons les données d'entraînement, utiliser leurs min/max
                            if 'X_train' in st.session_state:
                                X_ref = st.session_state['X_train']
                            else:
                                # Sinon utiliser les données actuelles
                                X_ref = datatrainset_clean[features]
                            
                            st.write("Entrez les valeurs pour obtenir une prédiction:")
                            
                            # Créer un formulaire pour les inputs utilisateur
                            user_inputs = {}
                            for feature in features:
                                if feature in X_ref.columns:
                                    min_val = float(X_ref[feature].min())
                                    max_val = float(X_ref[feature].max())
                                    mean_val = float(X_ref[feature].mean())
                                    
                                    # Ajouter un peu de marge aux min/max pour éviter des erreurs
                                    margin = (max_val - min_val) * 0.1
                                    min_val -= margin
                                    max_val += margin
                                    
                                    user_inputs[feature] = st.slider(
                                        f"{feature}:", 
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        step=(max_val-min_val)/100
                                    )
                                else:
                                    user_inputs[feature] = st.number_input(f"{feature}:", value=0.0)
                            
                            # Bouton pour prédire
                            if st.button("Obtenir la prédiction"):
                                # Créer un DataFrame avec les valeurs de l'utilisateur
                                user_df = pd.DataFrame([user_inputs])
                                
                                # Vérifier que toutes les colonnes nécessaires sont présentes
                                train_cols = None
                                if 'X_train' in st.session_state:
                                    train_cols = st.session_state['X_train'].columns
                                    
                                    # Ajouter les colonnes manquantes
                                    for col in train_cols:
                                        if col not in user_df.columns:
                                            user_df[col] = 0
                                    
                                    # Réorganiser pour correspondre à l'ordre d'apprentissage
                                    user_df = user_df[train_cols]
                                
                                try:
                                    # Effectuer la prédiction
                                    user_pred = selected_model.predict(user_df)[0]
                                    user_proba = selected_model.predict_proba(user_df)[0, 1]
                                    
                                    # Afficher le résultat avec style
                                    st.write("### Résultat de la prédiction")
                                    
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        if user_pred == 1:
                                            st.success("Client généreux ✓")
                                            print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT GÉNÉREUX - Probabilité: {user_proba:.2%}")
                                        else:
                                            st.error("Client non généreux ✗")
                                            print(f"RÉSULTAT DE LA PRÉDICTION: CLIENT NON GÉNÉREUX - Probabilité d'être généreux: {user_proba:.2%}")
                                    
                                    with col2:
                                        # Jauge de probabilité
                                        fig, ax = plt.subplots(figsize=(8, 2))
                                        ax.barh([0], [user_proba], color='green', height=0.4)
                                        ax.barh([0], [1-user_proba], left=[user_proba], color='red', height=0.4)
                                        ax.set_xlim(0, 1)
                                        ax.set_ylim(-0.5, 0.5)
                                        ax.set_yticks([])
                                        ax.set_xlabel('Probabilité d\'être généreux')
                                        
                                        # Ajouter la valeur de probabilité
                                        ax.text(user_proba, 0, f"{user_proba:.2%}", 
                                               ha='center', va='center',
                                               color='white' if user_proba > 0.3 else 'black',
                                               fontweight='bold')
                                        
                                        st.pyplot(fig)
                                    
                                    # Tableau récapitulatif des données entrées
                                    st.write("### Récapitulatif des données entrées")
                                    st.dataframe(user_df)
                                    
                                except Exception as e:
                                    st.error(f"Erreur lors de la prédiction: {str(e)}")
                                    st.info("Conseil: Vérifiez que les valeurs entrées sont dans les plages attendues.")
                            
                        elif not 'selected_features' in st.session_state:
                            st.error("Aucune feature n'a été sélectionnée pour la modélisation.")
                            
                            # Proposer les features par défaut
                            if st.button("Utiliser les features par défaut"):
                                default_features = safe_default_features(datatrainset_clean)
                                st.session_state['selected_features'] = default_features
                                st.success(f"Les features par défaut ont été sélectionnées: {', '.join(default_features)}")
                                st.experimental_rerun()
                    
                    else:  # "Tester avec de nouvelles données"
                        # Chargement des données de test
                        test_data_option = st.radio(
                            "Source des données de test:",
                            ["Utiliser les données actuelles", "Importer un nouveau fichier CSV"]
                        )
                        
                        test_data = None
                        
                        if test_data_option == "Utiliser les données actuelles":
                            test_data = datatrainset_clean.copy()
                            st.success("Utilisation des données actuelles comme données de test.")
                        else:
                            uploaded_file = st.file_uploader("Importer un fichier CSV pour le test", type=['csv'])
                            
                            if uploaded_file is not None:
                                try:
                                    test_data = pd.read_csv(uploaded_file)
                                    st.success(f"Fichier chargé avec succès! Dimensions: {test_data.shape}")
                                except Exception as e:
                                    st.error(f"Erreur lors du chargement du fichier: {str(e)}")
                        
                        if test_data is not None:
                            # Prétraitement des données de test
                            st.write("### Prétraitement des données de test")
                            
                            # Application du pipeline de prétraitement du notebook
                            datatestset_clean = test_data.copy()
                            
                            # Bouton pour appliquer le prétraitement
                            if st.button("Appliquer le prétraitement standard"):
                                try:
                                    with st.spinner("Prétraitement en cours..."):
                                        # Application du pipeline complet de prétraitement
                                        datatestset_clean = extrairedetime(datatestset_clean)
                                        st.info("Extraction des variables temporelles effectuée.")
                                        
                                        datatestset_clean = nettoyage(datatestset_clean)
                                        st.info("Nettoyage des données effectué.")
                                        
                                        datatestset_clean = encodage(datatestset_clean)
                                        st.info("Encodage des variables catégorielles effectué.")
                                        
                                        datatestset_clean = FeatureEngineering(datatestset_clean)
                                        st.info("Feature Engineering effectué.")
                                        
                                        # Standardisation des colonnes numériques
                                        numerical_cols = datatestset_clean.select_dtypes(include=['float64', 'int64']).columns
                                        datatestset_clean = standardscler(datatestset_clean, numerical_cols)
                                        st.info("Standardisation des colonnes numériques effectuée.")
                                        
                                        st.success("Prétraitement des données de test terminé avec succès!")
                                except Exception as e:
                                    st.error(f"Erreur lors du prétraitement: {str(e)}")
                            
                            # Afficher un aperçu des données prétraitées
                            st.write("### Aperçu des données prétraitées")
                            st.dataframe(datatestset_clean.head())
                            
                            # Sélection du modèle à utiliser pour le test
                            st.write("### Sélection du modèle pour le test")
                            
                            model_names = [model_result['Model'] for model_result in st.session_state['models_results']]
                            selected_model_name = st.selectbox("Choisir un modèle:", model_names)
                            
                            # Récupérer le modèle sélectionné
                            selected_model = None
                            
                            for model_result in st.session_state['models_results']:
                                if model_result['Model'] == selected_model_name:
                                    selected_model = model_result['Model Object']
                                    break
                            
                            if selected_model is not None:
                                # Vérifier si les features nécessaires sont présentes
                                if 'selected_features' in st.session_state:
                                    required_features = st.session_state['selected_features']
                                    
                                    # Vérifier que les features existent dans le dataframe
                                    available_features = [f for f in required_features if f in datatestset_clean.columns]
                                    missing_features = [f for f in required_features if f not in datatestset_clean.columns]
                                    
                                    if missing_features:
                                        st.warning(f"Attention : Les features suivantes sont manquantes dans les données de test : {', '.join(missing_features)}")
                                        
                                        # Choisir des features alternatives si aucune n'est disponible
                                        if not available_features:
                                            available_features = safe_default_features(datatestset_clean)
                                            st.info(f"Utilisation des features alternatives disponibles : {', '.join(available_features)}")
                                    
                                    # Préparer les données pour la prédiction
                                    X_test = datatestset_clean[available_features].copy()
                                    
                                    # S'assurer que X_test a les mêmes colonnes que celles utilisées pour l'entraînement
                                    if 'X_train' in st.session_state:
                                        train_cols = st.session_state['X_train'].columns
                                        
                                        # Ajouter les colonnes manquantes avec des valeurs par défaut
                                        for col in train_cols:
                                            if col not in X_test.columns:
                                                X_test[col] = 0
                                        
                                        # Supprimer les colonnes supplémentaires
                                        extra_cols = [col for col in X_test.columns if col not in train_cols]
                                        if extra_cols:
                                            X_test = X_test.drop(extra_cols, axis=1)
                                        
                                        # Réorganiser les colonnes pour qu'elles correspondent à l'ordre d'apprentissage
                                        X_test = X_test[train_cols]
                                    
                                    # Effectuer la prédiction
                                    if st.button("Lancer la prédiction"):
                                        st.write("### Résultats de la prédiction")
                                        
                                        try:
                                            # Prédire
                                            y_pred = selected_model.predict(X_test)
                                            y_proba = selected_model.predict_proba(X_test)[:, 1]
                                            
                                            # Ajouter les prédictions au dataframe
                                            datatestset_clean['prediction'] = y_pred
                                            datatestset_clean['probabilite'] = y_proba
                                            
                                            # Afficher les résultats
                                            st.write("#### Aperçu des prédictions")
                                            prediction_cols = ['prediction', 'probabilite']
                                            if 'is_generous' in datatestset_clean.columns:
                                                prediction_cols = ['is_generous'] + prediction_cols
                                            st.dataframe(datatestset_clean[prediction_cols].head(10))
                                            
                                            # Distribution des prédictions
                                            st.write("#### Distribution des prédictions")
                                            
                                            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                                            
                                            # Diagramme camembert des prédictions
                                            prediction_counts = pd.Series(y_pred).value_counts()
                                            labels = ['Non généreux (0)', 'Généreux (1)'] if len(np.unique(y_pred)) <= 2 else prediction_counts.index
                                            ax[0].pie(prediction_counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
                                            ax[0].set_title('Répartition des prédictions')
                                            
                                            # Histogramme des probabilités
                                            ax[1].hist(y_proba, bins=20, color='skyblue')
                                            ax[1].set_title('Distribution des probabilités')
                                            ax[1].set_xlabel('Probabilité d\'être généreux')
                                            ax[1].set_ylabel('Nombre d\'observations')
                                            ax[1].grid(True, alpha=0.3)
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            
                                            # Option pour évaluer les performances si la variable cible est présente
                                            if 'is_generous' in datatestset_clean.columns:
                                                st.write("#### Évaluation des performances")
                                                
                                                # Calculer les métriques
                                                accuracy = accuracy_score(datatestset_clean['is_generous'], y_pred)
                                                precision = precision_score(datatestset_clean['is_generous'], y_pred)
                                                recall = recall_score(datatestset_clean['is_generous'], y_pred)
                                                f1 = f1_score(datatestset_clean['is_generous'], y_pred)
                                                roc_auc = roc_auc_score(datatestset_clean['is_generous'], y_proba)
                                                
                                                # Box de métriques stylisée comme dans le notebook
                                                st.markdown(f"""                                                ╔══════════════════════════════╗
                                                ║       MÉTRIQUES DU MODÈLE    ║
                                                ╠══════════════════════════════╣
                                                ║ Accuracy: {accuracy:.4f}             ║
                                                ║ Precision: {precision:.4f}            ║
                                                ║ Recall: {recall:.4f}               ║
                                                ║ F1-score: {f1:.4f}             ║
                                                ║ ROC AUC: {roc_auc:.4f}              ║
                                                ╚══════════════════════════════╝
                                                ```
                                                """)
                                                
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    # Tableau des métriques
                                                    metrics_df = pd.DataFrame({
                                                        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                                                        'Valeur': [accuracy, precision, recall, f1, roc_auc]
                                                    })
                                                    st.dataframe(metrics_df)
                                                
                                                with col2:
                                                    # Matrice de confusion
                                                    cm = confusion_matrix(datatestset_clean['is_generous'], y_pred)
                                                    fig, ax = plt.subplots(figsize=(8, 6))
                                                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                                    disp.plot(cmap='Blues', ax=ax)
                                                    plt.title(f"Matrice de confusion - {selected_model_name}")
                                                    st.pyplot(fig)
                                                
                                                # Classification report
                                                st.write("**Rapport de classification détaillé:**")
                                                st.text(classification_report(datatestset_clean['is_generous'], y_pred))
                                            
                                            # Option pour télécharger les prédictions
                                            csv = datatestset_clean.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Télécharger les résultats (CSV)",
                                                data=csv,
                                                file_name="predictions.csv",
                                                mime="text/csv"
                                            )
                                            
                                            # Afficher les informations de prédiction dans la console
                                            print(f"RÉSULTATS DU MODÈLE {selected_model_name} SUR LE JEU DE TEST:")
                                            print(f"Nombre de prédictions 'Généreux' (1): {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred):.2%})")
                                            print(f"Nombre de prédictions 'Non généreux' (0): {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred):.2%})")
                                            
                                            # Si la variable cible est présente
                                            if 'is_generous' in datatestset_clean.columns:
                                                print(f"ÉVALUATION DES PERFORMANCES:")
                                                print(f"Accuracy: {accuracy:.4f}")
                                                print(f"Precision: {precision:.4f}")
                                                print(f"Recall: {recall:.4f}")
                                                print(f"F1-score: {f1:.4f}")
                                                print(f"ROC AUC: {roc_auc:.4f}")
                                            
                                        except Exception as e:
                                            st.error(f"Erreur lors de la prédiction: {str(e)}")
                                            st.info("Conseil: Vérifiez que les features du jeu de test correspondent à celles utilisées pour l'entraînement.")
                                else:
                                    st.error("Aucune feature n'a été sélectionnée pour la modélisation.")
                                    
                                    # Proposer les features par défaut
                                    if st.button("Utiliser les features par défaut"):
                                        default_features = safe_default_features(datatrainset_clean)
                                        st.session_state['selected_features'] = default_features
                                        st.success(f"Les features par défaut ont été sélectionnées: {', '.join(default_features)}")
                                        st.experimental_rerun()  # Rafraîchir la page
            else:
                    st.error("La variable cible 'is_generous' n'a pas pu être créée ou trouvée dans le jeu de données.")
st.markdown("---")


# Activer l'application
# Pour exécuter cette application: streamlit run app.py
                        
            
            