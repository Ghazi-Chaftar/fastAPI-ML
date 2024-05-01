import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def potential_growth():
    df = pd.read_excel('administration_data.xlsx')
    print(df.head())
    df['director_indicator'] = 0
    df['director_indicator'] = df['Grade'].apply(lambda x: 1 if x == 'Director' else 0)

    # Sélectionnez les colonnes que vous souhaitez inclure dans votre ensemble de données
    selected_columns = [ 'Grade',
                        'TYPE_DIPLOMA', 'EXP_YEARS',
                        'POSITION',
                        'SOURCE_of_employment', 'director_indicator']

    # Créez un nouveau DataFrame avec les colonnes sélectionnées
    df_selected = df[selected_columns]


    # Calculer la fréquence de chaque catégorie dans les variables sélectionnées
    grade_freq = df_selected['Grade'].value_counts(normalize=True)
    type_diploma_freq = df_selected['TYPE_DIPLOMA'].value_counts(normalize=True)
    position_freq = df_selected['POSITION'].value_counts(normalize=True)
    source_freq = df_selected['SOURCE_of_employment'].value_counts(normalize=True)

    # Remplacer chaque catégorie par sa fréquence dans les données
    df_selected['Grade_freq'] = df_selected['Grade'].map(grade_freq)
    df_selected['TYPE_DIPLOMA_freq'] = df_selected['TYPE_DIPLOMA'].map(type_diploma_freq)
    df_selected['POSITION_freq'] = df_selected['POSITION'].map(position_freq)
    df_selected['SOURCE_of_employment_freq'] = df_selected['SOURCE_of_employment'].map(source_freq)

    # Éliminer les colonnes originales
    df_selected.drop(columns=['Grade', 'TYPE_DIPLOMA', 'POSITION', 'SOURCE_of_employment'], inplace=True)

    df_selected.fillna(df_selected.median(), inplace=True)

    # Supposons que votre variable cible est 'director_indicator' et que toutes les autres colonnes sont des variables explicatives
    X = df_selected.drop('director_indicator', axis=1)
    y = df_selected['director_indicator']

    # Divisez les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créez un objet de modèle de régression logistique
    model = LogisticRegression()

    # Entraînez le modèle sur l'ensemble d'entraînement
    model.fit(X_train, y_train)

    # Faites des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluez les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy