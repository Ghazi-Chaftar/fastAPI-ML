import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn import svm 
from sklearn.preprocessing import LabelEncoder

def potential_growth(request:dict):
    df = pd.read_excel('administration_data.xlsx')
    print(df.head())
    df['director_indicator'] = 0
    df['director_indicator'] = df['Grade'].apply(lambda x: 1 if x == 'Director' else 0)

    # Sélectionnez les colonnes que vous souhaitez inclure dans votre ensemble de données
    selected_columns = [ 'TYPE_DIPLOMA', 'EXP_YEARS',
                        'POSITION',
                        'SOURCE_of_employment', 'director_indicator','TYPE_CONTRACT']

    # Créez un nouveau DataFrame avec les colonnes sélectionnées
    df_selected = df[selected_columns]

    label_encoder = LabelEncoder()

    # Encoder les colonnes catégorielles
    colonnes_catégorielles = ['TYPE_DIPLOMA', 'POSITION', 'SOURCE_of_employment','TYPE_CONTRACT']
    for colonne in colonnes_catégorielles:
        df_selected[colonne] = label_encoder.fit_transform(df_selected[colonne])


    # Créer une instance de LabelEncoder
    label_encoder = LabelEncoder()

    # Encoder les trois colonnes catégorielles
    colonnes_catégorielles = ['TYPE_DIPLOMA', 'POSITION', 'SOURCE_of_employment','TYPE_CONTRACT']

    for colonne in colonnes_catégorielles:
        df_selected[colonne] = label_encoder.fit_transform(df_selected[colonne])
        print(f"Colonnes '{colonne}' :")
        for classe, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
            print(f"{classe} : {label}")


    df_selected.fillna(df_selected.median(), inplace=True)



    X = df_selected.drop('director_indicator', axis=1)
    y = df_selected['director_indicator']

    # Divisez les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer une instance de SMOTE
    smote = SMOTE(random_state=42)

    # Appliquer SMOTE sur les données d'entraînement
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Créer un modèle de régression logistique
    logistic_model = LogisticRegression()

    # Entraîner le modèle sur les données d'entraînement rééquilibrées
    logistic_model.fit(X_train_resampled, y_train_resampled)

    # Faire des prédictions sur l'ensemble de test
    y_pred = logistic_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy SVM:", accuracy)

    example = {
    'TYPE_DIPLOMA': int(request.get("TYPE_DIPLOMA")),
    'EXP_YEARS': int(request.get("EXP_YEARS")),
    'POSITION': int(request.get("POSITION")),
    'SOURCE_of_employment': int(request.get("SOURCE_of_employment")),
    'TYPE_CONTRACT': int(request.get("TYPE_CONTRACT"))
    }
    print("aaaaaaaaaaa",example)

    # Créez un DataFrame à partir de l'exemple
    example_df = pd.DataFrame([example])

    print(example_df)

    # Assurez-vous que les colonnes de l'exemple sont dans le même ordre que celles des données d'entraînement
    # Par exemple, si vous avez utilisé Grade_freq, TYPE_DIPLOMA_freq, EXP_YEARS, POSITION_freq, SOURCE_of_employment_freq
    # Assurez-vous que les colonnes de votre exemple sont dans le même ordre
    example_df = example_df[['TYPE_DIPLOMA','EXP_YEARS','POSITION', 'SOURCE_of_employment','TYPE_CONTRACT']]

    # Faire une prédiction sur l'exemple prétraité
    prediction = logistic_model.predict(example_df)

    # Afficher la prédiction
    print("Prediction for the example:", prediction)
    print(prediction[0])
    return str(prediction[0])