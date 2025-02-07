from preprocessing.data_parser import FileOpener
from preprocessing.functions import DataPreprocessing
from model.utility import classification_evaluation
from input_managing import InputManager
import os

if __name__ == "__main__":
    file = input('Inserisci il nome del file con estensione: ')
    target_column = input('Inserisci il nome della colonna target: ')
    index_col = input('Inserisci il nome della colonna da impostare come indice (se presente): ')
    method_fill_nan = input('Scegli come fill nan values (mean or median): ')

    csv_directory = "data"
    file_path = os.path.join(csv_directory, file)

    file_opener = FileOpener()
    df = file_opener.open(file_path)

    preprocessor = DataPreprocessing(df)
    df = preprocessor.preprocessing(index_col, target_column, method_fill_nan)

    X, Y = preprocessor.features_and_target(target_column)
    
    splits = InputManager.get_user_choice_split(X, Y)

    k = int(input("Enter the value of k: "))
    
    user_choice = InputManager.get_user_choice()
    user_choice = InputManager.process_user_choice(user_choice)
    res = classification_evaluation.knn_metrics(k, splits, user_choice)