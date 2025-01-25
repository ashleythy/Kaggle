import pandas as pd 
import numpy as np
import re 
from sklearn.linear_model import LogisticRegression
from ml.TitanicPassenger import TitanicPassenger

def preprocess_names(
        name: str = 'ashley thoo'
):
    """
    Given passenger's name(s), count the number of unique name(s) and the total number of characters combined.
    
    Inputs
    ------
    name : str 
        Name(s)
    
    Returns
    ------
    len_name : int 
        Number of unique name(s)
    len_chars : int 
        Number of characters of name(s)
    """

    # number of unique names 
    len_names = len(name.split(' '))

    # number of characters
    len_chars = len(name.replace(' ','')) 

    return len_names, len_chars


def preprocess_prefix(
        prefix: str = 'mr'
):
    """
    Given passenger's prefix, return corresponding label.

    Inputs
    ------
    prefix : str 
        Prefix 

    Returns 
    ------ 
    prefix_name_Master : int 
        One hot encoded prefix for Master, Miss, Mlle, Mr, Mrs and Rev
    prefix_name_Miss : int 
    prefix_name_Mlle : int 
    prefix_name_Mr : int 
    prefix_name_Mrs : int 
    prefix_name_Rev : int 
    """

    # prefix 
    prefix = prefix.capitalize() 

    one_hot_encoded_prefix = {
        'prefix_name_Master': 0,
        'prefix_name_Miss': 0,
        'prefix_name_Mlle': 0,
        'prefix_name_Mr': 0,
        'prefix_name_Mrs': 0,
        'prefix_name_Rev': 0
    }

    one_hot_encoded_prefix[f"prefix_name_{prefix}"] = 1 

    prefix_name_Master = one_hot_encoded_prefix['prefix_name_Master']
    prefix_name_Miss = one_hot_encoded_prefix['prefix_name_Miss']
    prefix_name_Mlle = one_hot_encoded_prefix['prefix_name_Mlle']
    prefix_name_Mr = one_hot_encoded_prefix['prefix_name_Mr']
    prefix_name_Mrs = one_hot_encoded_prefix['prefix_name_Mrs']
    prefix_name_Rev = one_hot_encoded_prefix['prefix_name_Rev']

    return prefix_name_Master, prefix_name_Miss, prefix_name_Mlle, \
    prefix_name_Mr, prefix_name_Mrs, prefix_name_Rev


def preprocess_cabin(
        cabin: str = 'F45, H67'
):
    """
    Given cabin(s) a passenger booked, count the number of booked cabin(s) and extract the cabin type. 
    If multiple cabins were booked, then cabin type is based on the first cabin booked.
    Additionally, if cabin(s) were booked, assign 1, otherwise, assign 0.

    Inputs
    ------
    cabin : str 
        Cabin(s) booked 
    
    Returns 
    ------ 
    cabin_frequency : int 
        Number of cabin(s) booked 
    cabin_assigned : int 
        If cabin(s) were booked at all
    cabin_type_A : int 
        One hot encoded cabin from A to G
    cabin_type_B : int 
    cabin_type_C : int 
    cabin_type_D : int 
    cabin_type_E : int 
    cabin_type_F : int 
    cabin_type_G : int 
    """

    # cabin number pattern
    pattern = r'\b[A-Za-z]+\d+\b'

    split_cabin = re.findall(pattern, cabin)

    cabin_freqency = len(split_cabin)
    cabin_assigned = 1 if len(split_cabin) > 0 else 0
    cabin_type = split_cabin[0][0].upper()

    # refactor cabin_type to one hot encoded variable 
    one_hot_encoded_cabin_type = {
        'cabin_type_A': 0,
        'cabin_type_B': 0, 
        'cabin_type_C': 0, 
        'cabin_type_D': 0, 
        'cabin_type_E': 0,
        'cabin_type_F': 0,
        'cabin_type_G': 0
    }

    one_hot_encoded_cabin_type[f"cabin_type_{cabin_type}"] = 1

    cabin_type_A = one_hot_encoded_cabin_type['cabin_type_A']
    cabin_type_B = one_hot_encoded_cabin_type['cabin_type_B']
    cabin_type_C = one_hot_encoded_cabin_type['cabin_type_C']
    cabin_type_D = one_hot_encoded_cabin_type['cabin_type_D']
    cabin_type_E = one_hot_encoded_cabin_type['cabin_type_E']
    cabin_type_F = one_hot_encoded_cabin_type['cabin_type_F']
    cabin_type_G = one_hot_encoded_cabin_type['cabin_type_G']

    return cabin_freqency, cabin_assigned, cabin_type_A, \
    cabin_type_B, cabin_type_C, cabin_type_D, cabin_type_E, \
    cabin_type_F, cabin_type_G



def preprocess_age(
        age: int = 56
):
    """
    Given passenger's age, return corresponding label.

    Inputs
    ------ 
    age : int 
        Age
    
    Returns 
    ------ 
    age_bins_encoded : int 
        Given age mapped to corresponding label
    """

    age_bins_encoded = np.where(age > 0 and age <= 16, 0, 
                                np.where(age > 16 and age <= 32, 1,
                                        np.where(age > 32 and age <= 48, 2,
                                                np.where(age > 48 and age <= 64, 3,
                                                            np.where(age > 64 and age <= 80, 4,
                                                                    5)))))
    
    return age_bins_encoded.item()



def preprocess_passengerid(
        passengerid: int = 700
):
    """
    Given passenger's id, return corresponding label.

    Inputs
    ------ 
    passengerid : int 
        Passenger ID
    
    Returns 
    ------ 
    passengerid_bins_encoded : int 
        Given passenger id mapped to corresponding label
    """

    passengerid_bins_encoded = np.where(passengerid > 1 and passengerid <= 179, 0,
                                        np.where(passengerid > 179 and passengerid <= 357, 1,
                                                np.where(passengerid > 357 and passengerid <= 535, 2,
                                                        np.where(passengerid > 535 and passengerid <= 713, 3,
                                                                    np.where(passengerid > 713 and passengerid <= 891, 4,
                                                                            5)))))
    
    return passengerid_bins_encoded.item()



def preprocess_gender(
        gender: str = 'female'
): 
    """
    Given passenger's gender, return corresponding label.

    Inputs 
    ------
    gender : str 
        Gender

    Returns 
    ------
    sex_male : int 
        One hot encoded gender for male and female
    """

    gender = gender.lower() 

    sex_male = 1 if gender == 'male' else 0

    return sex_male



def preprocess_embarked(
        embarked: str = 's'
): 
    """ 
    Given passenger's embarkment, return corresponding label.
    
    Inputs 
    ------
    embarked : str 
        Embarkment

    Returns
    ------
    embarked_Q : int
        One hot encoded embarked for S, C and Q
    embarked_S : int
    """

    embarked = embarked.lower() 

    embarked_Q = 1 if embarked == 'q' else 0
    embarked_S = 1 if embarked == 's' else 0 

    return embarked_Q, embarked_S 



def preprocess_pclass(
        pclass: int = 1
): 
    """ 
    Given passenger's class, return corresponding label.

    Inputs
    ------
    pclass : int 
        Pclass

    Returns 
    ------
    pclass_2 : int 
        One hot encoded pclass for 1, 2 and 3
    pclass_3 : int
    """

    pclass_2 = 1 if pclass == 2 else 0 
    pclass_3 = 1 if pclass == 3 else 0 

    return pclass_2, pclass_3 



def preprocess_ticket(
        ticket: str = 'PC455'
): 
    """ 
    Given passenger's ticket, extract main ticket type and return corresponding label.

    Inputs
    ------
    ticket : str 
        Ticket number 
    
    Returns 
    ------
    ticket_type_CA : int 
        One hot encoded ticket for multiple ticket types
    ticket_type_DIGITS ONLY : int
    ticket_type_FCC : int 
    ticket_type_LINE : int 
    ticket_type_PC : int 
    ticket_type_SC : int 
    ticket_type_SCH : int 
    ticket_type_SO/PP : int
    ticket_type_SOC : int 
    ticket_type_SOTON : int 
    ticket_type_W/C : int 
    """

    alphabets = re.findall(r'\D*', ticket)
    digits = re.findall(r'^\d*', ticket)

    ticket_type = alphabets[0].replace('.','').strip() if (alphabets[0] != '') and (digits[0] == '') else 'DIGITS_ONLY'

    # remap 
    ticket_type_remap = {
        'CA': ['CA/SOTON'],
        'A': ['A/', 'A/S'],
        'SOTON': ['SOTON/O', 'SOTON/OQ', 'STON/O', 'STON/O ', 'SOTONQ'],
        'SC': ['SC/A', 'SC/AH ', 'SC/AH Basle', 'SC/PARIS ', 'SC/Paris '],
        'FCC ': ['FC ']
        }
    
    for k, v in ticket_type_remap.items():
        if ticket_type in v: 
            ticket_type = k  
    
    # one hot encode 
    one_hot_encoded_ticket_type = {
        'ticket_type_CA': 0,
        'ticket_type_DIGITS_ONLY': 0,
        'ticket_type_FCC': 0, 
        'ticket_type_LINE': 0, 
        'ticket_type_PC': 0, 
        'ticket_type_SC': 0, 
        'ticket_type_SCH': 0, 
        'ticket_type_SO/PP': 0,
        'ticket_type_SOC': 0, 
        'ticket_type_SOTON': 0, 
        'ticket_type_W/C': 0 
    }

    one_hot_encoded_ticket_type[f"ticket_type_{ticket_type}"] = 1

    ticket_type_CA = one_hot_encoded_ticket_type['ticket_type_CA'] 
    ticket_type_DIGITS_ONLY = one_hot_encoded_ticket_type['ticket_type_DIGITS_ONLY'] 
    ticket_type_FCC = one_hot_encoded_ticket_type['ticket_type_FCC'] 
    ticket_type_LINE = one_hot_encoded_ticket_type['ticket_type_LINE'] 
    ticket_type_PC = one_hot_encoded_ticket_type['ticket_type_PC'] 
    ticket_type_SC = one_hot_encoded_ticket_type['ticket_type_SC'] 
    ticket_type_SCH = one_hot_encoded_ticket_type['ticket_type_SCH'] 
    ticket_type_SO_PP = one_hot_encoded_ticket_type['ticket_type_SO/PP'] 
    ticket_type_SOC = one_hot_encoded_ticket_type['ticket_type_SOC'] 
    ticket_type_SOTON = one_hot_encoded_ticket_type['ticket_type_SOTON'] 
    ticket_type_W_C = one_hot_encoded_ticket_type['ticket_type_W/C'] 

    return ticket_type_CA, ticket_type_DIGITS_ONLY, ticket_type_FCC, \
    ticket_type_LINE, ticket_type_PC, ticket_type_SC, ticket_type_SCH, \
    ticket_type_SO_PP, ticket_type_SOC, ticket_type_SOTON, ticket_type_W_C



def preprocess_data(passenger: TitanicPassenger):
    """
    Preprocess passenger's data input into a format that is acceptable to be used in the
    trained model.
    """

    # passenger characteristics, in correct order
    # age 
    age = passenger.age 
    age_bins_encoded = preprocess_age(age)

    # sibsp 
    sibsp = passenger.number_siblings 

    # parch 
    parch = passenger.number_parch 

    # fare 
    fare = passenger.fare_price 

    # len_unq_firstname, len_char_firstname, len_unq_familyname, len_char_familyname 
    first_name = passenger.first_name 
    family_name = passenger.family_name 

    len_unq_firstname, len_char_firstname = preprocess_names(name=first_name)
    len_unq_familyname, len_char_familyname = preprocess_names(name=family_name)

    # cabin_frequency, cabin_assigned 
    cabin = passenger.cabin_name

    cabin_frequency, cabin_assigned, cabin_type_A, \
    cabin_type_B, cabin_type_C, cabin_type_D, \
    cabin_type_E, cabin_type_F, cabin_type_G = preprocess_cabin(cabin)

    # pclass 
    pclass = passenger.pclass

    pclass_2, pclass_3 = preprocess_pclass(pclass)

    # gender 
    gender = passenger.gender 

    sex_male = preprocess_gender(gender)

    # embarkment 
    embarked = passenger.embarked_port 

    embarked_Q, embarked_S = preprocess_embarked(embarked) 

    # prefix 
    prefix = passenger.prefix

    prefix_name_Master, prefix_name_Miss, prefix_name_Mlle, \
    prefix_name_Mr, prefix_name_Mrs, prefix_name_Rev = preprocess_prefix(prefix)

    # ticket type 
    ticket = passenger.ticket_name

    ticket_type_CA, ticket_type_DIGITS_ONLY, ticket_type_FCC, \
    ticket_type_LINE, ticket_type_PC, ticket_type_SC, ticket_type_SCH, \
    ticket_type_SO_PP, ticket_type_SOC, ticket_type_SOTON, ticket_type_W_C = preprocess_ticket(ticket)

    # passenger id 
    passengerid = passenger.passenger_id
    passengerid_bins_encoded = preprocess_passengerid(passengerid)

    # rearrange into a list -> array -> df
    preprocessed_list = [
    age, sibsp, parch, fare, len_unq_firstname, len_char_firstname, \
    len_unq_familyname, len_char_familyname, cabin_frequency, cabin_assigned, \
    pclass_2, pclass_3, sex_male, embarked_Q, embarked_S, prefix_name_Master, \
    prefix_name_Miss, prefix_name_Mlle, prefix_name_Mr, prefix_name_Mrs, \
    prefix_name_Rev, ticket_type_CA, ticket_type_DIGITS_ONLY, ticket_type_FCC, \
    ticket_type_LINE, ticket_type_PC, ticket_type_SC, ticket_type_SCH, \
    ticket_type_SO_PP, ticket_type_SOC, ticket_type_SOTON, ticket_type_W_C, \
    cabin_type_A, cabin_type_B, cabin_type_C, cabin_type_D, cabin_type_E, \
    cabin_type_F, cabin_type_G, passengerid_bins_encoded, age_bins_encoded     
    ]

    preprocessed_array = np.array(preprocessed_list)
    preprocessed_df = pd.DataFrame(preprocessed_array.reshape(1,-1))
    
    # rearrange 
    return preprocessed_df


def inference(model: LogisticRegression, X_data: pd.DataFrame):
    """
    Predict target label using X_data as input.
    X_data should be the preprocessed data from preprocess_data.

    Inputs 
    ------ 
    model : LogisticRegression 
        Trained model. 
    X_data : pd.DataFrame 
        X_input as a df.
    
    Returns 
    ------
    preds : list 
        List of predictions, 1/0 
    """

    preds = model.predict(X_data).tolist()

    survive_dict = {
        1: 'did not survive',
        0: 'survived'
    }

    preds_mapped = [survive_dict.get(i,i) for i in preds]

    return preds_mapped
