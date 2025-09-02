import os
import pandas as pd
import itertools
import pickle
from collections import Counter
import joblib
from umls_api import API
from tqdm import tqdm


def get_cuis_semantic_types():
    df_cuis = pd.read_csv("data\\raw\\cui_mapping.csv")
    xss = [cuis.split(";") for cuis in df_cuis["CUI"]]
    cuis = [x for xs in xss for x in xs]
    
    
    def get_cui_semantic_types(cui):
        try:
            api = API(api_key=os.getenv("UMLS_API_KEY"))  # Initialize once
            resp = api.get_cui(cui=cui)
            cui_semantic_types = resp["result"]["semanticTypes"]
            names = [st["name"] for st in cui_semantic_types]
            return {cui: set(names)}
        except Exception as e:
            print(f"Error processing CUI {cui}: {e}")
            return {cui: set()}


    results = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(get_cui_semantic_types)(cui) 
        for cui in tqdm(cuis, desc="Processing CUIs")
    )

    cuis_semantic_types = dict()
    for d in results:
        key = [*d][0]
        cuis_semantic_types[key] = d[key]

    with open("data\\interim\\cuis_semantic_types.pkl", "wb") as f:
        pickle.dump(cuis_semantic_types, f)
        
    return cuis_semantic_types


def get_dataframe(split:str = "train") -> pd.DataFrame:
    if os.path.exists(f"data\\processed\\data.pkl"):
        print("Dataframe found")
        return pd.read_pickle(f"data\\processed\\data.pkl")
    
    print("Dataframe not found, creating new one")
    important_st = {"Disease or Syndrome", "Neoplastic Process", "Anatomical Abnormality"}
    
    selected_cuis = [
        "C0032285",  # Pneumonia
        "C0020295",  # Hydronephrosis
        "C5203670",  # COVID19 (disease)
        "C0006826",  # Malignant neoplastic disease
        "C0006267",  # Bronchiectasis
        "C0031039",  # Pericardial effusion
        "C0032326",  # Pneumothorax
        "C0003962",  # Ascites
        "C0497156",  # Lymphadenopathy
        "C0001304",  # Acute abscess
        "C0000833",  # Abscess
        "C0025062",  # Mediastinal emphysema
    ]


    if os.path.exists("data\\interim\\cuis_semantic_types.pkl"):
        cuis_semantic_types = pickle.load(open("data\\interim\\cuis_semantic_types.pkl", "rb"))
    else:
        cuis_semantic_types = get_cuis_semantic_types()
    
    base_dir = "data\\raw\\"
    df = pd.read_csv(os.path.join(base_dir, f"{split}_concepts.csv"))
    
    print(f"Initial shape: {df.shape}")
    # split cuis in a list
    df["CUIs"] = df["CUIs"].apply(lambda x: x.split(";"))
    
    # get cuis of interest (Disease or Syndrome, Neoplastic Process, Anatomical Abnormality)
    def filter_cuis(cuis):
        filtered = [
            cui 
            for cui in cuis 
            if cui in cuis_semantic_types
            for semantic in cuis_semantic_types[cui] 
            if semantic in important_st
        ]
        return filtered if filtered else None
    
    df["CUIs"] = df["CUIs"].apply(filter_cuis)
    
    # eliminate empty cuis
    df = df.dropna(subset=["CUIs"])
    print(f"Shape after filtering CUIs: {df.shape}")
    
    # get captions
    df_captions = pd.read_csv(os.path.join(base_dir, f"{split}_captions.csv")).set_index("ID")
    df = df.set_index("ID").join(df_captions, how="inner").reset_index()
    print(f"Shape after joining with captions: {df.shape}")
    
    # get images
    df_images = pd.DataFrame()
    image_files = os.listdir(os.path.join(base_dir, split))
    df_images["Image"] = image_files
    df_images["ID"] = df_images["Image"].apply(lambda x: x.split(".")[0])
    
    df = df.set_index("ID").join(df_images.set_index("ID"), how="inner").reset_index()
    print(f"Shape after joining with images: {df.shape}")
    
    # filter cuis by frequency
    cui_freq = Counter(itertools.chain.from_iterable(df["CUIs"]))
    cui_freq = {k: v for k, v in cui_freq.items() if v >= 100}
    
    df_cui_map = pd.read_csv("..\\..\\data\\raw\\cui_mapping.csv")

    df_cui_map = df_cui_map[df_cui_map["CUI"].apply(lambda cui: cui in cui_freq)]
    df_cui_map["Frequency"] = df_cui_map["CUI"].apply(lambda cui: cui_freq.get(cui, 0))

    df_cui_map = df_cui_map.sort_values(by="Frequency", ascending=False)

    def filter_cuis_final(cuis):
        filtered = [cui for cui in cuis if cui in cui_freq]
        return filtered if filtered else None
    
    print(f"Shape before filtering by frequency: {df.shape}")
    df["CUIs"] = df["CUIs"].apply(filter_cuis_final)
    df = df.dropna(subset=["CUIs"])
    
    print(f"Shape after filtering by frequency: {df.shape}")
    
    
    # use only manually selected cuis
    all_cuis = set(df["CUIs"].explode().tolist())
    
    all_cuis = {k:v for k,v in zip(all_cuis, range(1,len(all_cuis)+1))}

    cuis_vec = [
                [
                    all_cuis[cui] 
                    for cui in cuis
                ]
                for cuis in df["CUIs"] 
                ]
    
    df["CUIs_vec"] = cuis_vec
    
    df = df[df["CUIs"].map(len) == 1].reset_index(drop=True)
    
    df["CUI"] = df["CUIs"].apply(lambda x: x[0])
    df["CUI_vec"] = df["CUIs_vec"].apply(lambda x: x[0])
    

    print(f"Shape before filtering by selected CUIs: {df.shape}")
    df = df[df["CUI"].isin(selected_cuis)]
    print(f"Shape after filtering by selected CUIs: {df.shape}")    
    
    
    df.to_pickle(f"data\\processed\\data.pkl")
    
    return df

