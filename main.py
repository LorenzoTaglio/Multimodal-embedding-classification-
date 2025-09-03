from src import models, utils

def main():
    TXT_EMB_DIR = "data\\processed\\bert_text_embeddings.pt"
    IMG_EMB_DIR = "data\\processed\\vit_image_embeddings.pt"
    # load data
    df = utils.get_dataframe()
    
    # load embedding models
    text_embedder = models.BertEmbedder(data=df["Caption"])
    image_embedder = models.VitEmbedder(data=df["Image"])
    
    # load image and text embeddings
    text_embedder.create_embeddings(output_file=TXT_EMB_DIR)
    image_embedder.create_embeddings(output_file=IMG_EMB_DIR)
    
    txt_embeddings = text_embedder.get_embeddings()
    img_embeddings = image_embedder.get_embeddings()
    
    # -----------------------------------early-fusion-----------------------------------
    print("Early fusion pipeline")
    early_fusion = models.EarlyFusionPipeline(txt_embeddings, img_embeddings, df["CUI"])
    ealry_fusion_results = early_fusion.early_fusion_stratified()
    
    ealry_fusion_results.to_dataframe().to_csv("out\\early_fusion.csv")
    print("Results saved to out\\early_fusion.csv")


    # -----------------------------------late-fusion-----------------------------------
    print("Late fusion pipeline")
    late_fusion = models.LateFusionPipeline(txt_embeddings, img_embeddings, df["CUI"])
    late_fusion_results = late_fusion.late_fusion_stratified()
    
    for key, results in late_fusion_results.items():
        results.to_dataframe().to_csv(f"out\\late_fusion_{key}.csv")
    
    print("Results saved to dir 'out'.")
    
if __name__ == "__main__":
    main()
