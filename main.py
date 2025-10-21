import argparse
from src import models, utils
from datetime import datetime
import os

def main(args):
    # load data
    df = utils.get_dataframe()

    # get date
    date = datetime.today().strftime('%Y-%m-%d')

    if args.text_model == "bert":
        TXT_EMB_DIR = "data\\processed\\bert_text_embeddings.pt"
        text_embedder = models.BertEmbedder(data=df["Caption"])

    print("Text model loaded")

    if args.image_model == "vit":
        IMG_EMB_DIR = "data\\processed\\vit_image_embeddings.pt"
        image_embedder = models.VitEmbedder(data=df["Image"])
    elif args.image_model == "medimageinsight":
        IMG_EMB_DIR = "data\\processed\\MedImageInsight_image_embeddings.pt"
        image_embedder = models.MedImageEmbedder(data=df["Image"])

    print("Image model loaded")


    # create dirs for output
    out_dir = f"out\\{text_embedder.name}_{image_embedder.name}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    print("Loading embeddings")
    # load image and text embeddings
    text_embedder.create_embeddings(output_file=TXT_EMB_DIR)
    image_embedder.create_embeddings(output_file=IMG_EMB_DIR)

    txt_embeddings = text_embedder.get_embeddings()
    img_embeddings = image_embedder.get_embeddings()

    # -----------------------------------early-fusion-----------------------------------
    if args.fusion in ["early", "both"]:
        early_fusion_dir = out_dir + f"\\{date}_early_fusion.csv"
        print("Early fusion pipeline")
        early_fusion = models.EarlyFusionPipeline(txt_embeddings, img_embeddings, df["CUI"])
        ealry_fusion_results = early_fusion.early_fusion_stratified()

        ealry_fusion_results.to_dataframe().to_csv(early_fusion_dir)
        print("Results saved to out\\early_fusion.csv")


    # -----------------------------------late-fusion-----------------------------------
    if args.fusion in ["late","late_mean", "late_meta","both"]:
        print("Late fusion pipeline")
        late_fusion = models.LateFusionPipeline(txt_embeddings, img_embeddings, df["CUI"])
        if args.fusion in ["late", "late_mean", "both"]:
            print("Late fusion with mean and weighted mean")
            late_fusion_results = late_fusion.late_fusion_stratified()

            for key, results in late_fusion_results.items():
                results.to_dataframe().to_csv(f"{out_dir}\\{date}_late_fusion_{key}.csv")

        if args.fusion in ["late", "late_meta", "both"]:
            print("Late fusion with meta classifier")
            late_fusion_meta_results = late_fusion.late_fusion_meta()
            late_fusion_meta_results.to_dataframe().to_csv(f"{out_dir}\\{date}_late_fusion_meta.csv")
            print("Results saved to dir 'out'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fusion pipelines with text and image embeddings")

    parser.add_argument(
        "-text_model", type=str, default="bert",
        help="Text embedding model. (options: bert) (default: bert)"
    )

    parser.add_argument(
        "-image_model", type=str, default="vit",
        help="Image embedding model. (options: vit, medimageinsight) (default: vit)"
    )

    parser.add_argument(
        "-fusion", type=str, default="both",
        help="Fusion types. (options: early,late,late_mean,late_meta,both) (default: both)"
    )

    args = parser.parse_args()
    main(args)


