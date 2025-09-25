import pandas as pd
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import torch
from tqdm import tqdm
import os
from PIL import Image
from MedImageInsights.medimageinsightmodel import MedImageInsight
import base64
import numpy as np


class Embedder:
    """Parent class for every embedding model.
    
    ---
    Attributes:
    - data: a pandas dataframe column of the the data that you want embedd.
    - 
    
    
    """
    def __init__(self, data, tokenizer, model, embedding_type, name):
        self.embeddings = None
        self.embedding_type = embedding_type
        self.data = data.tolist()
        self.tokenizer = tokenizer
        self.model = model
        self.name = name
        

    def create_embeddings(self):
        pass  
    
    def get_embeddings(self):
        return self.embeddings 
      
    def load_embeddings(self, output_file):
        if os.path.exists(output_file):
            self.embeddings = torch.load(output_file)
            self.embeddings = self.embeddings.detach().cpu().numpy().astype("float32")
            return True
        return None
      
    

class BertEmbedder(Embedder):
    def __init__(self, data, 
                 tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'), 
                 model = BertModel.from_pretrained('bert-base-uncased'), 
                 embedding_type = "text",
                 name="BERT"
                 ):
        super().__init__(data, tokenizer, model, embedding_type, name)
        self.model.eval()
    
    def create_embeddings(self, batch_size=64, output_file='..\\..\\data\\processed\\bert_text_embeddings.pt'):
        texts = self.data
        preloaded = self.load_embeddings(output_file)
        if preloaded is not None:
            print(f"{self.embedding_type} embeddings already found")
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to the selected device
        self.model.to(device)
        
        
        num_texts = len(texts)
        embeddings = []

        # Initialize tqdm to track progress
        pbar = tqdm(total=num_texts, desc=f"Embedding texts", unit="texts")
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i+batch_size]
            # Tokenize batch of texts
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            # Process batch with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Calculate embeddings (mean pooling)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            # Accumulate batch embeddings
            embeddings.append(batch_embeddings)
            # Update progress bar
            pbar.update(len(batch_texts))
        # Close progress bar
        pbar.close()
        # Concatenate embeddings of all batches
        embeddings = torch.cat(embeddings, dim=0)
        # Save embeddings directly as a tensor
        torch.save(embeddings, output_file)
        print(f"Embeddings saved to {output_file}")
        
        self.embeddings = embeddings.detach().cpu().numpy().astype("float32")
    
    
    def load_embeddings(self, output_file):
        return super().load_embeddings(output_file)
        
    def get_embeddings(self):
        return super().get_embeddings()



class VitEmbedder(Embedder):
    def __init__(self, data, 
                 tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k"), 
                 model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k"), 
                 embedding_type = "image",
                 name="ViT"
                 ):
        super().__init__(data, tokenizer, model, embedding_type, name)
        self.model.eval()
        
    def create_embeddings(self, base_dir = "data\\raw\\train", batch_size=32, output_file="data\\processed\\vit_image_embeddings.pt"):
        image_paths = self.data
        preloaded = self.load_embeddings(output_file)
        if preloaded is not None:
            print(f"{self.embedding_type} embeddings already found")
            return
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        
        all_emb = []
        num_images = len(image_paths)
        
        pbar = tqdm(total=num_images, desc=f"Embedding images", unit="images")
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = [Image.open(os.path.join(base_dir, path)).convert("RGB") for path in batch_paths]

            inputs = self.tokenizer(images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_tokens = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
            all_emb.append(cls_tokens.cpu())
            pbar.update(len(batch_paths))

        pbar.close()
        # Combine embeddings in one tensor
        full_embeddings = torch.cat(all_emb, dim=0)  # shape: (n_images, hidden_size)

        # Save file
        torch.save(full_embeddings, output_file)

        self.embeddings = full_embeddings
    
    
    def get_embeddings(self):
        return super().get_embeddings()
    
    def load_embeddings(self, output_file):
        return super().load_embeddings(output_file)
    

class MedImageEmbedder(Embedder):
    def __init__(self, data,
                 processor = None,
                 model = MedImageInsight(
                            model_dir="MedImageInsights\\2024.09.27",
                            vision_model_name="medimageinsigt-v1.0.0.pt",
                            language_model_name="language_model.pth"),
                 embedding_type="image",
                 name="MedImageInsight"
                 ):        
        super().__init__(data, processor, model, embedding_type, name)
        

    def create_embeddings(self, base_dir="data\\raw\\train", batch_size=32,
                          output_file="data\\processed\\medimage_embeddings.pt"):

        image_paths = self.data
        
        preloaded = self.load_embeddings(output_file)
        if preloaded is not None:
            print(f"{self.embedding_type} embeddings already found")
            return
        
        # load model only if necessary, since it's time-consuming
        self.model.load_model()
        
        # Pre-allocate tensor for better memory management
        all_embeddings = []
        
        print("Starting encoding...")
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding batches"):
            batch_paths = image_paths[i:i+batch_size]

            # Load and encode batch
            batch_images = []
            for image_path in batch_paths:
                with open(os.path.join(base_dir, image_path), "rb") as f:
                    base64_img = base64.encodebytes(f.read()).decode("utf-8")
                    batch_images.append(base64_img)

            # Encode batch and immediately free GPU memory
            with torch.no_grad():
                batch_embeddings = self.model.encode(images=batch_images)
                all_embeddings.append(batch_embeddings["image_embeddings"])

            # Explicit cleanup if needed for very large batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all embeddings
        self.embeddings = torch.cat(all_embeddings, dim=0)

        # Save results
        torch.save(self.embeddings, output_file)
        print(f"Saved {len(self.embeddings)} embeddings to {output_file}")
        

    def get_embeddings(self):
        return super().get_embeddings()
    
    def load_embeddings(self, output_file):
        return super().load_embeddings(output_file)