import pandas as pd
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import torch
from tqdm import tqdm
import os
from PIL import Image


class Embedder:
    def __init__(self, data, tokenizer, model, embedding_type):
        self.embeddings = None
        self.embedding_type = embedding_type
        self.data = data
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def create_embeddings(self):
        pass  
    
    def get_embeddings(self):
        return self.embeddings 
      
    def load_embeddings(self, output_file):
        return torch.load(output_file)  if os.path.exists(output_file) else None
      
    

class BertEmbedder(Embedder):
    def __init__(self, data: pd.DataFrame, 
                 tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'), 
                 model = BertModel.from_pretrained('bert-base-uncased'), 
                 embedding_type = "text"):
        super().__init__(self, embedding_type, data, tokenizer, model)
    
    def create_embeddings(self, texts, batch_size=64, output_file='..\\..\\data\\processed\\bert_text_embeddings.pt'):
        preloaded = self.load_embeddings()
        if preloaded is not None:
            return preloaded
        
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
                 embedding_type = "image"):
        super().__init__(self,embedding_type, data, tokenizer, model)
        
    def create_embeddings(self,image_paths, batch_size=32, output_file="..\\..\\data\\processed\\vit_image_embeddings.pt"):
        preloaded = self.load_embeddings()
        if preloaded is not None:
            return preloaded
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        base_dir = "..\\..\\data\\raw\\train"
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
    
    

        