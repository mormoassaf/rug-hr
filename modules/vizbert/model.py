import torch
from transformers import AutoTokenizer, BertForMaskedLM
from .parts import VizBERTEncoder



class VizBERTForMaskedLM(torch.nn.Module):

    def __init__(self, 
            max_sequence_length=256,
            spatial_size=(),
        ):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = VizBERTEncoder(
            embedding_size=768, 
            max_sequence_length=max_sequence_length, 
            spatial_size=spatial_size)


        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, image):
        sequence_embedding = self.encoder(image)
        outputs = self.bert(inputs_embeds=sequence_embedding)
        return outputs.logits
    
    def predict_tokens(self, image):
        logits = self.forward(image)
        predicted_tokens = torch.argmax(logits, dim=-1)
        predicted_tokens = predicted_tokens.detach().cpu().numpy() # (batch, sequence_length)
        predicted_tokens = [self.tokenizer.convert_ids_to_tokens(token) for token in predicted_tokens]
        return predicted_tokens
