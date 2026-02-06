import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor



class grpClasificador(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", freeze_clip=True, dropout_rate=0.35, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(grpClasificador, self).__init__()

        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False

        clip_output_dim = self.clip_encoder.config.hidden_size

        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(clip_output_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        weights_path = "./modelo_final.pth"

        checkpoint = torch.load(weights_path, weights_only=False,map_location=map_location)
        state_dict = checkpoint['model_state_dict']
        self.load_state_dict(state_dict)


    def forward(self, images):

        pixel_values = self.processor(images=images, return_tensors="pt", do_rescale=False)['pixel_values']
        pixel_values = pixel_values.to(images.device if torch.is_tensor(images) else 'cuda')

        with torch.no_grad() if not self.training else torch.enable_grad():
            clip_outputs = self.clip_encoder(pixel_values=pixel_values)
            features = clip_outputs.pooler_output

        features = self.dropout(features)
        logits = self.linear(features)

        output = 2 * torch.sigmoid(logits) - 1

        return output
