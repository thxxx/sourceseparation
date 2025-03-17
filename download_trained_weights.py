import torch
from stable_audio_tools import get_pretrained_model
import os

device = "cuda" if torch.cuda.is_available() else "cpu" 

if __name__ == "__main__":
    print("Download Pre trained Stable audio weights")
    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"] 
    sample_size = model_config["sample_size"] # 47 * 44.1kHz
    # model.load_state_dict(torch.load('/workspace/mel_con_sample/model_epoch_350.pth')['model_state_dict'])
    
    if not os.path.exists("./pretrained2/"):
        os.mkdir("./pretrained2/")
    
    # torch.save(model.pretransform.state_dict(), './pretrained2/vae_weight.pth')
    
    torch.save(model.model.model.transformer.state_dict(), './pretrained2/transformer_weight.pth')
    
    torch.save(model.model.model.to_cond_embed.state_dict(), './pretrained2/to_cond_embed.pth')
    torch.save(model.model.model.to_global_embed.state_dict(), './pretrained2/to_global_embed.pth')
    torch.save(model.model.model.to_timestep_embed.state_dict(), './pretrained2/to_timestep_embed.pth')
    torch.save(model.model.model.preprocess_conv.state_dict(), './pretrained2/preprocess_conv.pth')
    torch.save(model.model.model.postprocess_conv.state_dict(), './pretrained2/postprocess_conv.pth')
    torch.save(model.model.model.timestep_features.state_dict(), './pretrained2/timestep_features.pth')
    
    torch.save(model.conditioner.conditioners.seconds_start.state_dict(), './pretrained2/sec_start.pth')
    torch.save(model.conditioner.conditioners.seconds_total.state_dict(), './pretrained2/sec_total.pth')
    
    