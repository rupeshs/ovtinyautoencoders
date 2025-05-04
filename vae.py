import torch


class VAEEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        return {"latent_sample": self.vae.encode(x=sample)["latents"]}


class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample)
