from pathlib import Path

import openvino as ov
import torch
from diffusers import AutoencoderTiny

from vae import VAEDecoder, VAEEncoder


class TAEOVConverter:
    """TAESDXL Converter for OpenVINO"""

    def __init__(
        self,
        model_id,
        output_path,
    ):
        self.model_id = model_id
        self.output_path = Path(output_path)
        self.tiny_vae = AutoencoderTiny.from_pretrained(self.model_id)
        self.tiny_vae.eval()

    def _convert_tiny_vae_encoder(self):
        print("⏳ Converting VAE encoder...")
        vae_encoder = VAEEncoder(self.tiny_vae)
        ov_model = ov.convert_model(
            vae_encoder,
            example_input=torch.zeros((1, 3, 512, 512)),
        )
        print("💾 Saving OpenVINO vae_encoder ...")
        ov.save_model(ov_model, self.output_path / "vae_encoder/openvino_model.xml")
        self.tiny_vae.save_config(self.output_path / "vae_encoder")
        print("✅ Encoder conversion complete!")

    def _convert_tiny_vae_decoder(self):
        print("⏳ Converting VAE decoder...")
        vae_decoder = VAEDecoder(self.tiny_vae)
        example_input = {
            "latent_sample": torch.zeros((1, 4, 64, 64)),
        }
        ov_model = ov.convert_model(
            vae_decoder,
            example_input=example_input,
        )
        print("💾 Saving OpenVINO vae_decoder ...")
        ov.save_model(ov_model, self.output_path / "vae_decoder/openvino_model.xml")
        self.tiny_vae.save_config(self.output_path / "vae_decoder")
        print("✅ Decoder conversion complete!")

    def convert(self):
        self._convert_tiny_vae_encoder()
        self._convert_tiny_vae_decoder()
