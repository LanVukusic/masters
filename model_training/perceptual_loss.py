import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as autograd_grad


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_sizes=(512, 1024, 2048),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.num_scales = len(fft_sizes)

    def _spectrogram(self, x, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x.squeeze(1).float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        mag = spec.abs()
        return torch.log(mag + 1e-7)

    def forward(self, x_fake, x_real):
        loss = 0.0
        for i in range(self.num_scales):
            s_fake = self._spectrogram(
                x_fake, self.fft_sizes[i], self.hop_sizes[i], self.win_sizes[i]
            )
            s_real = self._spectrogram(
                x_real, self.fft_sizes[i], self.hop_sizes[i], self.win_sizes[i]
            )
            loss = loss + F.l1_loss(s_fake, s_real)
        return loss / self.num_scales


class DifferentiableDecoder(nn.Module):
    def __init__(self, encodec_model):
        super().__init__()
        self.quantizer = encodec_model.quantizer
        self.decoder = encodec_model.decoder
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, logits, tau=1.0):
        B, T, K, V = logits.shape
        total_emb = None
        for k in range(K):
            soft_oh = F.gumbel_softmax(logits[:, :, k, :], tau=tau, hard=False, dim=-1)
            codebook = self.quantizer.vq.layers[k].codebook
            emb = torch.matmul(soft_oh, codebook)
            total_emb = emb if total_emb is None else total_emb + emb

        total_emb = total_emb.transpose(1, 2).contiguous()
        audio = self.decoder(total_emb)
        return audio


class PerceptualLoss(nn.Module):
    def __init__(self, encodec_model):
        super().__init__()
        self.diff_decoder = DifferentiableDecoder(encodec_model)
        self.stft_loss = MultiResolutionSTFTLoss()
        self.register_buffer("_scale", torch.tensor(1.0))
        self._scale_initialized = False

    def _init_scale(self, fake_waveform, real_waveform):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            val = self.stft_loss(fake_waveform.float(), real_waveform.float()).detach()
        self._scale = val.clamp(min=1e-8)
        self._scale_initialized = True

    def forward(self, logits, real_waveform, tau=1.0, loss_ce=None):
        fake_waveform = self.diff_decoder(logits, tau=tau)
        loss = self.stft_loss(fake_waveform, real_waveform)

        if not self._scale_initialized:
            self._init_scale(fake_waveform, real_waveform)
            return loss / self._scale

        # Gradient-norm adaptive scaling: match perceptual gradient magnitude
        # to the cross-entropy gradient magnitude at the logits.
        if loss_ce is not None:
            # retain_graph=True so the graph survives for the main backward()
            g_ce = autograd_grad(loss_ce, logits, retain_graph=True)[0]
            g_percep = autograd_grad(loss, logits, retain_graph=True)[0]
            n_ce = g_ce.norm().detach()
            n_percep = g_percep.norm().detach()
            scale = torch.clamp(n_percep / (n_ce + 1e-8), min=0.01, max=100.0)
            return loss / scale

        return loss / self._scale
