import torch
import numpy as np


def fda_transfer(target: np.ndarray, reference: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    Fourier Domain Adaptation:
    Replace the low-frequency amplitude of `target` with that of `reference`,
    while keeping the phase of `target` intact.

    Args:
        target:    H x W x C float32 image in [0, 1]
        reference: H x W x C float32 image in [0, 1]
        beta:      Controls the size of the low-frequency window (0 < beta <= 0.5)

    Returns:
        FDA-adapted image, same shape as target, clipped to [0, 1]
    """
    assert target.shape == reference.shape, (
        f"Target and reference must have the same spatial size. "
        f"Got {target.shape} vs {reference.shape}"
    )

    H, W, C = target.shape

    # Low-frequency mask size
    h_half = int(H * beta)
    w_half = int(W * beta)

    result_channels = []
    for c in range(C):
        # FFT on each channel
        fft_target = np.fft.fft2(target[:, :, c])
        fft_ref    = np.fft.fft2(reference[:, :, c])

        # Shift zero-frequency to center
        fft_target_shift = np.fft.fftshift(fft_target)
        fft_ref_shift    = np.fft.fftshift(fft_ref)

        # Decompose into amplitude and phase
        amp_target = np.abs(fft_target_shift)
        pha_target = np.angle(fft_target_shift)
        amp_ref    = np.abs(fft_ref_shift)

        # Center coordinates
        cy, cx = H // 2, W // 2

        # Replace low-frequency amplitude of target with that of reference
        amp_adapted = amp_target.copy()
        amp_adapted[
            cy - h_half : cy + h_half,
            cx - w_half : cx + w_half
        ] = amp_ref[
            cy - h_half : cy + h_half,
            cx - w_half : cx + w_half
        ]

        # Reconstruct complex spectrum
        fft_adapted = amp_adapted * np.exp(1j * pha_target)

        # Inverse FFT
        fft_adapted_ishift = np.fft.ifftshift(fft_adapted)
        img_adapted = np.fft.ifft2(fft_adapted_ishift).real

        result_channels.append(img_adapted)

    adapted = np.stack(result_channels, axis=2)
    return np.clip(adapted, 0.0, 1.0).astype(np.float32)


class FourierDomainAdaptation:
    """
    ComfyUI node: Fourier Domain Adaptation (FDA)

    Transfers low-frequency color/style statistics from a reference image
    into a target image via amplitude spectrum swapping in the frequency domain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image":    ("IMAGE",),
                "reference_image": ("IMAGE",),
                "beta": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                    "display": "number",
                    "tooltip": (
                        "Low-frequency window size as a fraction of image dimensions. "
                        "Larger values transfer more global style; smaller values are subtler."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adapted_image",)
    FUNCTION = "apply_fda"
    CATEGORY = "image/transform"
    DESCRIPTION = (
        "Fourier Domain Adaptation: replaces the low-frequency amplitude spectrum "
        "of the target image with that of the reference image, transferring global "
        "color/style while preserving structural content."
    )

    def apply_fda(
        self,
        target_image: torch.Tensor,
        reference_image: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor]:
        """
        Args:
            target_image:    (B, H, W, C) float32 tensor, values in [0, 1]
            reference_image: (B, H, W, C) float32 tensor, values in [0, 1]
            beta:            Low-frequency window fraction

        Returns:
            Tuple of a single (B, H, W, C) float32 tensor.
        """
        B_t = target_image.shape[0]
        B_r = reference_image.shape[0]

        # Convert to numpy for FFT processing
        target_np    = target_image.cpu().numpy()     # (B, H, W, C)
        reference_np = reference_image.cpu().numpy()  # (B, H, W, C)

        H_t, W_t = target_np.shape[1], target_np.shape[2]
        H_r, W_r = reference_np.shape[1], reference_np.shape[2]

        results = []
        for i in range(B_t):
            # Cycle through reference frames if batch sizes differ
            ref_frame = reference_np[i % B_r]

            # Resize reference to match target spatial size if necessary
            if (H_r, W_r) != (H_t, W_t):
                ref_resized = _resize_np(ref_frame, H_t, W_t)
            else:
                ref_resized = ref_frame

            adapted = fda_transfer(target_np[i], ref_resized, beta=beta)
            results.append(adapted)

        output = np.stack(results, axis=0)  # (B, H, W, C)
        return (torch.from_numpy(output),)


def _resize_np(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear resize of a (h, w, c) float32 numpy array to (H, W, c)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, h, w)
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


NODE_CLASS_MAPPINGS = {
    "FourierDomainAdaptation": FourierDomainAdaptation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FourierDomainAdaptation": "Fourier Domain Adaptation (FDA)",
}
