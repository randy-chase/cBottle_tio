# Changelog

## Latest
New features:
  - hurricane guidance (@manshap)
  - high level inference helpers in `cbottle.inference`. See examples/tc_guidance_inference.py for usage.

Enhancements
- performance optimization to unet (@akshaysubr). Approximately 3x faster.

Breaking changes:
- cbottle.models.networks.SongUNet now returns a dataclass. used w/ hurricane guidance and all coarse models. `cbottle.denoiser_factories` are modified to work with this new API. Super-resolution models still use the old API---the return a tensor.
