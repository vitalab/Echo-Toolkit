import json
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from scipy import ndimage
from tqdm import tqdm

from echotk.sector_tools.ransac_sector_validation import ransac_sector_w_metrics
from echotk.utils.file_utils import open_nifti_file, save_nifti_file
from echotk.utils.viz_utils import show_gif


def predict_with_torchscript(model: torch.jit.ScriptModule, vol: np.ndarray, device: str) -> np.ndarray:
    """Run a single volume through the TorchScript model and return a numpy segmentation mask.

    Args:
        model: Loaded TorchScript model (SectorEnetInferenceWrapper).
        vol: (H, W) or (H, W, T) numpy array.
        device: e.g. "cuda" or "cpu".

    Returns:
        Segmentation mask as a numpy array with the same spatial dims as vol.
    """
    x = torch.tensor(vol, dtype=torch.float32, device=device)

    # Model expects (B, H, W, T) — add batch dim if needed
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(-1)   # (1, H, W, 1)
    elif x.dim() == 3:
        x = x.unsqueeze(0)                 # (1, H, W, T)

    with torch.no_grad():
        out = model(x).squeeze(0).squeeze(0)          # (H, W)

    return out.cpu().numpy()


def extract_sector(cfg: DictConfig):
    out_path = Path(cfg.output)
    out_path.mkdir(exist_ok=True, parents=True)

    device = cfg.get("accelerator", "cpu")
    model = torch.jit.load(cfg.model, map_location=device)
    model.eval()

    if isinstance(cfg.input, ListConfig):
        filenames = list(cfg.input)
        vol = [open_nifti_file(p) for p in filenames]
    elif Path(cfg.input).is_dir():
        filenames = [p for p in Path(cfg.input).glob("*.nii.gz")]
        vol = [open_nifti_file(p) for p in filenames]
    elif Path(cfg.input).is_file():
        filenames = [Path(cfg.input)]
        vol = [open_nifti_file(cfg.input)]
    else:
        raise Exception("Invalid input file")

    pred = [predict_with_torchscript(model, v[0], device) for v in vol]

    # zip with filenames
    for p, v, f in tqdm(zip(pred, vol, filenames), total=len(pred)):
        if len(v[0].shape) < 3:
            img_3d = v[0].copy()[..., None]
        else:
            img_3d = v[0].copy()

        # compute final mask with ransac and return metrics used for validity
        ransac_mask, diff, ratio, annot, sig, ransac_param_dict = ransac_sector_w_metrics(p.astype(np.uint8),
                                                                                          img=img_3d,
                                                                                          plot=cfg.show_intermediate_plots)

        # Check if ransac mask passes metrics
        # combining metrics with diff means that we trust the nnUnet segmentation to be very good
        # It is possible that saturated images or other differences may trigger invalid results,
        # use these metrics accordingly
        passed = True
        if diff > cfg.ransac_thresh.diff and sig > cfg.ransac_thresh.signal_lost:
            if cfg.verbose:
                print(f"Difference between masks {diff}, signal lost {sig}")
            passed = False
        if ratio < cfg.ransac_thresh.ratio:
            if cfg.verbose:
                print(f"Mask ratio is too small {ratio}")
            passed = False
        if diff > cfg.ransac_thresh.diff and annot > cfg.ransac_thresh.remaining_annotations:
            if cfg.verbose:
                print(f"Annotations remain {annot}")
            passed = False

        # log metrics to dataframe
        metrics = {
            'ransac_params': ransac_param_dict,
            'valid': passed,
            'diff': diff,
            'signal_lost': sig,
            'mask_cov_ratio': ratio,
            'annotations_remain': annot,
        }

        # if 2D image, mask has been extented on last dimension
        if len(v[0].shape) < 3:
            ransac_mask = ransac_mask[..., 0]  # mask is same on all frames

        # appy mask and normalisation
        masked_image = v[0].copy()
        masked_image[~ransac_mask] = 0

        f_name = Path(f).stem.split('.')[0]

        if cfg.show_result_gifs:
            p = np.repeat(p[..., None], img_3d.shape[2], axis=-1)
            p_gif = show_gif(p.transpose((2, 1, 0)), f'{f_name}: Initial nn-UNet prediction')
            if len(v[0].shape) < 3:
                plt.figure()
                plt.imshow(masked_image.transpose((1, 0)))
                plt.title(f'{f_name}: Masked output image')

                plt.figure()
                plt.imshow(v[0].transpose((1, 0)))
                plt.title(f'{f_name}: Original input image')
            else:
                im_gif = show_gif(img_3d.transpose((2, 1, 0)), f'{f_name}: Original input image')
                m_gif = show_gif(masked_image.transpose((2, 1, 0)), f'{f_name}: Masked output image')
            plt.show()

        # save mask and metrics dict (if wanted)
        save_nifti_file(f"{out_path}/{f_name}_sector_removed.nii.gz", masked_image, v[1], v[2])
        if cfg.save_sector_mask:
            save_nifti_file(f"{out_path}/{f_name}_sector_mask.nii.gz", ransac_mask, v[1], v[2])
        if cfg.save_metrics:
            with open(f"{out_path}/{f_name}_metrics.json", "w") as outfile:
                json.dump(metrics, outfile)


@hydra.main(version_base="1.2", config_path="config", config_name="sector_extract_light.yaml")
def main(cfg: DictConfig):
    # set project root
    if 'PROJECT_ROOT' not in os.environ:
        project_root = os.path.join("/", *os.path.abspath(__file__).split('/')[:-2])
        print(f"Setting env variable PROJECT_ROOT to : {project_root}")
        os.environ['PROJECT_ROOT'] = project_root
    # run
    extract_sector(cfg)


if __name__ == '__main__':
    main()