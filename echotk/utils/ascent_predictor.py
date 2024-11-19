import hydra
from hydra.core.global_hydra import GlobalHydra
from monai.transforms import ToTensord
from omegaconf import OmegaConf
from hydra import compose, initialize
from lightning import LightningModule

import torch
from ASCENT.ascent.predict import AscentPredictor
from ASCENT.ascent.utils.file_and_folder_operations import load_pickle
from ASCENT.ascent.utils.transforms import Preprocessd, Convert2Dto3DIfNeeded
from monai.data import CacheDataset, DataLoader, ArrayDataset, MetaTensor
from monai import transforms
from lightning import Trainer
import os

ASCENT_CONFIGS = f"../../ASCENT/ascent/configs/"  # must be relative path


class CustomASCENTPredictor:
    def __init__(self, model_name, use_tta=False):
        GlobalHydra.instance().clear()
        initialize(version_base="1.2", config_path=ASCENT_CONFIGS, job_name="model")
        cfg = compose(config_name=f"model/{model_name}.yaml", overrides=["++trainer.max_epochs=1",
                                                                         f"model.tta={use_tta}",
                                                                         "model.save_predictions=False"])
        print(OmegaConf.to_yaml(cfg))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.trainer = Trainer(
            max_epochs=2,
            deterministic=False,
            limit_train_batches=20,
            limit_val_batches=10,
            limit_test_batches=2,
            gradient_clip_val=12,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False
        )

        self.model: LightningModule = hydra.utils.instantiate(cfg.model)

        self.dataset_properties = load_pickle(
            os.path.join(
                f"{os.environ['PROJECT_ROOT']}/data",
                model_name.replace("_3d", "").replace("_2d", ""),
                "preprocessed",
                "dataset_properties.pkl",
            )
        )

        self.data_transforms = AscentPredictor.get_predict_transforms(self.dataset_properties)

    def predict_from_paths(self, seq_paths, ckpt_path):
        datalist = []
        for seq in seq_paths:
            datalist.append({'image': seq})

        # add transform to convert to 3d if image is 2d, must be before preprocessing (at index 2)
        transf = [t for t in self.data_transforms.transforms]
        transf.insert(2, Convert2Dto3DIfNeeded(keys='image', num_channel=1, num_time=4))
        transf = transforms.compose.Compose(transf)

        dataset = CacheDataset(data=datalist, transform=transf, cache_rate=1.0)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=os.cpu_count() - 1,
            pin_memory=False,
            shuffle=False,
        )

        return self.trainer.predict(model=self.model,
                                    dataloaders=dataloader,
                                    ckpt_path=ckpt_path)

    def predict_from_numpy(self, numpy_arr, ckpt_path):
        preprocessed = Preprocessd(keys='image',
                                   target_spacing=self.dataset_properties['spacing_after_resampling'],
                                   intensity_properties=self.dataset_properties['spacing_after_resampling'],
                                   do_resample=self.dataset_properties['do_resample'],
                                   do_normalize=self.dataset_properties['do_normalize'],
                                   modalities=self.dataset_properties['modalities'])
        tforms = transforms.compose.Compose([Convert2Dto3DIfNeeded(keys='image', num_channel=1, num_time=4),
                                             preprocessed,
                                             ToTensord(keys="image", track_meta=True)])
        dataset = ArrayDataset(img=numpy_arr, img_transform=tforms)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=os.cpu_count() - 1,
            pin_memory=False,
            shuffle=False,
        )

        return self.trainer.predict(model=self.model,
                                    dataloaders=dataloader,
                                    ckpt_path=ckpt_path)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from pathlib import Path
    from echotk.utils.file_utils import open_nifti_file
    import numpy as np

    predictor = CustomASCENTPredictor(model_name='sector_3d')

    seq_paths = "./../../data/examples/"
    seq_paths = [p for p in Path(seq_paths).glob("*.nii.gz")]

    #
    # From sequence of paths
    #
    pred = predictor.predict_from_paths(seq_paths, ckpt_path='../../data/model_weights/sector_extract.ckpt')
    print(pred[0].shape)

    plt.figure()
    plt.imshow(pred[-1][:, :, 0].T)
    plt.show()

    #
    # From numpy
    #
    data, _, _ = open_nifti_file(seq_paths[0])
    print(data.shape)
    data = np.expand_dims(data, 0)
    meta = {"filename_or_obj": 'FILENAME',
            "pixdim": np.asarray([1, 0.43022227, 0.43022227, 1, 1, 1, 1, 1])}
    pred = predictor.predict_from_numpy([{'image': MetaTensor(torch.tensor(data, dtype=torch.float32), meta=meta)}],
                                        ckpt_path='../../data/model_weights/sector_extract.ckpt')
    plt.figure()
    plt.imshow(pred[0][:, :, 0].T)
    plt.show()
