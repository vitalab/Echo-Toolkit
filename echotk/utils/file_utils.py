import nibabel as nib


def open_nifti_file(path):
    img = nib.load(path)
    return img.get_fdata(), img.header, img.affine


def save_nifti_file(path, img, hdr, aff):
    nifti_img = nib.Nifti1Image(img, aff, hdr)
    nifti_img.to_filename(path)
