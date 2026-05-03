import os
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("Error: 'nibabel' is not installed. Please install it using 'pip install nibabel'")
    exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: 'matplotlib' is not installed. Please install it using 'pip install matplotlib'")
    exit(1)

def preprocess(base_dir, hdr_path, cdr, count):

    # Load the image using nibabel
    # nibabel automatically finds the corresponding .img file if you point it to the .hdr file
    print(f"Loading MRI from: {hdr_path}...")
    try:
        img = nib.load(hdr_path)
    except Exception as e:
        print(f"Could not load MRI from {hdr_path}")
        print(f"Error: {e}")
        return count
    
    # Extract the 3D numpy array
    data = img.get_fdata()
    print(f"Image loaded successfully!")
    print(f"Image shape (X, Y, Z): {data.shape}")
    print(f"Voxel intensity range: Min = {data.min()}, Max = {data.max()}")

    z_mid = data.shape[2] // 2 # Axial

    z_start = z_mid - 5
    z_end = z_mid + 17

    for z in range(z_start, z_end+1):
        save_path = os.path.join(base_dir, "OASIS", cdr, f"{cdr}_{count}.jpg")
        one_slice = data[:, :, z, 0]
        # Rotate 90 degrees clockwise
        one_slice = np.rot90(one_slice, k=-1)
        plt.imsave(save_path, one_slice, cmap='gray')
        count += 1

    return count

if __name__ == "__main__":
    oasis_folder = 'disc1'


    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Base directory: {base_dir}")

    disc_dir = os.path.join(base_dir, oasis_folder)
    print(f"Disc directory: {disc_dir}")
    cdr_dict = {}
    
    for subject_folder in sorted(os.listdir(disc_dir)):
        subject_path = os.path.join(disc_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue
            
        txt_file_path = os.path.join(subject_path, f"{subject_folder}.txt")
        if not os.path.exists(txt_file_path):
            continue
            
        with open(txt_file_path, 'r') as f:
            cdr_val = None
            for line in f:
                if line.startswith('CDR:'):
                    val_str = line.split(':')[1].strip()
                    if val_str:
                        try:
                            cdr_val = float(val_str)
                        except ValueError:
                            pass
                    break
                    
        if cdr_val is not None:
            cdr_label_map = {
                0.0: 'Non_Demented',
                0.5: 'Very_Mild_Demented',
                1.0: 'Mild_Demented',
                2.0: 'Moderate_Demented'
            }
            label = cdr_label_map.get(cdr_val, str(cdr_val))
            
            if label not in cdr_dict:
                cdr_dict[label] = []
            
            # Extract just the ID number (e.g., '0004' from 'OAS1_0004_MR1')
            subj_id_num = subject_folder.split('_')[1] if '_' in subject_folder else subject_folder
            cdr_dict[label].append(subj_id_num)

    print("CDR Dictionary:")
    for cdr_label, subjects in cdr_dict.items():
        print(f"CDR {cdr_label}: {len(subjects)} subjects - {subjects}")

    for cdr in cdr_dict.keys():
        os.makedirs(os.path.join(base_dir, "OASIS", cdr), exist_ok=True)
        count = 0
        for subject in cdr_dict[cdr]:
            file_path = os.path.join(disc_dir, f"OAS1_{subject}_MR1", "PROCESSED", "MPRAGE", "T88_111", f"OAS1_{subject}_MR1_mpr_n4_anon_111_t88_masked_gfc.img")
            count = preprocess(base_dir, file_path, cdr, count)
