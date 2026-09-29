import os
import mne
from asrpy import ASR

# ----------------------------
# CONFIG
# ----------------------------
dataset = 1
rootpath = f'./dataset {dataset}/'

categories = ['Control', 'Schizophrenia']

# ----------------------------
# MAIN LOOP
# ----------------------------
for c in categories:
    datapath = os.path.join(rootpath, f"{c}_bp")

    if dataset == 1:
        savepath = os.path.join(rootpath, c)
    else:
        savepath = os.path.join(rootpath, f"{c}_asr")

    os.makedirs(savepath, exist_ok=True)

    file_list = os.listdir(datapath)

    for file_name in file_list:
        if file_name.endswith(".edf"):
            full_path = os.path.join(datapath, file_name)
            print(f"Processing: {full_path}")

            # ----------------------------
            # Step 1: Load EDF
            # ----------------------------
            raw = mne.io.read_raw_edf(full_path, preload=True)

            # ----------------------------
            # Step 2: Apply ASR (FIXED)
            # ----------------------------
            asr = ASR(sfreq=raw.info['sfreq'], cutoff=16)

            # ✅ IMPORTANT: pass raw, not numpy
            asr.fit(raw)
            raw_clean = asr.transform(raw)

            # ----------------------------
            # Step 3: Save file
            # ----------------------------
            save_name = file_name.replace('.edf', '.fif')
            save_file = os.path.join(savepath, save_name)

            raw_clean.save(save_file, overwrite=True)

print("✅ ASR processing completed.")