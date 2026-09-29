import os
import mne
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
dataset = 1
rootpath = f'./dataset {dataset}/'

categories = ['Control', 'Schizophrenia']

# Bipolar pairs
bipolar_pairs = [
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('T3', 'C3'), ('C3', 'Cz'), ('Cz', 'C4'), ('C4', 'T4')
]

# ----------------------------
# MAIN LOOP
# ----------------------------
for c in categories:
    datapath = os.path.join(rootpath, c)  # now reading .fif
    savepath = os.path.join("./dataset1_for_crossvalidation", c)

    os.makedirs(savepath, exist_ok=True)

    file_list = [f for f in os.listdir(datapath) if f.endswith(".fif")]

    for file_name in file_list:
        full_path = os.path.join(datapath, file_name)
        print(f"Processing: {full_path}")

        # ----------------------------
        # Step 1: Load FIF (ASR already done)
        # ----------------------------
        raw = mne.io.read_raw_fif(full_path, preload=True)

        ch_names = raw.ch_names
        data = raw.get_data()

        new_data = []
        new_labels = []

        # ----------------------------
        # Step 2: Bipolar conversion
        # ----------------------------
        for ch1, ch2 in bipolar_pairs:
            if ch1 in ch_names and ch2 in ch_names:
                idx1 = ch_names.index(ch1)
                idx2 = ch_names.index(ch2)

                bipolar_signal = data[idx1] - data[idx2]
                new_data.append(bipolar_signal)
                new_labels.append(f"{ch1}-{ch2}")
            else:
                print(f"⚠️ Missing channel: {ch1} or {ch2}")

        new_data = np.array(new_data)

        # Create new Raw object
        info = mne.create_info(
            ch_names=new_labels,
            sfreq=raw.info['sfreq'],
            ch_types='eeg'
        )

        raw = mne.io.RawArray(new_data, info)

        # ----------------------------
        # Step 3: Bandpass filter
        # ----------------------------
        raw.filter(0.5, 45, fir_design='firwin')

        # ----------------------------
        # Step 4: Downsample
        # ----------------------------
        raw.resample(125)

        # ----------------------------
        # Step 5: Save processed data
        # ----------------------------
        save_name = file_name.replace('.fif', '.fif')
        save_file = os.path.join(savepath, save_name)

        mne.export.export_raw(
        save_file.replace('.fif', '.set'),
        raw,
        fmt='eeglab'
        )

print("✅ Cross preprocessing completed.")