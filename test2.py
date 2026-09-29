import os
import mne
import numpy as np
import torch
from pathlib import Path
import sys
from captum.attr import Saliency

from models import *
from func import *

mne.set_log_level(verbose=0)

class Test:
    def __init__(self, args):
        self.rootpath = args.savepath
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        self.total_fold = args.fold

        self.channel = args.channel
        self.sfreq = args.sfreq
        self.window_length = args.window_length
        self.window_overlap = args.window_overlap 

        self.best_param = {}
        self.seed = args.seed

    def get_best_param(self):
        f = open(self.rootpath + 'pilot/pilot_result.txt')
        for line in f.readlines():
            params = line.split(',')
            self.best_param[params[0]] = {
                'training_batch': int(params[1]),
                'training_lr': float(params[2])
            }
        f.close()

    # ==========================================================
    # NEW: Load ONE EEG file from Kaggle dataset
    # Supports .txt / .csv / .eea / .npy / .edf / .set
    # ==========================================================
    def load_single_subject_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".set":
            raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
            epochs = mne.make_fixed_length_epochs(
                raw_data,
                duration=self.window_length,
                overlap=self.window_overlap
            )
            epochs_data = epochs.get_data()

        elif ext == ".edf":
            raw_data = mne.io.read_raw_edf(file_path, preload=True)
            epochs = mne.make_fixed_length_epochs(
                raw_data,
                duration=self.window_length,
                overlap=self.window_overlap
            )
            epochs_data = epochs.get_data()

        elif ext == ".npy":
            epochs_data = np.load(file_path)

        elif ext in [".txt", ".csv", ".eea"]:
            # Kaggle schizophrenia EEG dataset case
            data = np.loadtxt(file_path)

            expected = 16 * 7680
            if len(data.shape) > 1:
                data = data.flatten()

            if len(data) != expected:
                raise ValueError(
                    f"Expected {expected} values for 16 channels x 7680 samples, got {len(data)}"
                )

            data = data.reshape(16, 7680)   # (channels, samples)

            # convert to MNE Raw
            ch_names = [
                "F7", "F3", "F4", "F8",
                "T3", "C3", "Cz", "C4",
                "T4", "T5", "P3", "Pz",
                "P4", "T6", "O1", "O2"
            ]

            info = mne.create_info(
                ch_names=ch_names,
                sfreq=self.sfreq,
                ch_types="eeg"
            )

            raw_data = mne.io.RawArray(data, info)

            epochs = mne.make_fixed_length_epochs(
                raw_data,
                duration=self.window_length,
                overlap=self.window_overlap
            )
            epochs_data = epochs.get_data()

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return epochs_data

    # ==========================================================
    # NEW: Build test loader for ONE file only
    # ==========================================================
    def get_single_file_loader(self, file_path, label_name, model_name, dev):
        """
        label_name:
            'Control' or 'Schizophrenia'
        """

        class_id = 0 if label_name.lower() == "control" else 1
        class_prefix = "h" if class_id == 0 else "s"

        epochs_data = self.load_single_subject_data(file_path)

        # z-score normalization
        mean = epochs_data.mean()
        std = epochs_data.std()
        x = (epochs_data - mean) / (std + 1e-8)

        y = [class_id] * epochs_data.shape[0]

        full_epoch_data = [[class_prefix + "single", x, y]]

        test_loader = to_tensor(
            full_epoch_data,
            dev,
            self.best_param[model_name]['training_batch'],
            model_name,
            False
        )

        return test_loader, class_id

    def evaluate_and_get_gradient(self, model, data_loader, dev, model_name, class_id):
        predict_label = []
        label_list = []
        gradient_list = []

        saliency_inst = Saliency(model)
        if model_name != 'SzHNN':
            model.eval()
        else:
            model.train()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(data_loader):
                x_batch.requires_grad = False
                x_batch, y_batch = x_batch.to(dev, dtype=torch.float), y_batch.to(dev, dtype=torch.float)

                output = model(x_batch)

                label_list.append(y_batch.argmax(dim=1).detach().cpu().numpy())
                predict_label.extend(output.argmax(dim=1).tolist())

                x_batch.requires_grad = True
                gradient_list.append(
                    saliency_inst.attribute(
                        x_batch,
                        target=y_batch.argmax(dim=1).detach().cpu().numpy().tolist(),
                        abs=False,
                    ).detach().cpu().numpy()
                )

            label_list = np.concatenate(label_list)
            gradient_list = np.concatenate(gradient_list)

            if gradient_list.shape[1] == 1:
                gradient_list = np.squeeze(gradient_list, axis=1)

            arr = np.array([
                index for index, (x, y) in enumerate(zip(label_list, predict_label))
                if x == y and x == class_id
            ])

            if arr.shape[0] != 0:
                gradient = gradient_list[arr]
            else:
                gradient = np.array([])

        return predict_label, gradient

    def get_predict_result(self, test_loader, class_id, dev, model_name, best_model):
        predict_label, gradient = self.evaluate_and_get_gradient(
            best_model, test_loader, dev, model_name, class_id
        )

        correct_count = predict_label.count(class_id)
        predict_class = max(predict_label, key=predict_label.count)

        return correct_count, len(predict_label), predict_class, gradient

    # ==========================================================
    # NEW: Single file testing
    # ==========================================================
    def test_single_file(self, model_name, file_path, label_name, output_txt="single_test_result.txt"):
        print("Single file testing ----------------------")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.get_best_param()
        set_seed(self.seed)

        model_path = self.rootpath + 'full/' + model_name

        results = []

        # Use all fold models and do majority vote
        all_fold_predictions = []
        all_fold_segment_acc = []

        for fold in range(1, self.total_fold + 1):
            best_model = getattr(sys.modules[__name__], model_name)
            best_model = best_model(self.channel, int(self.sfreq * self.window_length), sfreq=self.sfreq)
            best_model.to(dev)

            best_model.load_state_dict(
                torch.load(model_path + '/fold' + str(fold % self.total_fold) + '_best_model.pth', map_location=dev)
            )
            best_model.eval()

            test_loader, class_id = self.get_single_file_loader(file_path, label_name, model_name, dev)

            correct_count, segment_count, predict_class, gradient = self.get_predict_result(
                test_loader, class_id, dev, model_name, best_model
            )

            segment_acc = correct_count / segment_count
            all_fold_segment_acc.append(segment_acc)
            all_fold_predictions.append(predict_class)

            results.append({
                "fold": fold,
                "correct_segments": correct_count,
                "total_segments": segment_count,
                "segment_accuracy": segment_acc,
                "predicted_class": predict_class
            })

        final_prediction = max(all_fold_predictions, key=all_fold_predictions.count)
        avg_segment_acc = sum(all_fold_segment_acc) / len(all_fold_segment_acc)

        predicted_label_name = "Control" if final_prediction == 0 else "Schizophrenia"
        true_label_id = 0 if label_name.lower() == "control" else 1
        is_correct = (final_prediction == true_label_id)

        # Save result to text
        with open(output_txt, "w") as f:
            f.write("===== SINGLE EEG TEST RESULT =====\n")
            f.write(f"Input file: {file_path}\n")
            f.write(f"True label: {label_name}\n")
            f.write(f"Predicted label: {predicted_label_name}\n")
            f.write(f"Correct prediction: {is_correct}\n")
            f.write(f"Average segment accuracy: {avg_segment_acc * 100:.2f}%\n\n")

            f.write("===== FOLD-WISE RESULTS =====\n")
            for r in results:
                pred_name = "Control" if r["predicted_class"] == 0 else "Schizophrenia"
                f.write(
                    f"Fold {r['fold']}: "
                    f"Segments Correct = {r['correct_segments']}/{r['total_segments']}, "
                    f"Segment Accuracy = {r['segment_accuracy'] * 100:.2f}%, "
                    f"Predicted = {pred_name}\n"
                )

        print(f"Result saved to: {output_txt}")

        return {
            "input_file": file_path,
            "true_label": label_name,
            "predicted_label": predicted_label_name,
            "correct": is_correct,
            "avg_segment_accuracy": avg_segment_acc,
            "fold_predictions": all_fold_predictions
        }