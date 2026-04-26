import os
import librosa

DATASET = "data"   # your dataset folder

total_files = 0
durations = {}

for dialect in os.listdir(DATASET):
    path = os.path.join(DATASET, dialect)
    if not os.path.isdir(path):
        continue
    
    durations[dialect] = []
    
    for file in os.listdir(path):
        if file.lower().endswith(".wav"):
            total_files += 1
            file_path = os.path.join(path, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                durations[dialect].append(len(audio) / sr)
            except Exception as e:
                print("Error:", file_path, e)

print("\n---------- DATASET SUMMARY ----------")
print("Total files:", total_files)

total_hours = 0
for dialect, durs in durations.items():
    if len(durs) > 0:
        avg_dur = sum(durs) / len(durs)
        total_dur = sum(durs)
        total_hours += total_dur / 3600
        print(f"{dialect}: {len(durs)} files, avg duration = {avg_dur:.2f}s, total = {total_dur/3600:.3f} hours")

print("\nTotal dataset duration (hours):", total_hours)
