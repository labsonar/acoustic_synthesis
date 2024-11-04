import soundfile as sf
from scipy.signal import resample

input_filename = './result/scenario.wav'
output_filename = './result/scenario_31250hz.wav'

data, fs_original = sf.read(input_filename)

if fs_original != 16000:
    raise ValueError("A taxa de amostragem do arquivo de entrada não é 16 kHz.")

fs_nova = 31250

num_samples = int(len(data) * fs_nova / fs_original)

data_reamostrado = resample(data, num_samples)

sf.write(output_filename, data_reamostrado, fs_nova)

print(f"Arquivo reamostrado salvo como {output_filename}.")
