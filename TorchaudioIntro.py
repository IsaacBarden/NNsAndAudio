#An exercise in loading and processing audio using torchaudio
#https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

#NOTE: torchaudio backend (PySoundFile) is not available as a conda package,
# so be sure to use a non-conda environment or a conda environment with 
# that specific pip package installed.

#%%
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.datasets as dset
import requests
import matplotlib.pyplot as plt

# %%

#Opening file, graphing waveform

url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = requests.get(url)

with open("steam-train-whistle-daniel_simon-converted-from-mp3.wav", "wb") as f:
    f.write(r.content)

filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print(f"Shape of waveform: {waveform.size()}")
print(f"Sample rate of waveform: {sample_rate}")

fig, axis = plt.subplots(2)
axis[0].plot(waveform[0].t().numpy())
axis[0].set_ylabel("R")
axis[1].plot(waveform[1].t().numpy())
axis[1].set_ylabel("L")
# %%

#Analyzing audio as a spectrogram

specgram = torchaudio.transforms.Spectrogram()(waveform)

print(f"Shape of spectrogram: {specgram.size()}")

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap="gray")
# %%

#Analyzing audio as a Mel spectrogram

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print(f"Shape of spectrogram: {specgram.size()}")

plt.figure()
plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap="gray")
# %%

#Resampling

new_sample_rate = 1600

#Resample applies to a single channel, so resample first channel here
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print(f"Shape of transformed waveform: {transformed.size()}")

plt.figure()
plt.plot(transformed[0,:].numpy(
# %%

#Mu-Law encoding

#Make sure that the waveform stays within -1 and 1
print(f"Min: {waveform.min()}\nMax: {waveform.max()}")

#If it was not, it would need to be normalized to be between -1 and 1 

transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print(f"Shape of transformed waveform: {transformed.size()}")

plt.figure()
plt.plot(transformed[0,:].numpy())

#Mu-Law decoding

reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print(f"Shape of recovered waveform: {reconstructed.size()}")

plt.figure()
plt.plot(reconstructed[0,:].numpy())

err = ((waveform-reconstructed).abs() / waveform.abs()).median()
print(f"Median relative difference between original and MuLaw reconstructed signals: {err:.2%}")
# %%

#Using torchaudio.functional

mu_law_encoding_waveform = F.mu_law_encoding(waveform, quantization_channels=256)

print(f"Shape of transformed waveform: {mu_law_encoding_waveform.size()}")

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())
# %%

#Functional spectrogram deltas

specgram = torchaudio.transforms.Spectrogram()(waveform)
computed = torchaudio.functional.compute_deltas(specgram.contiguous(), win_length=3)
print(f"Shape of computed deltas: {computed.shape}")

plt.figure()

#The image is blank?? Probably matplotlib issue
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')
# %%

#Applying gain and dithering

gain_waveform = F.gain(waveform, gain_db=5.0)
print(f'''
Min of gain_waveform: {gain_waveform.min()}
Max of gain_waveform: {gain_waveform.max()}
Mean of gain_waveform: {gain_waveform.mean()}
''')

dither_waveform = F.dither(waveform)
print(f'''
Min of dither_waveform: {dither_waveform.min()}
Max of dither_waveform: {dither_waveform.max()}
Mean of dither_waveform: {dither_waveform.mean()}
''')
# %%

#Applying filters

lowpass_waveform = F.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print(f'''
Min of lowpass_waveform: {lowpass_waveform.min()}
Max of lowpass_waveform: {lowpass_waveform.max()}
Mean of lowpass_waveform: {lowpass_waveform.mean()}
''')

fig, axis = plt.subplots(2)
axis[0].plot(lowpass_waveform[0].t().numpy())
axis[0].set_ylabel("R")
axis[1].plot(lowpass_waveform[1].t().numpy())
axis[1].set_ylabel("L")

highpass_waveform = F.lowpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print(f'''
Min of highpass_waveform: {highpass_waveform.min()}
Max of highpass_waveform: {highpass_waveform.max()}
Mean of highpass_waveform: {highpass_waveform.mean()}
''')

fig, axis = plt.subplots(2)
axis[0].plot(highpass_waveform[0].t().numpy())
axis[0].set_ylabel("R")
axis[1].plot(highpass_waveform[1].t().numpy())
axis[1].set_ylabel("L")


#NOTE: Skipping kaldi migration, probably won't be necessary
# %%

#Importing datasets

yesno_data = torchaudio.datasets.YESNO('./', download = True)
# A data point in Yesno is a tuple (waveform, sample_rate, labels) 
# where labels is a list of integers with 1 for yes and 0 for no.

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print(f'''
Waveform: {waveform}
Sample rate: {sample_rate}
Labels: {labels}
''')

plt.figure()
plt.plot(waveform.t().numpy())
