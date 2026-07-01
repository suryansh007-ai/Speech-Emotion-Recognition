import librosa
import numpy as np
def extract_features(audio_path):
    audio,sr=librosa.load(audio_path,sr=None)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
    mfcc_mean=np.mean(mfcc,axis=1)
    mfcc_std = np.std(mfcc,axis=1)

    chroma=librosa.feature.chroma_stft(
         y=audio,
         sr=sr)
    chroma_mean=np.mean(chroma,axis=1)
    chroma_std = np.std(chroma, axis=1)

    rms=librosa.feature.rms(y=audio)
    rms_mean=np.mean(rms,axis=1)
    rms_std = np.std(rms,axis=1)

    zcr=librosa.feature.zero_crossing_rate(audio)
    zcr_mean=np.mean(zcr,axis=1)
    zcr_std = np.std(zcr,axis=1)

    centroid=librosa.feature.spectral_centroid(
         y=audio,
         sr=sr
    )
    centroid_mean=np.mean(centroid,axis=1)
    centroid_std = np.std(centroid,axis=1)

    contrast=librosa.feature.spectral_contrast(
         y=audio,
         sr=sr)
    contrast_mean=np.mean(contrast,axis=1)
    contrast_std = np.std(contrast,axis=1)

    rolloff=librosa.feature.spectral_rolloff(
         y=audio,
         sr=sr
    )
    rolloff_mean=np.mean(rolloff,axis=1)
    rolloff_std=np.std(rolloff,axis=1)

    features=np.hstack((
        mfcc_mean,
        mfcc_std,

        chroma_mean,
        chroma_std,

        rms_mean,
        rms_std,

        zcr_mean,
        zcr_std,

        centroid_mean,
        centroid_std,

        contrast_mean,
        contrast_std,

        rolloff_mean,
        rolloff_std
    ))
    return features