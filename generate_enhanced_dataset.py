import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

GENRES_DIR = '../genres_original'
TRAIN_TXT='../julianofoleiss/f1_train.txt'
TEST_TXT='../julianofoleiss/f1_test.txt'

def compute_features(y,sr):
    features={}

    #basic features (mean and variance)
    chroma=librosa.feature.chroma_stft(y=y,sr=sr)
    features['chroma_stft_mean']=np.mean(chroma)
    features['chroma_stft_var']=np.var(chroma)

    rms=librosa.feature.rms(y=y) #energy
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    spec_cent=librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = np.mean(spec_cent)
    features['spec_cent_var'] = np.var(spec_cent)

    spec_bw=librosa.feature.spectral_bandwidth(y=y,sr=sr)
    features['spec_bw_mean'] = np.mean(spec_bw)
    features['spec_bw_var'] = np.var(spec_bw)

    rolloff=librosa.feature.spectral_rolloff(y=y,sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    zcr=librosa.feature.zero_crossing_rate(y=y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_var'] = np.var(zcr)

    y_harm, y_perc = librosa.effects.hpss(y=y)
    rms_harm=librosa.feature.rms(y=y_harm)
    features['harmony_mean']=np.mean(rms_harm) #How much the harmonic energy fluctuates.
    features['harmony_var']=np.var(rms_harm)

    rms_perc=librosa.feature.rms(y=y_perc)
    features['perceptr_mean']=np.mean(rms_perc) #Average energy of the drum / transient part of the clip.
    features['perceptr_var']=np.var(rms_perc)
    #advanced features
    #spectral contrast (distinguishes peak-y and noise-like sound)
    contrast=librosa.feature.spectral_contrast(y=y,sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_mean_{i}']=np.mean(contrast[i])
        features[f'spectral_contrast_var_{i}']=np.var(contrast[i])
    
    #tonnetz (captures harmonic/chord progression - great for jazz and pop)
    try:
        #tonnetz requires harmonic component
        y_harmonic=librosa.effects.harmonic(y)
        tonnetz=librosa.feature.tonnetz(y=y_harmonic,sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_mean_{i}']=np.mean(tonnetz[i])
            features[f'tonnetz_var_{i}']=np.var(tonnetz[i])
    except:
        #fallback if silence/error
        for i in range(6):features[f'tonnetz_{i}']=0.0

    #mfcc + delta (captures timbre + rhythm/change)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    for i in range(20):
        features[f'mfcc_mean_{i}']=np.mean(mfcc[i])
        features[f'mfcc_var_{i}']=np.var(mfcc[i])

    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    for i in range(20):
        features[f'delta_mean_{i}']=np.mean(mfcc_delta[i])
        features[f'delta_var_{i}']=np.var(mfcc_delta[i])

    return features

def process_split(txt_path, output):
    print(f"\nProcessing List: {txt_path}")
    valid_songs=[]
    with open(txt_path,'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                filename=parts[0].replace('./','')
                genre=filename.split('_')[0]

                real_filename=filename.replace('_','.')
                full_path=os.path.join(GENRES_DIR,genre,real_filename)

                valid_songs.append((full_path,genre))

    #extract features
    data=[]
    for file_path,genre in tqdm(valid_songs):
        if not os.path.exists(file_path):
            continue
        
        try:
            y_full, sr = librosa.load(file_path,duration=30)
            #split into 10 chunks of 3 seconds
            chunk_samples=3*sr
            num_chunks=int(len(y_full)/chunk_samples)

            for i in range(num_chunks):
                start=i*chunk_samples
                end=start+chunk_samples
                y_chunk=y_full[start:end]

                #compute features for this chunk
                row=compute_features(y_chunk,sr)
                row['label']=genre
                row['filename']=os.path.basename(file_path)
                data.append(row)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")

if __name__ == "__main__":
    # Generate Training Data (Clean Sturm Split)
    process_split(TRAIN_TXT, 'train_enhanced.csv')
    
    # Generate Test Data (Clean Sturm Split)
    process_split(TEST_TXT, 'test_enhanced.csv')