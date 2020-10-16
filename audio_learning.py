from pathlib import Path
import numpy as np
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import utils
import matplotlib.pyplot as plt
import moviepy.editor as mp
import python_speech_features as psf
import scipy.io.wavfile as wav
import svmutil as svm
import csv
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
import librosa

spect_list = []
category_list = []
spect_list_mfcc = []
category_list_mfcc = []
spect_list_mfcc_plus = []
category_list_mfcc_plus = []

def k_fold(sample_x, sample_y, ratio):
    size = int(len(sample_x) / ratio)
    
    rnd = np.arange(len(sample_x))
    np.random.shuffle(rnd)
    sample_x = sample_x[rnd]
    sample_y = sample_y[rnd]
    
    test_x = sample_x[:size]
    test_y = sample_y[:size]
    sample_x = sample_x[size + 1:]
    sample_y = sample_y[size + 1:]
    return test_x, test_y, sample_x, sample_y
 
    

def vectorize(mfcc_list):
    for mfcc in mfcc_list:
        for i in range(0, len(mfcc)):
            if(mfcc[i] <= 0):
                mfcc[i] = 0
            else:
                mfcc[i] = 1
    mfcc_list = np.asarray(mfcc_list)
    mfcc_list = mfcc_list.reshape(mfcc_list.shape[0], mfcc_list.shape[1])
    return mfcc_list

def build_problem(mfcc_list, categories):
    prob = svm.svm_problem(categories, mfcc_list)
    param = svm.svm_parameter('-t 0 -c 4 -b 1')
    m = svm.svm_train(prob, param)
    return m
    

def youtube_fetch():
    with open('eval_segments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        order_violent = 95
        order_non_violent = 100
        for row in csv_reader:
            if(line < 3):
                line = line + 1
            else:
                available = True
                link       = row[0]
                start      = float(row[1][1:])
                end        = float(row[2][1:])
                classifier = row[3]
                classifier = (classifier.split(','))[0][2:]
                if(classifier[-1] == '\"'):
                    classifier = classifier[:-1]
                if(classifier == '/m/03qc9zr' or 
                   classifier == '/m/032s66' or
                   classifier == '/m/04zjc' or
                   classifier == '/m/07rn7sz' or
                   classifier == '/m/03qc9zr' or
                   classifier == '/m/07pjjrj' or
                   classifier == '/m/07pc8lb' or 
                   classifier == '/m/014zdl' or
                   classifier == '/m/07plct2'):       
                    try:
                        print(link + " " + " " + str(start) + " " + str(end) + " " + classifier)
                        YouTube('https://youtu.be/' + link).streams.filter(subtype="mp4").first().download(output_path='Tests/Violent/', filename=str(order_violent + 1))
                    except VideoUnavailable:
                        print("Video is unavailable.")
                        available = False
                    except KeyError:
                        print("Signature stream error, going through the next.")
                        available = False
                    if(available == True):
                        order_violent = order_violent + 1
                        audio = mp.AudioFileClip('Tests/Violent/' + str(order_violent) + ".mp4")
                        audio = audio.subclip(int(start), int(end))
                        try:
                            audio.write_audiofile('Tests/Violent/' + str(order_violent) + ".wav")
                        except OSError:
                            print("File not fully written, going through the next.")
                        audio.close()
                else:
                    try:
                        print(link + " " + " " + str(start) + " " + str(end) + " " + classifier)
                        YouTube('https://youtu.be/' + link).streams.filter(subtype="mp4").first().download(output_path='Tests/Non-Violent/', filename=str(order_non_violent + 1))
                    except VideoUnavailable:
                        print("Video is unavailable.")
                        available = False
                    if(available == True):
                        order_non_violent = order_non_violent + 1
                        audio = mp.AudioFileClip('Tests/Non-Violent/' + str(order_non_violent) + ".mp4")
                        audio = audio.subclip(int(start), int(end))
                        try:
                           audio.write_audiofile('Tests/Non-Violent/' + str(order_non_violent) + ".wav")
                        except OSError:
                            print("File not fully written, going through the next.")
                        audio.close()         
                        
def fold_network(sample_x, sample_y, network_function):
    for i in range(2, 11):
        test_x, test_y, sample_new_x, sample_new_y = k_fold(sample_x, sample_y, i)
        network_function(sample_new_x, sample_new_y, test_x, test_y)

def create_network(spect_list, category_list, spect_val, category_val):
    net = models.Sequential()
    net.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape=(257, 100, 1), padding='same'))
    net.add(MaxPooling2D(pool_size = (2, 2)))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size = (2, 2)))
    net.add(Dropout(0.25))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size = (2, 2)))
    net.add(Flatten())
    net.add(Dense(2, activation='softmax'))    
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    net.fit(spect_list, category_list,epochs=50)
    net.summary()
    return net    

def create_network_mfcc(spect_list_mfcc, category_list_mfcc, spect_val_mfcc, category_val_mfcc):
    net = models.Sequential()
    net.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape=(1, 26, 1), padding='same'))
    net.add(Conv2D(32, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Flatten())
    net.add(Dense(2, activation='softmax'))    
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    net.fit(spect_list_mfcc, category_list_mfcc, epochs=200)
    net.summary()
    return net  

def create_network_mfcc_plus(spect_list_mfcc_plus, category_list_mfcc_plus, spect_val_mfcc_plus, category_val_mfcc_plus):
    net = models.Sequential()
    net.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape=(1, 39, 1), padding='same'))
    net.add(Conv2D(32, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same'))
    net.add(Flatten())
    net.add(Dense(2, activation='softmax'))    
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    net.fit(spect_list_mfcc_plus, category_list_mfcc_plus,epochs=200)
    net.summary()
    return net   

def load_audio_mfcc_plus(path, category, fileid):
    print(fileid)
    audio_file = mp.AudioFileClip(path, fps=16000);
    audio = audio_file.to_soundarray()
    audio = (audio[:, 0] + audio[:, 1]) / 2
    mfcc_structure = psf.mfcc(audio, samplerate=16000, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)

    
    mfcc_structure = np.asarray(mfcc_structure) 
    
    #plt.show()
    r = int(len(mfcc_structure[:,0]))
    for i in range(0, r):
        a = audio[i * 9216 : (i + 1) * 9216]
        m = mfcc_structure[i,:]
    
        zero_crossings       = ((a[:-1] * a[1:]) < 0).sum() # Source: https://stackoverflow.com/questions/30272538/python-code-for-counting-number-of-zero-crossings-in-an-array
        zero_crossings       = zero_crossings / (10 ** 3)
        maximum_amplitude    = np.max(plt.psd(a)[0])
        spectral_centroid    = librosa.feature.spectral_centroid(y=a, n_fft=16384, sr=16000)
        spectral_centroid    = np.resize(spectral_centroid, (1, 11))
        spectral_centroid    = spectral_centroid / (10 ** 3)
    
        m = np.append(m, zero_crossings)
        m = np.append(m, maximum_amplitude)
        m = np.append(m, spectral_centroid)
        m = utils.normalize(m)
        spect_list_mfcc_plus.append(m)
        category_list_mfcc_plus.append(category)
    audio_file.close()

def load_audio_mfcc(path, category, fileid):
    print(fileid)
    audio_file = mp.AudioFileClip(path, fps=16000);
    audio = audio_file.to_soundarray()
    audio = (audio[:, 0] + audio[:, 1]) / 2
    mfcc_structure = psf.mfcc(audio, samplerate=16000, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)
    mfcc_structure = utils.normalize(mfcc_structure)
    mfcc_structure = np.asarray(mfcc_structure) 
    #plt.show()
    r = int(len(mfcc_structure[:,0]))
    for i in range(0, r):
        m = mfcc_structure[i,:]
        spect_list_mfcc.append(m)
        category_list_mfcc.append(category)
    audio_file.close()
    
def load_audio(path, category, fileid):
    print(fileid)
    audio_file = mp.AudioFileClip(path, fps=44375);
    audio = audio_file.to_soundarray()
    audio = (audio[:, 0] + audio[:, 1]) / 2
    spectrogram, f, t, im = plt.specgram(audio, window=np.hanning(512), noverlap=0, Fs=44375, NFFT=512)
    spectrogram = utils.normalize(spectrogram)
    spectrogram = np.asarray(spectrogram) 
    #plt.show()
    if(len(spectrogram[0]) < 100):
        print("Small: " + str(len(spectrogram[0])))
        spectrogram = np.resize(spectrogram, (129, 100))
        print("Resized to " + str(len(spectrogram[0])) + ".")
    r = int(len(spectrogram[0]) / 100)
    for i in range(1, r + 1):
        s = spectrogram[:,100*(i - 1):100*i]
        spect_list.append(s)	
        category_list.append(category)
    audio_file.close()
        
def load_movie(path, start, length, FPS):
    videoclip = mp.VideoFileClip(path, audio=False).subclip(start, start + length)
    audioclip = mp.AudioFileClip(path, fps=FPS).subclip(start, start + length)
    return audioclip, videoclip

def test_audio(path, start, length, FPS):
    i = 0
    v = 0
    for this_start in range(start, start + length, 30):
        j = 0
        test_list = []
        result = []
        print(this_start)
        audio, video = load_movie(path, this_start, 30, FPS)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        spectrogram, f, time, im = plt.specgram(audio_array, window=np.hanning(512), noverlap=0, Fs=FPS, NFFT=512)
        spectrogram = utils.normalize(spectrogram)
        spectrogram = np.asarray(spectrogram)
        r = int(len(spectrogram[0]) / 100)
        for k in range(1, r + 1):
            s = spectrogram[:,100*(k - 1):100*k]
            test_list.append(s)  
        
        for t in test_list:
            t = t.reshape(1, t.shape[0], t.shape[1], 1)
            result.append(cnn.predict(t))
        for res in result:
            m = max(res)
            m = max(m)
            i = i + 1
            j = j + 1
            if(res[0][0] == m):
                print("Segment " + str(i) + " is non-violent.")
                video.save_frame("Output/Spectrogram/Non-Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 1.152)
                wav.write("Output/Spectrogram/Non-Violent/Sound/frame" + str(i) + ".wav", 44375, audio_array[int((j - 1) * FPS * 1.152):int(j * FPS * 1.152)])
            if(res[0][1] == m):
                v = v + 1
                print("Segment " + str(i) + " is violent.")
                video.save_frame("Output/Spectrogram/Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 1.152)
                wav.write("Output/Spectrogram/Violent/Sound/frame" + str(i) + ".wav", 44375, audio_array[int((j - 1) * FPS * 1.152):int(j * FPS * 1.152)])
        video.close()
        audio.close()
    print("Amount of violence: " + str(v / i * 100) + "%")
    
def test_audio_mfcc(path, start, length, FPS):
    i = 0
    v = 0
    for this_start in range(start, start + length, 30):
        j = 0
        test_list_mfcc = []
        result = []
        print(this_start)
        audio, video = load_movie(path, this_start, 30, FPS)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        mfcc_structure = psf.mfcc(audio_array, samplerate=16000, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)
        mfcc_structure = utils.normalize(mfcc_structure)
        mfcc_structure = np.asarray(mfcc_structure)
        r = int(len(mfcc_structure[:,0]))
        for k in range(0, r):
            s = mfcc_structure[k,:]
            test_list_mfcc.append(s)  
        
        for t in test_list_mfcc:
            t = t.reshape(1, 1, 26, 1)
            result.append(cnn_mfcc.predict(t))
        for res in result:
            m = max(res)
            m = max(m)
            i = i + 1
            j = j + 1
            if(res[0][0] == m):
                print("Segment " + str(i) + " is non-violent.")
                video.save_frame("Output/MFCC/Non-Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/MFCC/Non-Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
            if(res[0][1] == m):
                v = v + 1
                print("Segment " + str(i) + " is violent.")
                video.save_frame("Output/MFCC/Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/MFCC/Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
        video.close()
        audio.close()
    print("Amount of violence: " + str(v / i * 100) + "%")
    
def test_audio_mfcc_plus(path, start, length, FPS):
    i = 0
    v = 0
    for this_start in range(start, start + length, 30):
        j = 0
        test_list_mfcc_plus = []
        result = []
        print(this_start)
        audio, video = load_movie(path, this_start, 30, FPS)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        mfcc_structure = psf.mfcc(audio_array, samplerate=16000, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)
        mfcc_structure = np.asarray(mfcc_structure)
        r = int(len(mfcc_structure[:,0]))
        for k in range(0, r):
            s = mfcc_structure[k,:]
            a = audio_array[k * 9056 : (k + 1) * 9056]
    
            zero_crossings       = ((a[:-1] * a[1:]) < 0).sum() # Source: https://stackoverflow.com/questions/30272538/python-code-for-counting-number-of-zero-crossings-in-an-array
            zero_crossings       = zero_crossings / (10 ** 3)
            maximum_amplitude    = np.max(plt.psd(a)[0])
            spectral_centroid    = librosa.feature.spectral_centroid(y=a, n_fft=16384, sr=16000)
            spectral_centroid    = np.resize(spectral_centroid, (1, 11))
            spectral_centroid    = spectral_centroid / (10 ** 3)
        
            s = np.append(s, zero_crossings)
            s = np.append(s, maximum_amplitude)
            s = np.append(s, spectral_centroid)
            s = utils.normalize(s)
            
            test_list_mfcc_plus.append(s)  
        
        for t in test_list_mfcc_plus:
            t = t.reshape(1, 1, 39, 1)
            result.append(cnn_mfcc_plus.predict(t))
        for res in result:
            m = max(res)
            m = max(m)
            i = i + 1
            j = j + 1
            if(res[0][0] == m):
                print("Segment " + str(i) + " is non-violent.")
                video.save_frame("Output/MFCC+/Non-Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/MFCC+/Non-Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
            if(res[0][1] == m):
                v = v + 1
                print("Segment " + str(i) + " is violent.")
                video.save_frame("Output/MFCC+/Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/MFCC+/Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
        video.close()
        audio.close()
    print("Amount of violence: " + str(v / i * 100) + "%")
    
def test_audio_svm(path, start, length, FPS, machine):
    i = 0
    v = 0
    for this_start in range(start, start + length, 30):
        j = 0
        test_list_mfcc = []
        result = []
        print(this_start)
        audio, video = load_movie(path, this_start, 30, FPS)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        mfcc_structure = psf.mfcc(audio_array, samplerate=16000, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)
        mfcc_structure = utils.normalize(mfcc_structure)
        mfcc_structure = np.asarray(mfcc_structure)
        r = int(len(mfcc_structure[:,0]))
        for k in range(0, r):
            s = mfcc_structure[k,:]
            test_list_mfcc.append(s)  
            
        test_list_mfcc = vectorize(test_list_mfcc)
        
        for t in test_list_mfcc:
            t = t.reshape(1, 26)
            print(t)
            label, acc, val = svm.svm_predict([], t, machine, '-b 1')
            result.append(label)
        
        for res in result:
            res = res[0]
            i = i + 1
            j = j + 1
            if(res < 0.5):
                print("Segment " + str(i) + " is non-violent.")
                video.save_frame("Output/SVM/Non-Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/SVM/Non-Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
            else:
                v = v + 1
                print("Segment " + str(i) + " is violent.")
                video.save_frame("Output/SVM/Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 0.566)
                wav.write("Output/SVM/Violent/Sound/frame" + str(i) + ".wav", FPS, audio_array[int((j - 1) * FPS * 0.566):int(j * FPS * 0.566)])
        video.close()
        audio.close()
    print("Amount of violence: " + str(v / i * 100) + "%")
    
def test_audio_both(path, start, length, FPS1, FPS2):
    i = 0
    v = 0
    for this_start in range(start, start + length, 30):
        j = 0
        test_list_spect = []
        test_list_mfcc = []
        result_mfcc = []
        result_spect = []
        print(this_start)
        audio, video = load_movie(path, this_start, 30, FPS1)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        mfcc_structure = psf.mfcc(audio_array, samplerate=FPS1, winlen=0.576, winstep=0.576, nfft=16384, numcep=26, nfilt=52)
        mfcc_structure = utils.normalize(mfcc_structure)
        mfcc_structure = np.asarray(mfcc_structure)
        
        audio.close()
        video.close()
        
        audio, video = load_movie(path, this_start, 30, FPS2)
        audio_array = audio.to_soundarray()
        audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2
        spectrogram, f, time, im = plt.specgram(audio_array, window=np.hanning(512), noverlap=0, Fs=FPS2, NFFT=512)
        spectrogram = utils.normalize(spectrogram)
        spectrogram = np.asarray(spectrogram)
        r1 = int(len(mfcc_structure[:,0]))
        r2 = int(len(spectrogram[0]) / 100)
        for k in range(0, r1):
            s = mfcc_structure[k,:]
            test_list_mfcc.append(s)  
        for k in range(1, r2 + 1):
            s = spectrogram[:,100*(k - 1):100*k]
            test_list_spect.append(s)  
        
        for t in test_list_mfcc:
            t = t.reshape(1, 1, 26, 1)
            result_mfcc.append(cnn_mfcc.predict(t))
        for t in test_list_spect:
            t = t.reshape(1, t.shape[0], t.shape[1], 1)
            result_spect.append(cnn.predict(t))
            
        result_mfcc = result_mfcc[:-1]

        result_mfcc_new = result_mfcc
        
        for z in range(0, int(len(result_mfcc) / 2) - 1):
            result_mfcc_new[z] = (result_mfcc[2 * z] + result_mfcc[2 * z + 1]) / 2
            
        result_mfcc = result_mfcc_new[:int(len(result_mfcc) / 2)]
        
        for z in range(0, len(result_mfcc)):
            avg = (result_mfcc[z] + result_spect[z]) / 2
            m = max(avg)
            m = max(m)
            i = i + 1
            j = j + 1
            if(avg[0][0] == m):
                print("Segment " + str(i) + " is non-violent.")
                video.save_frame("Output/MFCC + Spectrogram/Non-Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 1.152)
                wav.write("Output/MFCC + Spectrogram/Non-Violent/Sound/frame" + str(i) + ".wav", 44375, audio_array[int((j - 1) * FPS2 * 1.152):int(j * FPS2 * 1.152)])
            if(avg[0][1] == m):
                v = v + 1
                print("Segment " + str(i) + " is violent.")
                video.save_frame("Output/MFCC + Spectrogram/Violent/Image/frame" + str(i) +".jpeg", t = (j - 1) * 1.152)
                wav.write("Output/MFCC + Spectrogram/Violent/Sound/frame" + str(i) + ".wav", 44375, audio_array[int((j - 1) * FPS2 * 1.152):int(j * FPS2 * 1.152)])
        video.close()
        audio.close()
    print("Amount of violence: " + str(v / i * 100) + "%")
    
def five_sec_segments(path, start, length, FPS):
    audio, video = load_movie(path, start, length, FPS)
    for i in range(0, int(length / 5)):
        audio.subclip(i * 5, i * 5 + 5).write_audiofile("Segments of Sound/frame" + str(i) + ".wav")
    audio.close()
    
def load_files(path1, path2):
    fileid = 1
    
    while(Path(path1 + str(fileid) + ".wav").is_file()):
        load_audio(path1 + str(fileid) + ".wav", 1, fileid)
        load_audio_mfcc(path1 + str(fileid) + ".wav", 1, fileid)
        load_audio_mfcc_plus(path1 + str(fileid) + ".wav", 1, fileid)
        fileid = fileid + 1
    
    fileid = 1
        
    while(Path(path2 + str(fileid) + ".wav").is_file()):
        load_audio(path2 + str(fileid) + ".wav", 0, fileid)
        load_audio_mfcc(path2 + str(fileid) + ".wav", 0, fileid)
        load_audio_mfcc_plus(path2 + str(fileid) + ".wav", 0, fileid)
        fileid = fileid + 1
        
def test_audioset_data(path1, path2):
    global spect_list
    global spect_list_mfcc
    global category_list
    global category_list_mfcc
    load_files(path1, path2)
    result_mfcc = []
    result_spect = []
    nv = 0
    v = 0
    nv_all = 0
    v_all = 0
   
    spect_list = np.asarray(spect_list)
    spect_list = spect_list.reshape(spect_list.shape[0], spect_list.shape[1], spect_list.shape[2], 1)
    category_list = utils.to_categorical(category_list, 2)
    spect_list_mfcc = np.asarray(spect_list_mfcc)
    spect_list_mfcc = spect_list_mfcc.reshape(spect_list_mfcc.shape[0], 1, spect_list_mfcc.shape[1], 1)
    category_list_mfcc = utils.to_categorical(category_list_mfcc, 2)
    
    for t in spect_list_mfcc:
        t = t.reshape(1, 1, 26, 1)
        result_mfcc.append(cnn_mfcc.predict(t))
    for t in spect_list:
        t = t.reshape(1, t.shape[0], t.shape[1], 1)
        result_spect.append(cnn.predict(t))
    
    for z in range(0, len(result_mfcc)):
        res = result_mfcc[z]
        m = max(res)
        m = max(m)
        if(res[0][0] == m):
            nv_all = nv_all + 1
            if(category_list_mfcc[z][0] == 1):
                nv = nv + 1
        if(res[0][1] == m):
            v_all = v_all + 1
            if(category_list_mfcc[z][1] == 1):
                v = v + 1
    print("True classifications:" + str(v + nv) + "/" + str(len(result_mfcc)))
    print("True violent classifications:" + str(v) + "/" + str(v_all))
    print("True non-violent classifications:" + str(nv) + "/" + str(nv_all))
    print("Success rate for MFCC:")
    print(str(((v + nv) / len(result_mfcc)) * 100) + "%")
    
    nv = 0
    v = 0
    nv_all = 0
    v_all = 0 
    
    for z in range(0, len(result_spect)):
        res = result_spect[z]
        m = max(res)
        m = max(m)
        if(res[0][0] == m):
            nv_all = nv_all + 1
            if(category_list[z][0] == 1):
                nv = nv + 1
        if(res[0][1] == m):
            v_all = v_all + 1
            if(category_list[z][1] == 1):
                v = v + 1
                
    print("True classifications:" + str(v + nv) + "/" + str(len(result_spect)))       
    print("True violent classifications:" + str(v) + "/" + str(v_all))
    print("True non-violent classifications:" + str(nv) + "/" + str(nv_all))         
    print("Success rate for spectrogram:")
    print(str(((v + nv) / len(result_spect)) * 100) + "%")
    

#                
##    
load_files("Violent/", "Non-violent/")
#    
# SVM
vector = vectorize(spect_list_mfcc)
m = build_problem(spect_list_mfcc, category_list_mfcc)
svm.svm_save_model('audio.model', m)

m = svm.svm_load_model('audio.model')



# CNN
#  
spect_list = np.asarray(spect_list)
spect_list = spect_list.reshape(spect_list.shape[0], spect_list.shape[1], spect_list.shape[2], 1)
category_list = utils.to_categorical(category_list, 2)
spect_list_mfcc = np.asarray(spect_list_mfcc)
spect_list_mfcc = spect_list_mfcc.reshape(spect_list_mfcc.shape[0], 1, spect_list_mfcc.shape[1], 1)
category_list_mfcc = utils.to_categorical(category_list_mfcc, 2)
spect_list_mfcc_plus    = np.asarray(spect_list_mfcc_plus)
spect_list_mfcc_plus    = spect_list_mfcc_plus.reshape(spect_list_mfcc_plus.shape[0], 1, spect_list_mfcc_plus.shape[2], 1)
category_list_mfcc_plus = utils.to_categorical(category_list_mfcc_plus, 2)

#fold_network(spect_list, category_list, create_network)
#fold_network(spect_list_mfcc, category_list_mfcc, create_network_mfcc)
#fold_network(spect_list_mfcc_plus, category_list_mfcc_plus, create_network_mfcc_plus)
#
#print("First, spectrograms.")
##
#cnn = create_network(spect_list, category_list, np.zeros(0), np.zeros(0))
#
#model_json = cnn.to_json()
#with open("audio.json", "w") as json_file:
#    json_file.write(model_json)
#
#cnn.save_weights("audio.h5")
# 
#print("Now, MFCC.")
#
#cnn_mfcc = create_network_mfcc(spect_list_mfcc, category_list_mfcc, np.zeros(0), np.zeros(0))
#    
#model_json_mfcc = cnn_mfcc.to_json()
#with open("audio_mfcc.json", "w") as json_file_mfcc:
#    json_file_mfcc.write(model_json_mfcc)
#
#cnn_mfcc.save_weights("audio_mfcc.h5")
#
#print("Now, MFCC+.")
##
#cnn_mfcc_plus = create_network_mfcc_plus(spect_list_mfcc_plus, category_list_mfcc_plus, np.zeros(0), np.zeros(0))
##    
#model_json_mfcc_plus = cnn_mfcc_plus.to_json()
#with open("audio_mfcc+.json", "w") as json_file_mfcc_plus:
#    json_file_mfcc_plus.write(model_json_mfcc_plus)
#
#cnn_mfcc_plus.save_weights("audio_mfcc+.h5")

# 
json_file = open('audio.json', 'r')
json = json_file.read()
json_file.close()
cnn = models.model_from_json(json)
cnn.load_weights("audio.h5")

json_file_mfcc = open('audio_mfcc.json', 'r')
json = json_file_mfcc.read()
json_file_mfcc.close()
cnn_mfcc = models.model_from_json(json)
cnn_mfcc.load_weights("audio_mfcc.h5")

json_file_mfcc_plus = open('audio_mfcc+.json', 'r')
json = json_file_mfcc_plus.read()
json_file_mfcc_plus.close()
cnn_mfcc_plus = models.model_from_json(json)
cnn_mfcc_plus.load_weights("audio_mfcc+.h5")
#
#five_sec_segments("E:/Diğer/Filmler/Upgrade (2018)/Upgrade.2018.1080p.WEBRip.x264-[YTS.AM].mp4", 0, 5640, 44375)
#    
#test_audio("E:/Diğer/Filmler/Martyrs (2008)/Martyrs.mp4", 1800, 600, 44375)
#test_audio_mfcc("E:/Diğer/Filmler/Martyrs (2008)/Martyrs.mp4", 1800, 600, 16000)
#test_audio_mfcc_plus("E:/Diğer/Filmler/Martyrs (2008)/Martyrs.mp4", 1800, 600, 16000)
#test_audio_both("E:/Diğer/Filmler/Martyrs (2008)/Martyrs.mp4", 1800, 600, 16000, 44375)
#test_audio_svm("E:/Diğer/Filmler/Martyrs (2008)/Martyrs.mp4", 1800, 600, 16000, m)
#   
#youtube_fetch()
#
#test_audioset_data("Tests/Violent/", "Tests/Non-Violent/")
