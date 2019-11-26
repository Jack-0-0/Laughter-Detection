# Preprocessing
These files will convert the audio files into train, test and validation datasets in the format [id, spectrogram, label]. There is a jupyter notebook for each of the three datasets:
* English - preprocessing_en
* Chinese - preprocessing_ch
* German - preprocessing_de

## Requirements
The following packages are needed for preprocessing:
* Python 3.7.0
* tgt 1.4.4
* tqdm 4.36.1
* numpy 1.17.2
* pandas 0.25.1
* librosa 0.7.0
* jupyter 1.0.0
* networkx 2.4
* SoX 14.4.1

## Datasets
The raw datasets for all languages and the English metadata file can be downloaded from [link to be added].

Expected German data structure:

* audio
  * r1
    * r1.wav
  * r2
    * r2.wav
  * ...
  * r19
    * r19.wav
* transcriptions_annotations
  * r1
    * r1.TextGrid
  * r2
    * r2.TextGrid
  * ...
  * r19
    * r19.TextGrid

Expected Chinese data structure:

* audio
  * r1
    * r1_1.mp3
    * r1_2.mp3
    * r1_3.mp3
  * r2
    * r1_1.mp3
    * r1_2.mp3
    * r1_3.mp3
  * ...
  * r10
    * r10_1.mp3
    * r10_2.mp3
    * r10_3.mp3

* transcriptions_annotations
  * r1
    * r1_1.TextGrid
    * r1_2.TextGrid
    * r1_3.TextGrid
  * r2
    * r2_1.TextGrid
    * r2_2.TextGrid
    * r2_3.TextGrid
  * ...
  * r10
    * r10_1.TextGrid
    * r10_2.TextGrid
    * r10_3.TextGrid

Expected English data structure:

* switchboard1
  * swb1
    * sw02001.sph
    * sw02005.sph
    * sw02006.sph
    * …
* MS_word_laughter_timings
  * data
    * alignments
      * 2
        * sw2005A-ms98-a-penn.text
        * sw2005B-ms98-a-penn.text
        * ...
      * 3
        * sw3000A-ms98-a-penn.text
        * sw3000B-ms98-a-penn.text
        * ...
      * 4
        * sw4004A-ms98-a-penn.text
        * sw4004B-ms98-a-penn.text
        * ...

The English audio is in .sph format, to convert to .wav use SoX:

```shell
for f in *.sph; do sox -t sph "$f" -b 16  -t wav "${f%.*}.wav"; done
```
## Usage
At the top of the notebook file, set the paths to load the audio, annotations and where to save the spectrograms and datasets, for example on the German file change below to the appropriate locations:

```python
audio_path = '/laughter/DUEL/de/audio'
annotation_path = '/laughter/DUEL/de/transcriptions_annotations/'
save_path = '/laughter/DUEL/datasets/de/'
```
Then run each code block in order. This will result in a train, test and validation datset saved in the format [id, spectrogram, label].
