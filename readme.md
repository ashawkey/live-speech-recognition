# live speech recognition

This is a simple live ASR application example, with a simple GUI.

We use [transformers](https://huggingface.co/models) for the ASR models.

A sliding window is used to split the infinite audio input:
```
IN:            [ -----------------valid audio input----------------- ] ...
W1: [ -left- ] [ -----middle1----- ] [ -right- ]
W2:                       [ -left- ] [ -----middle2----- ] [ -right- ] 
                                                    ......................
OUT:           [ ------text1------ ] [ ------text2------ ] ...............
```


### Install

```bash
# for ubuntu, portaudio is needed for pyaudio to work.
sudo apt install portaudio19-dev

# install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python asr.py # transcribe from your microphone.
python asr.py --wav aud.wav # transcribe a local wav file.

python asr.py --model ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt # change the ASR model

# adjust the sliding window (left - middle - right)
# by default the audio is at 50 FPS (20ms per frame), so the latency will be (m + r) * 20ms
# a larger window generally leads to better recoginition performance, at the cost of a longer latency.
python asr.py -l 20 -m 60 -r 20

```

### Acknowlegement
* [LxgwWenKai](https://github.com/lxgw/LxgwWenKai).
* [wav2vec2-live](https://github.com/oliverguhr/wav2vec2-live).