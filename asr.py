import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor

import soundfile as sf
import resampy

import dearpygui.dearpygui as dpg

from multiprocessing import Queue, Process, Event

class ASRGUI:
    def __init__(self, opt):

        self.opt = opt

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.mode = 'live' if opt.wav == '' else 'file'

        # prepare context cache
        # each segment is (stride_left + ctx + stride_right) * 20ms, latency should be (ctx + stride_right) * 20ms
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.text = '[START]\n'
        self.terminated = False
        self.frames = []

        # pad left frames
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        # create input stream
        if self.mode == 'file':
            self.stream = self.create_file_stream()
        else:
            # start a background process to read frames
            self.queue = Queue()
            self.exit_event = Event()

            def _read_frame():
                audio_instance, stream = self.create_pyaudio_stream()

                while True:
                    if self.exit_event.is_set():
                        break
                    frame = stream.read(self.chunk, exception_on_overflow=False)
                    frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767 # [chunk]
                    # print(f'[INFO] read frame {frame.shape}')
                    self.queue.put(frame)

                stream.stop_stream()
                stream.close()
                audio_instance.terminate()
            
            self.process_read_frame = Process(target=_read_frame)

        # current location of audio
        self.idx = 0

        # create wav2vec model
        print(f'[INFO] loading model {self.opt.model}...')
        self.processor = AutoProcessor.from_pretrained(opt.model)
        self.model = AutoModelForCTC.from_pretrained(opt.model).to(self.device)

        # prepare to save logits
        if self.opt.save_logits:
            self.logits = []

        # start gui
        dpg.create_context()
        self.register_dpg()

        print(f'[INFO] starting read frame thread...')
        self.process_read_frame.start()
        
        self.test_step()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        
        if self.mode == 'live':
            
            self.exit_event.set()
            self.process_read_frame.join()

        dpg.destroy_context()


    def test_step(self):

        if self.terminated:
            return

        # get a frame of audio
        frame = self.get_audio_frame()
        
        # the last frame
        if frame is None:
            # terminate, but always run the network for the left frames
            self.terminated = True
        else:
            self.frames.append(frame)
            # context not enough, do not run network.
            if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
                return
        
        inputs = np.concatenate(self.frames) # [N * chunk]

        # discard the old part
        if not self.terminated:
            self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        t0 = time.time()

        logits, text = self.frame_to_text(inputs)

        # very naive, just concat the text output.
        if text != '':
            self.text = self.text + ' ' + text
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.time() - t0

        # save logits
        if self.opt.save_logits:
            self.logits.append(logits)

        # will only run once at ternimation
        if self.terminated:
            self.text += '\n[END]'
            if self.opt.save_logits:
                print(f'[INFO] save logits... ')
                logits = torch.cat(self.logits, dim=0) # [N, 32]
                print(logits.shape)
                # temp: unfold 16x window...
                K = logits.shape[-1] # n_characters, 32 for wav2vec2
                window_size = 16
                padding = window_size // 2
                logits = logits.view(-1, K).permute(1, 0).contiguous() # [K, M]
                logits = logits.view(1, K, -1, 1) # [1, K, M, 1]
                unfold_logits = F.unfold(logits, kernel_size=(window_size, 1), padding=(padding, 0), stride=(2, 1)) # [1, K * window_size, M / 2 + 1]
                unfold_logits = unfold_logits.view(K, window_size, -1).permute(2, 1, 0).contiguous() # [M / 2 + 1, window_size, K]
                print(unfold_logits.shape)
                np.save(self.opt.wav.replace('.wav', '.npy'), unfold_logits.cpu().numpy())
                print(f"[INFO] saved logits to {self.opt.wav.replace('.wav', '_wv.npy')}")

        dpg.set_value("_log_text", self.text)
        dpg.set_value("_log_infer_time", f'{1000 * t:.4f}ms ({int(1/t)} FPS)')

    
    def create_file_stream(self):
    
        stream, sample_rate = sf.read(opt.wav) # [T*sample_rate,] float64
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        print(f'[INFO] loaded audio stream {opt.wav}: {stream.shape}')

        return stream


    def create_pyaudio_stream(self):

        import pyaudio

        print(f'[INFO] creating live audio stream ...')

        audio = pyaudio.PyAudio()
        
        # get devices
        info = audio.get_host_api_info_by_index(0)
        n_devices = info.get('deviceCount')

        for i in range(0, n_devices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f'[INFO] choose audio device {name}, id {i}')
                break
        
        # get stream
        stream = audio.open(input_device_index=i,
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk)

        return audio, stream

    
    def get_audio_frame(self):

        if self.mode == 'file':
        
            if self.idx < self.stream.shape[0]:
                frame = self.stream[self.idx: self.idx + self.chunk]
                self.idx = self.idx + self.chunk
                return frame
            else:
                return None
        
        else:

            # get from multiprocessing queue.
            frame = self.queue.get()

            self.idx = self.idx + self.chunk

            return frame

        
    def frame_to_text(self, frame):
        # frame: [N * 320], N = (context_size + 2 * stride_size)
        
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            result = self.model(inputs.input_values.to(self.device))
            logits = result.logits # [1, N - 1, 32]
        
        # cut off stride
        left = max(0, self.stride_left_size)
        right = min(logits.shape[1], logits.shape[1] - self.stride_right_size + 1) # +1 to make sure output is the same length as input.

        # do not cut right if terminated.
        if self.terminated:
            right = logits.shape[1]

        logits = logits[:, left:right]

        # print(frame.shape, inputs.input_values.shape, logits.shape)
    
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        # labels = np.array([' ', ' ', ' ', '-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'])
        # print(''.join(labels[predicted_ids[0].detach().cpu().long().numpy()]))
        # print(predicted_ids[0])
        # print(transcription)

        return logits[0], transcription

        
    def register_dpg(self):

        dpg.create_viewport()
        dpg.setup_dearpygui()

        ### register font (for chinese)
        with dpg.font_registry():
            with dpg.font('LXGWWenKai-Regular.ttf', 20) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
        dpg.bind_font(default_font)

      
        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=400, height=400):

            # time
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            dpg.add_text("", tag="_log_text", wrap=0)

        dpg.set_primary_window("_primary_window", True)
        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            self.test_step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, default='')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    parser.add_argument('--save_logits', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length.
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)
    
    opt = parser.parse_args()

    with ASRGUI(opt) as gui:
        gui.render()