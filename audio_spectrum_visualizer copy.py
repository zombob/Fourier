import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import threading
import time

samplerate = 44100
frames_per_buffer = 2048

fig, ax = plt.subplots()
x = np.fft.rfftfreq(frames_per_buffer, d=1./samplerate)

min_freq = 20
max_freq = 20000
music_band = (x >= min_freq) & (x <= max_freq)
x_music = x[music_band]

line, = ax.plot(x_music, np.zeros_like(x_music))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Log Amplitude')
ax.set_title('Real-time Audio Spectrum (Music Band, Log Scale)')
ax.set_xlim(min_freq, max_freq)
ax.set_xscale('log')
ax.set_ylim(-80, 0)

latest_fft = np.zeros_like(x)
fft_lock = threading.Lock()

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    window = np.hanning(len(audio_data))
    windowed_data = audio_data * window
    fft_data = np.abs(np.fft.rfft(windowed_data)) / frames_per_buffer
    with fft_lock:
        global latest_fft
        latest_fft = fft_data
    return (in_data, pyaudio.paContinue)

def update_plot(frame):
    with fft_lock:
        y = latest_fft.copy()
    db_y = 20 * np.log10(y[music_band] + 1e-8)
    db_y = np.clip(db_y, -80, 0)
    line.set_ydata(db_y)
    return line,

def select_device(is_input=True):
    print("\n可用音频设备列表：")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if is_input and dev_info['maxInputChannels'] > 0:
            print(f"{i}: {dev_info['name']} (输入)")
        elif not is_input and dev_info['maxOutputChannels'] > 0:
            print(f"{i}: {dev_info['name']} (输出)")
    
    while True:
        try:
            idx = int(input(f"\n请输入要使用的{'输入' if is_input else '输出'}设备编号: "))
            dev_info = p.get_device_info_by_index(idx)
            if is_input and dev_info['maxInputChannels'] > 0:
                return idx, p
            elif not is_input and dev_info['maxOutputChannels'] > 0:
                return idx, p
            else:
                print("设备类型不符，请重新输入。")
        except Exception:
            print("输入无效，请重新输入。")

if __name__ == "__main__":
    print("请选择音频采集来源：")
    print("1. 麦克风")
    print("2. 扬声器（需声卡支持loopback）")
    src = input("输入1或2: ").strip()
    
    if src == "1":
        device_idx, p = select_device(is_input=True)
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=samplerate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=frames_per_buffer,
            stream_callback=audio_callback
        )
        print("正在采集麦克风音频并显示音乐频段频谱图（对数坐标），按Ctrl+C退出。")
    elif src == "2":
        device_idx, p = select_device(is_input=False)
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=samplerate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=frames_per_buffer,
            stream_callback=audio_callback,
            as_loopback=True
        )
        print("正在采集扬声器（系统回放）音频并显示音乐频段频谱图（对数坐标），按Ctrl+C退出。")
    else:
        print("无效选择，程序退出。")
        exit(1)

    stream.start_stream()
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=False)
    plt.show()
    
    stream.stop_stream()
    stream.close()
    p.terminate()