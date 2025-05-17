import numpy as np
import matplotlib.pyplot as plt
import time

# 理解傅里叶傅里叶计算方式，以及载波和数据的处理。
## 1. 做一个函数发生器，并根据当前时间实时采样绘图
##    公式：f(t) = A * sin(2π * f * T + φ)
## 2. 通过傅里叶算法，从采样中计算出原始波函数

# 设置参数
freqs = [3, 7, 13]           # 不同频率
amps = [1, 0.5, 0.3]         # 不同幅度
phases = [0, np.pi/4, np.pi/2]  # 不同相位
num_signals = len(freqs)  # 信号数量

# 测试，获取当前时间下的叠加信号
def get_signal(t):
    signalcur = 0.0
    for i in range(num_signals):
        signal = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
        signalcur += signal
    return signalcur

# 实时绘制叠加信号
def plot_realtime_signal(duration=2, interval=0.01, amp=1.0, t_step=0.01):
    """ 实时绘制叠加信号
    :param duration: 绘制画布持续时间
    :param interval: 刷新间隔
    :param amp: 信号幅度
    :param t_step: 采样间隔
    :return: None
    """
    plt.ion()
    fig, ax = plt.subplots()
    t_vals = np.arange(0, duration, t_step)
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, duration)
    ax.set_ylim(-amp * 2, amp * 2)
    ax.grid(True)
    while plt.fignum_exists(fig.number):
        t0 = time.time()
        t_vals = np.arange(0, duration, t_step)
        y_vals = np.array([get_signal(t + time.time() % duration) for t in t_vals])
        line.set_data(t_vals, y_vals)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval - ((time.time() - t0) % interval))


# 实时采样
def sample_signal(duration=2, t_step=0.01):
    """ 采样信号 
    :param duration: 采样持续时间
    :param t_step: 采样间隔
    :return: 采样时间点和对应的信号值
    :rtype: tuple
    """
    t_vals = np.arange(0, duration, t_step)
    y_vals = np.array([get_signal(t) for t in t_vals])
    return t_vals, y_vals

def show_sampled_signal(duration = 5, t_step = 0.001):
    """ 显示静态采样信号
    :param duration: 采样持续时间
    :param t_step: 采样间隔
    """
    t_vals, y_vals = sample_signal(duration, t_step)
    plt.plot(t_vals, y_vals)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Sampled Signal')
    plt.grid()
    plt.show()

def show_sampled_signal_with_fft(duration=5, t_step=0.001):
    """ 显示静态采样信号和FFT
    :param duration: 采样持续时间
    :param t_step: 采样间隔
    """
    t_vals, y_vals = sample_signal(duration, t_step)
    plt.subplot(2, 1, 1)
    plt.plot(t_vals, y_vals)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Sampled Signal')
    plt.grid()

    # 计算FFT
    N = len(y_vals)
    fft_vals = np.fft.fft(y_vals)
    freqs = np.fft.fftfreq(N, d=t_step)

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:N // 2], np.abs(fft_vals)[:N // 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of Sampled Signal')
    plt.grid()
    plt.tight_layout()
    plt.show()

def dft(signal,t_step=1):
    """ 手动计算DFT (离散傅里叶变换), 时间复杂度O(N^2)
    公式: X[k] = Σ x[n] * e^(-2πi * k * n / N)
    如果需要恢复原始幅度值，需要将幅度值结果乘以2除以N。乘以2是因为DFT的对称性（正负频率各占一半），除以N是因为累加了N次。
    公式：X[k] = 2/N Σ x[n] * e^(-2πi * k * n / N)
    :param signal: 输入信号
    :param t_step: 采样间隔
    :return: 频率和幅度
    """
    N = len(signal)
    freq = np.fft.fftfreq(N, d=t_step)
    dft_vals = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            dft_vals[k] += signal[n] * np.exp(-2j * np.pi * k * n / N) # 公式：X[k] = Σ x[n] * e^(-2πi * k * n / N)
            # if k == 1 and n ==1:
            #     print(f"n: {n}, k: {k}, signal[n]: {signal[n]}, exp(-2j * np.pi * k * n / N): {np.exp(-2j * np.pi * k * n / N)}")
            #     print(f"k: {k}, n: {n}, dft_vals[k]: {dft_vals[k]}")
        # print(f"DFT Value for k={k}: abs: {np.abs(dft_vals[k])} , {dft_vals[k]}")
    # print(f"Frequency: {freq}")
    # print(f"DFT Values: {dft_vals}")
    # 归一化幅度值 *2 / N
    dft_vals *= 2
    dft_vals /= N
    return freq, dft_vals

# 测试FFT函数
def test_dft():
    """ 测试FFT函数
    """
    # t_vals, y_vals = sample_signal(duration=2, t_step=0.005)
    # freq, fft_vals = dft(y_vals, t_step=0.005)
    
    y_vals = [1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5]
    freq, fft_vals = dft(y_vals, t_step=0.125)
    
    # 只需取前一半，因为DFT是对称的，后一半是前一半的共轭，其频率为负值，表示逆时针。
    plt.plot(freq[:len(freq)//2], np.abs(fft_vals)[:len(fft_vals)//2]) 
    # plt.plot(freq[:len(freq)//2], np.abs(np.real(fft_vals))[:len(fft_vals)//2]) 
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of Sampled Signal')
    plt.grid()
    plt.show()


# 只计算某个频率的DFT
def dft_single_freq(signal, freq):
    """ 计算单个频率的DFT
    :param signal: 输入信号
    :param freq: 频率
    :return: 频率和幅度
    """
    N = len(signal)
    dft_val=0.0
    for n in range(N):
        dft_val += signal[n] * np.exp(-2j * np.pi * freq * n / N) # 公式：X[k] = Σ x[n] * e^(-2πi * k * n / N)
    return dft_val

def test_dft_single_freq(freqs=[0.01, 0.2, 1, 3, 7, 8, 9, 10]):
    """ 只计算指定频率的DFT并绘图
    :param freqs: 频率列表
    """
    t_vals, y_vals = sample_signal(duration=5, t_step=0.001)
    magnitudes = []
    for freq in freqs:
        dft_vals = dft_single_freq(y_vals, freq)
        print(f"Frequency: {freq}, Magnitude: {np.abs(dft_vals)}")
        

if __name__ == "__main__":
    # 示例：采样，幅度范围2，采样间隔0.001s，图表显示2秒
    # show_sampled_signal(duration=5, t_step=0.001)

    # 示例：实时绘制，幅度范围2，采样间隔0.001s，图表显示2秒
    # plot_realtime_signal(duration=5, interval=0.05, amp=1.0, t_step=0.01)

    # 示例：从采样信号中计算FFT
    # show_sampled_signal_with_fft(duration=5, t_step=0.001)

    # 示例：测试FFT函数
    test_dft()

    # 示例：测试单个频率的DFT
    # 生成从 0.01 到 1 Hz 的频率列表
    # freqs = np.arange(1, 10, 1)
    # test_dft_single_freq(freqs=freqs)

