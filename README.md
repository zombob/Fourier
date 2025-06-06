# 傅里叶变换 DFT 和 FFT 的算法原理演示

## DFT: 计算出的结果是*某1个*频率的*余弦分量*和*正弦分量*
## 公式: X[k] = Σ x[n] * e^(-2πi * k * n / N) = Σ x[n] * (cos(2π * k * n / N) - i * sin(2π * k * n / N))
## 幅度值: |X[k]| = sqrt( (Re(X[k]))^2 + (Im(X[k]))^2 ) = sqrt( (Σ x[n] * cos(2π * k * n / N))^2 + (Σ x[n] * sin(2π * k * n / N))^2 )
## 真实幅度值: X'[k] = X[k] / N * 2
  * 解释：X[k] 为累加了 N 次的值，所以要除以 N，但频谱分为了正负2部分（正表示顺时针、负表示逆时针）且2部分共轭，所以人格一个部分都代表半个原始值，所以要乘以2
* 将所有频率的DFT的幅度值绘制出来，会形成一个所有频段的分布图

# 算法演示
* **fourier.py** 为 DTF（离散傅里叶变换） 的手动算法演示，没有使用FFT算法
  * DFT 和 FFT 的区别主要为：
    * FFT先将旋转因子 e^(-2πi * k * n / N) 中的一部分（把k*n看作一个整体，取值 0 ~ N-1）保存起来，计算时只需计算出第一个周期的数值，将数值保存，后续计算时，只需计算出 x % N , 然后查表直接取数值。
    * 这样就可以将时间复杂度从 O(N^2) 降低到 O(NlogN)

* **audioaudio_spectrum_visualizer.py** 使用了FFT算法将麦克风采集到的音频数据进行FFT变换，绘制出频谱图
  * 该文件使用了 PyAudio 库来获取麦克风音频数据
  * 使用了 numpy 库来进行 FFT 计算
  * 使用了 matplotlib 库来绘制频谱图