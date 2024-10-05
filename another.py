import numpy as np
from obspy import Stream, Trace
from obspy.core import UTCDateTime

# Налаштування параметрів
n_samples = 1000  # Кількість вибірок
sampling_rate = 100  # Частота дискретизації (Гц)
t = np.linspace(0, n_samples / sampling_rate, n_samples, endpoint=False)  # Часова шкала

# Генерація синусоїдального сигналу з частотою 5 Гц
frequency = 5  # Частота сигналу (менше 10 Гц)
signal = np.sin(2 * np.pi * frequency * t)

# Генерація випадкового шуму
noise = 0.1 * np.random.randn(n_samples)  # Амплітуда шуму
data = signal + noise  # Об'єднання сигналу та шуму

# Створення об'єкта Trace
trace = Trace()
trace.data = data
trace.stats.network = "NET"
trace.stats.station = "STA"
trace.stats.location = "00"
trace.stats.channel = "HHZ"
trace.stats.sampling_rate = sampling_rate
trace.stats.starttime = UTCDateTime(2024, 10, 5, 0, 0, 0)  # Початковий час

# Створення об'єкта Stream і додавання до нього Trace
stream = Stream(traces=[trace])

# Збереження даних у форматі MiniSEED
stream.write("clear_input.mseed", format="MSEED")

print("Файл clear_input.mseed успішно створено!")