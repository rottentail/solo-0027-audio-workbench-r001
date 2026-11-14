import array
import math
from .waveio import Audio, saturate


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number between {low} and {high}')
    return value


def inspect(audio):
    if audio.samples.typecode == 'f':
        peak = max((abs(sample) for sample in audio.samples), default=0.0)
    else:
        peak = max(abs(sample) for sample in audio.samples) / 32768
    return {'sample_rate': audio.rate, 'channels': audio.channels, 'frames': audio.frames,
            'duration_seconds': audio.frames / audio.rate,
            'sample_format': audio.sample_format, 'peak': peak}


def clip(audio, start, end, gain):
    start = number(start, 'start', 0, 30)
    end = number(end, 'end', 0, 30)
    gain = number(gain, 'gain', 0, 4)
    first, last = round(start * audio.rate), round(end * audio.rate)
    if first >= last or last > audio.frames:
        raise ValueError('clip interval must contain frames within the audio')
    segment = audio.samples[first * audio.channels:last * audio.channels]
    if audio.samples.typecode == 'f':
        return Audio(audio.rate, audio.channels,
                     array.array('f', (sample * gain for sample in segment)))
    samples = (saturate(sample * gain) for sample in segment)
    return Audio(audio.rate, audio.channels, array.array('h', samples))


def conform(audio, rate, channels):
    if type(rate) is not int or not 8000 <= rate <= 48000 or rate < audio.rate:
        raise ValueError('sample rate must be an integer from the source rate through 48000')
    if type(channels) is not int or channels not in (1, 2):
        raise ValueError('channels must be 1 or 2')
    frames = round(audio.frames * rate / audio.rate)
    floats = audio.samples.typecode == 'f'
    samples = array.array('f' if floats else 'h')
    for frame in range(frames):
        position = frame * audio.rate / rate
        first = min(int(position), audio.frames - 1)
        second = min(first + 1, audio.frames - 1)
        fraction = position - first
        values = []
        for channel in range(audio.channels):
            left = audio.samples[first * audio.channels + channel]
            right = audio.samples[second * audio.channels + channel]
            values.append(left + (right - left) * fraction)
        if channels == 1:
            value = sum(values) / len(values)
            samples.append(value if floats else saturate(value))
        elif audio.channels == 1:
            samples.extend((values[0], values[0]) if floats else (saturate(values[0]), saturate(values[0])))
        else:
            samples.extend(values if floats else map(saturate, values))
    return Audio(rate, channels, samples)


def mix(tracks):
    reference = tracks[0][0]
    for audio, offset, gain in tracks:
        if (audio.rate, audio.channels) != (reference.rate, reference.channels):
            raise ValueError('track formats must match')
        number(offset, 'offset', 0, 30)
        number(gain, 'gain', 0, 4)
    length = max(round(offset * reference.rate) + audio.frames for audio, offset, _ in tracks)
    if length > reference.rate * 30:
        raise ValueError('mixed audio exceeds 30 seconds')
    floats = any(audio.samples.typecode == 'f' for audio, _, _ in tracks)
    sums = [0.0] * (length * reference.channels)
    for audio, offset, gain in tracks:
        position = round(offset * audio.rate) * audio.channels
        scale = gain / 32768 if floats and audio.samples.typecode == 'h' else gain
        for index, sample in enumerate(audio.samples):
            sums[position + index] += sample * scale
    if floats:
        return Audio(reference.rate, reference.channels, array.array('f', sums))
    return Audio(reference.rate, reference.channels,
                 array.array('h', (saturate(value) for value in sums)))
