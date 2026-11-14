import array
import io
import math
import struct
import sys
import wave
from dataclasses import dataclass

PCM16 = 'pcm16'
FLOAT32 = 'float32'
_FORMATS = {'h': PCM16, 'f': FLOAT32}
_FORMAT_ERROR = 'expected 16-bit PCM or 32-bit float WAV, 1 or 2 channels, 8000-48000 Hz'
_EXTENSIBLE = 0xFFFE
_PCM_GUID = bytes.fromhex('0100000000001000800000aa00389b71')
_FLOAT_GUID = bytes.fromhex('0300000000001000800000aa00389b71')


@dataclass
class Audio:
    rate: int
    channels: int
    samples: array.array

    @property
    def frames(self):
        return len(self.samples) // self.channels

    @property
    def sample_format(self):
        return _FORMATS[self.samples.typecode]


def saturate(value):
    return max(-32768, min(32767, round(value)))


def convert(audio, sample_format):
    """Re-express audio in another sample format using codec quantization."""
    if sample_format == audio.sample_format:
        return audio
    if sample_format == FLOAT32:
        return Audio(audio.rate, audio.channels,
                     array.array('f', (sample / 32768 for sample in audio.samples)))
    if sample_format == PCM16:
        return Audio(audio.rate, audio.channels,
                     array.array('h', (saturate(sample * 32768) for sample in audio.samples)))
    raise ValueError("sample_format must be 'pcm16' or 'float32'")


def _chunks(data):
    if len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('invalid WAV data')
    position = 12
    while position + 8 <= len(data):
        tag = data[position:position + 4]
        size = int.from_bytes(data[position + 4:position + 8], 'little')
        start = position + 8
        if len(data) - start < size:
            raise ValueError('invalid WAV data')
        yield tag, data[start:start + size]
        position = start + size + (size & 1)


def _format_tag(data):
    try:
        for tag, body in _chunks(data):
            if tag == b'fmt ':
                return int.from_bytes(body[:2], 'little') if len(body) >= 2 else None
    except ValueError:
        pass
    return None


def decode(data):
    if _format_tag(data) in (3, _EXTENSIBLE):
        return _decode_extended(data)
    return _decode_pcm(data)


def _decode_pcm(data):
    try:
        with wave.open(io.BytesIO(data), 'rb') as reader:
            channels, width, rate, frames = (reader.getnchannels(), reader.getsampwidth(),
                                             reader.getframerate(), reader.getnframes())
            if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 48000:
                raise ValueError(_FORMAT_ERROR)
            if frames < 1 or frames > rate * 30:
                raise ValueError('audio duration must be greater than zero and at most 30 seconds')
            raw = reader.readframes(frames)
            if len(raw) != frames * channels * width:
                raise ValueError('truncated audio data')
    except (wave.Error, EOFError) as exc:
        raise ValueError('invalid WAV data') from exc
    samples = array.array('h')
    samples.frombytes(raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    return Audio(rate, channels, samples)


def _decode_extended(data):
    fmt = None
    payload = None
    for tag, body in _chunks(data):
        if tag == b'fmt ' and fmt is None:
            fmt = body
        elif tag == b'data' and payload is None:
            payload = body
    if fmt is None or payload is None or len(fmt) < 16:
        raise ValueError('invalid WAV data')
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack('<HHIIHH', fmt[:16])
    if tag == _EXTENSIBLE:
        if len(fmt) < 40:
            raise ValueError('invalid WAV data')
        if fmt[24:40] == _PCM_GUID:
            tag = 1
        elif fmt[24:40] == _FLOAT_GUID:
            tag = 3
    if (tag, bits) == (1, 16):
        typecode, width = 'h', 2
    elif (tag, bits) == (3, 32):
        typecode, width = 'f', 4
    else:
        raise ValueError(_FORMAT_ERROR)
    if channels not in (1, 2) or not 8000 <= rate <= 48000:
        raise ValueError(_FORMAT_ERROR)
    frames = len(payload) // (channels * width)
    if frames < 1 or frames > rate * 30:
        raise ValueError('audio duration must be greater than zero and at most 30 seconds')
    samples = array.array(typecode)
    samples.frombytes(payload[:frames * channels * width])
    if sys.byteorder != 'little':
        samples.byteswap()
    if typecode == 'f' and not all(math.isfinite(sample) for sample in samples):
        raise ValueError('audio samples must be finite')
    return Audio(rate, channels, samples)


def encode(audio, sample_format=None):
    if sample_format is not None:
        audio = convert(audio, sample_format)
    if audio.samples.typecode == 'f':
        return _encode_float(audio)
    output = io.BytesIO()
    samples = array.array('h', audio.samples)
    if sys.byteorder != 'little':
        samples.byteswap()
    with wave.open(output, 'wb') as writer:
        writer.setnchannels(audio.channels)
        writer.setsampwidth(2)
        writer.setframerate(audio.rate)
        writer.writeframes(samples.tobytes())
    return output.getvalue()


def _encode_float(audio):
    samples = array.array('f', audio.samples)
    if sys.byteorder != 'little':
        samples.byteswap()
    payload = samples.tobytes()
    fmt = struct.pack('<HHIIHH', 3, audio.channels, audio.rate,
                      audio.rate * audio.channels * 4, audio.channels * 4, 32)
    size = 4 + (8 + len(fmt)) + (8 + 4) + (8 + len(payload))
    return b''.join((b'RIFF', struct.pack('<I', size), b'WAVE',
                     b'fmt ', struct.pack('<I', len(fmt)), fmt,
                     b'fact', struct.pack('<II', 4, audio.frames),
                     b'data', struct.pack('<I', len(payload)), payload))
