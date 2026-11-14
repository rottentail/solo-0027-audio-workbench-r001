import base64
import binascii
from . import processing, waveio


def audio_from(value):
    if not isinstance(value, str):
        raise ValueError('audio must be base64 text')
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('invalid base64 audio') from exc
    return waveio.decode(raw)


def sample_format(value):
    if value is None:
        return None
    if value not in (waveio.PCM16, waveio.FLOAT32):
        raise ValueError("sample_format must be 'pcm16' or 'float32'")
    return value


def output(audio, delivery=None):
    if delivery is not None:
        audio = waveio.convert(audio, delivery)
    return {'audio': base64.b64encode(waveio.encode(audio)).decode('ascii'),
            'properties': processing.inspect(audio)}


def execute(path, body):
    if not isinstance(body, dict):
        raise ValueError('request must be a JSON object')
    if path == '/inspect':
        return processing.inspect(audio_from(body.get('audio')))
    if path == '/clip':
        delivery = sample_format(body.get('sample_format'))
        audio = audio_from(body.get('audio'))
        if delivery == waveio.FLOAT32:
            audio = waveio.convert(audio, waveio.FLOAT32)
        return output(processing.clip(audio, body.get('start'),
                                      body.get('end'), body.get('gain', 1)), delivery)
    if path == '/mix':
        tracks = body.get('tracks')
        if not isinstance(tracks, list) or not 1 <= len(tracks) <= 8:
            raise ValueError('tracks must contain between 1 and 8 items')
        if any(not isinstance(track, dict) for track in tracks):
            raise ValueError('each track must be an object')
        delivery = sample_format(body.get('sample_format'))
        decoded = [(audio_from(t.get('audio')), t.get('offset', 0), t.get('gain', 1)) for t in tracks]
        if delivery == waveio.FLOAT32:
            decoded = [(waveio.convert(audio, waveio.FLOAT32), offset, gain)
                       for audio, offset, gain in decoded]
        target = body.get('format')
        if target is not None:
            if not isinstance(target, dict):
                raise ValueError('format must be an object')
            decoded = [(processing.conform(audio, target.get('sample_rate'), target.get('channels')), offset, gain)
                       for audio, offset, gain in decoded]
        return output(processing.mix(decoded), delivery)
    raise LookupError('unknown endpoint')
