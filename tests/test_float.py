import array
import base64
import hashlib
import io
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

from audio_workbench import processing, waveio
from audio_workbench.arrangement import compile_document
from audio_workbench.assets import AssetLibrary
from audio_workbench.catalog import Catalog
from audio_workbench.rendering import render_manifest
from audio_workbench.server import Handler
from audio_workbench.waveio import Audio, convert, decode, encode

PCM_GUID = bytes.fromhex('0100000000001000800000aa00389b71')
FLOAT_GUID = bytes.fromhex('0300000000001000800000aa00389b71')


def extensible_wav(guid, bits, payload, rate=8000, channels=1):
    align = channels * bits // 8
    fmt = struct.pack('<HHIIHH', 0xFFFE, channels, rate, rate * align, align, bits)
    fmt += struct.pack('<HHI', 22, bits, 0) + guid
    size = 4 + (8 + len(fmt)) + (8 + len(payload))
    return b''.join((b'RIFF', struct.pack('<I', size), b'WAVE',
                     b'fmt ', struct.pack('<I', len(fmt)), fmt,
                     b'data', struct.pack('<I', len(payload)), payload))


class FloatCodecTest(unittest.TestCase):
    def test_float_round_trip_keeps_over_range_values(self):
        audio = Audio(8000, 2, array.array('f', [0.5, -1.0, 1.5, -2.0, 0.0, 3.25]))
        data = encode(audio)
        self.assertEqual(data[:4], b'RIFF')
        self.assertEqual(data[20:22], b'\x03\x00')
        restored = decode(data)
        self.assertEqual(restored.sample_format, waveio.FLOAT32)
        self.assertEqual(restored.samples.typecode, 'f')
        self.assertEqual((restored.rate, restored.channels, restored.frames), (8000, 2, 3))
        self.assertEqual(list(restored.samples), [0.5, -1.0, 1.5, -2.0, 0.0, 3.25])

    def test_pcm16_codec_is_unchanged(self):
        audio = Audio(8000, 1, array.array('h', [1000, -2000, 32767, -32768]))
        data = encode(audio)
        self.assertEqual(len(data), 44 + 8)
        self.assertEqual(data[20:22], b'\x01\x00')
        with wave.open(io.BytesIO(data), 'rb') as reader:
            self.assertEqual((reader.getsampwidth(), reader.getnframes()), (2, 4))
        restored = decode(data)
        self.assertEqual(restored.sample_format, waveio.PCM16)
        self.assertEqual(list(restored.samples), [1000, -2000, 32767, -32768])

    def test_extensible_float_and_pcm_decode(self):
        floats = array.array('f', [0.25, -2.5])
        restored = decode(extensible_wav(FLOAT_GUID, 32, floats.tobytes()))
        self.assertEqual(restored.sample_format, waveio.FLOAT32)
        self.assertEqual(list(restored.samples), [0.25, -2.5])
        ints = array.array('h', [100, -200])
        restored = decode(extensible_wav(PCM_GUID, 16, ints.tobytes()))
        self.assertEqual(restored.sample_format, waveio.PCM16)
        self.assertEqual(list(restored.samples), [100, -200])

    def test_decode_rejects_unsupported_formats_and_nonfinite_floats(self):
        with self.assertRaises(ValueError):
            decode(extensible_wav(FLOAT_GUID, 64, b'\x00' * 16))
        with self.assertRaises(ValueError):
            decode(extensible_wav(bytes(16), 32, b'\x00' * 8))
        for samples in ([float('nan'), 0.0], [float('inf')], [float('-inf')]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                decode(encode(Audio(8000, 1, array.array('f', samples))))

    def test_convert_between_formats(self):
        pcm = Audio(8000, 1, array.array('h', [32767, -32768, 16384, -16384, 0]))
        floats = convert(pcm, waveio.FLOAT32)
        self.assertEqual(floats.sample_format, waveio.FLOAT32)
        self.assertEqual(list(floats.samples), [32767 / 32768, -1.0, 0.5, -0.5, 0.0])
        self.assertIs(convert(pcm, waveio.PCM16), pcm)
        self.assertIs(convert(floats, waveio.FLOAT32), floats)
        self.assertEqual(list(convert(floats, waveio.PCM16).samples), [32767, -32768, 16384, -16384, 0])
        over = Audio(8000, 1, array.array('f', [2.0, -2.5, 1.0, -1.0]))
        self.assertEqual(list(convert(over, waveio.PCM16).samples), [32767, -32768, 32767, -32768])
        with self.assertRaises(ValueError):
            convert(pcm, 'mp3')

    def test_encode_converts_on_request(self):
        over = Audio(8000, 1, array.array('f', [2.0, -2.0]))
        delivered = encode(over, waveio.PCM16)
        self.assertEqual(decode(delivered).sample_format, waveio.PCM16)
        self.assertEqual(list(decode(delivered).samples), [32767, -32768])
        pcm = Audio(8000, 1, array.array('h', [16384, -16384]))
        restored = decode(encode(pcm, waveio.FLOAT32))
        self.assertEqual(restored.sample_format, waveio.FLOAT32)
        self.assertEqual(list(restored.samples), [0.5, -0.5])


class FloatProcessingTest(unittest.TestCase):
    def test_clip_float_keeps_headroom_while_pcm16_saturates(self):
        floats = Audio(8000, 1, array.array('f', [0.75, -0.75] * 40))
        clipped = processing.clip(floats, 0, 0.005, 2)
        self.assertEqual(clipped.sample_format, waveio.FLOAT32)
        self.assertEqual(list(clipped.samples[:4]), [1.5, -1.5, 1.5, -1.5])
        pcm = Audio(8000, 1, array.array('h', [20000, -20000] * 40))
        legacy = processing.clip(pcm, 0, 0.005, 2)
        self.assertEqual(legacy.sample_format, waveio.PCM16)
        self.assertEqual(list(legacy.samples[:4]), [32767, -32768, 32767, -32768])

    def test_mix_float_cancels_inverted_signal(self):
        signal = Audio(8000, 1, array.array('f', [1.75, -1.75] * 40))
        inverted = Audio(8000, 1, array.array('f', [-1.75, 1.75] * 40))
        mixed = processing.mix([(signal, 0, 1), (inverted, 0, 1)])
        self.assertEqual(mixed.sample_format, waveio.FLOAT32)
        self.assertEqual(set(mixed.samples), {0.0})
        doubled = processing.mix([(signal, 0, 1), (signal, 0, 1)])
        self.assertEqual(list(doubled.samples[:2]), [3.5, -3.5])

    def test_mix_pcm16_matches_legacy_saturation(self):
        a = Audio(8000, 1, array.array('h', [20000, -20000, 100]))
        b = Audio(8000, 1, array.array('h', [20000, -20000, 100]))
        mixed = processing.mix([(a, 0, 1), (b, 0, 1)])
        self.assertEqual(mixed.sample_format, waveio.PCM16)
        self.assertEqual(list(mixed.samples), [32767, -32768, 200])

    def test_mix_promotes_mixed_representations_to_float(self):
        pcm = Audio(8000, 1, array.array('h', [16384, -8192]))
        floats = Audio(8000, 1, array.array('f', [0.25, -0.25]))
        mixed = processing.mix([(pcm, 0, 1), (floats, 0, 1)])
        self.assertEqual(mixed.sample_format, waveio.FLOAT32)
        self.assertEqual(list(mixed.samples), [0.75, -0.5])

    def test_conform_float_avoids_quantization(self):
        floats = Audio(8000, 1, array.array('f', [0.5, 1.5]))
        upsampled = processing.conform(floats, 16000, 1)
        self.assertEqual(upsampled.sample_format, waveio.FLOAT32)
        self.assertEqual(list(upsampled.samples), [0.5, 1.0, 1.5, 1.5])
        stereo = processing.conform(Audio(8000, 1, array.array('f', [2.0, -2.0])), 8000, 2)
        self.assertEqual(list(stereo.samples), [2.0, 2.0, -2.0, -2.0])

    def test_inspect_reports_sample_format_and_peak(self):
        report = processing.inspect(Audio(8000, 1, array.array('f', [-2.0, 0.5])))
        self.assertEqual(report['sample_format'], waveio.FLOAT32)
        self.assertEqual(report['peak'], 2.0)
        report = processing.inspect(Audio(8000, 1, array.array('h', [-32768, 100])))
        self.assertEqual((report['sample_format'], report['peak']), (waveio.PCM16, 1.0))


class FloatHTTPTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(3)

    def call(self, path, body):
        request = urllib.request.Request(f'http://127.0.0.1:{self.server.server_port}' + path,
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.load(exc)

    def encoded(self, audio):
        return base64.b64encode(encode(audio)).decode()

    def test_clip_float_default_stays_float_and_pcm16_delivery_saturates(self):
        source = self.encoded(Audio(8000, 1, array.array('f', [0.75, -0.75] * 4000)))
        status, result = self.call('/clip', {'audio': source, 'start': 0, 'end': .5, 'gain': 2})
        self.assertEqual(status, 200)
        self.assertEqual(result['properties']['sample_format'], 'float32')
        self.assertEqual(result['properties']['peak'], 1.5)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.FLOAT32)
        self.assertEqual(list(delivered.samples[:4]), [1.5, -1.5, 1.5, -1.5])
        # Re-reading and inspecting the product confirms format and peak.
        status, again = self.call('/inspect', {'audio': result['audio']})
        self.assertEqual((status, again), (200, result['properties']))
        status, result = self.call('/clip', {'audio': source, 'start': 0, 'end': .5,
                                             'gain': 2, 'sample_format': 'pcm16'})
        self.assertEqual(status, 200)
        self.assertEqual(result['properties']['sample_format'], 'pcm16')
        self.assertEqual(result['properties']['peak'], 1.0)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.PCM16)
        self.assertEqual(list(delivered.samples[:4]), [32767, -32768, 32767, -32768])

    def test_clip_pcm16_legacy_results_are_sample_exact(self):
        pattern = [10000, -20000, 30000, -30000]
        source = self.encoded(Audio(8000, 1, array.array('h', pattern * 100)))
        status, result = self.call('/clip', {'audio': source, 'start': .01, 'end': .02, 'gain': 1.5})
        self.assertEqual(status, 200)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.PCM16)
        self.assertEqual(list(delivered.samples), [processing.saturate(s * 1.5) for s in pattern * 20])
        status, result = self.call('/clip', {'audio': source, 'start': .01, 'end': .02,
                                             'gain': 1.5, 'sample_format': 'float32'})
        self.assertEqual(status, 200)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.FLOAT32)
        self.assertEqual(list(delivered.samples[:4]),
                         [15000 / 32768, -30000 / 32768, 45000 / 32768, -45000 / 32768])

    def test_mix_float_inverse_cancels_and_pcm16_stays_legacy(self):
        signal = self.encoded(Audio(8000, 1, array.array('f', [1.5, -1.5] * 100)))
        inverse = self.encoded(Audio(8000, 1, array.array('f', [-1.5, 1.5] * 100)))
        status, result = self.call('/mix', {'tracks': [{'audio': signal}, {'audio': inverse}]})
        self.assertEqual(status, 200)
        self.assertEqual(result['properties']['sample_format'], 'float32')
        self.assertEqual(result['properties']['peak'], 0.0)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.FLOAT32)
        self.assertEqual(set(delivered.samples), {0.0})
        pcm = self.encoded(Audio(8000, 1, array.array('h', [20000] * 100)))
        status, result = self.call('/mix', {'tracks': [{'audio': pcm}, {'audio': pcm}]})
        self.assertEqual(status, 200)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.PCM16)
        self.assertEqual(list(delivered.samples[:4]), [32767] * 4)
        status, result = self.call('/mix', {'tracks': [{'audio': pcm}, {'audio': pcm}],
                                            'sample_format': 'float32'})
        self.assertEqual(status, 200)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.FLOAT32)
        self.assertEqual(list(delivered.samples[:4]), [40000 / 32768] * 4)

    def test_mix_mixed_representations_confirmed_by_rereading(self):
        pcm = self.encoded(Audio(8000, 1, array.array('h', [16384] * 100)))
        floats = self.encoded(Audio(8000, 1, array.array('f', [1.0] * 100)))
        status, result = self.call('/mix', {'tracks': [{'audio': pcm}, {'audio': floats}]})
        self.assertEqual(status, 200)
        self.assertEqual(result['properties']['sample_format'], 'float32')
        self.assertEqual(result['properties']['peak'], 1.5)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.FLOAT32)
        self.assertEqual(list(delivered.samples[:2]), [1.5, 1.5])
        status, again = self.call('/inspect', {'audio': result['audio']})
        self.assertEqual((status, again), (200, result['properties']))
        status, result = self.call('/mix', {'tracks': [{'audio': pcm}, {'audio': floats}],
                                            'sample_format': 'pcm16'})
        self.assertEqual(status, 200)
        delivered = decode(base64.b64decode(result['audio']))
        self.assertEqual(delivered.sample_format, waveio.PCM16)
        self.assertEqual(list(delivered.samples[:2]), [32767, 32767])

    def test_sample_format_validation_and_float_inspect(self):
        pcm = self.encoded(Audio(8000, 1, array.array('h', [100] * 10)))
        self.assertEqual(self.call('/clip', {'audio': pcm, 'start': 0, 'end': .001,
                                             'sample_format': 'mp3'})[0], 400)
        self.assertEqual(self.call('/mix', {'tracks': [{'audio': pcm}], 'sample_format': 16})[0], 400)
        floats = self.encoded(Audio(8000, 1, array.array('f', [2.0] * 10)))
        status, report = self.call('/inspect', {'audio': floats})
        self.assertEqual(status, 200)
        self.assertEqual((report['sample_format'], report['peak']), ('float32', 2.0))
        self.assertEqual(self.call('/inspect', {'audio': pcm})[1]['sample_format'], 'pcm16')


class FloatRenderGuardTest(unittest.TestCase):
    def test_assets_stay_pcm16(self):
        with tempfile.TemporaryDirectory() as root:
            assets = AssetLibrary(Catalog(root))
            float_wav = encode(Audio(8000, 1, array.array('f', [0.5, -0.5])))
            with self.assertRaisesRegex(ValueError, '16-bit'):
                assets.import_audio('float', float_wav)

    def test_pcm_v1_render_rejects_float_sources(self):
        float_wav = encode(Audio(8000, 1, array.array('f', [0.5, -0.5])))
        metadata = {'sample_rate': 8000, 'channels': 1, 'frames': 2,
                    'sha256': hashlib.sha256(float_wav).hexdigest()}
        manifest = compile_document(
            {'schema_version': 1, 'format': {'sample_rate': 8000, 'channels': 1},
             'clips': [{'id': 'x', 'asset_id': 'a'}]}, lambda asset_id: metadata)
        with self.assertRaisesRegex(ValueError, '16-bit'):
            render_manifest(manifest, lambda asset_id, sha256: float_wav)


if __name__ == '__main__':
    unittest.main()
