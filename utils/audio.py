import numpy as np
import scipy.io
import scipy.signal
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


class AudioProcessor(object):

    def __init__(self,
                 sample_rate=22050,
                 num_mels=80,
                 num_freq=1025,
                 frame_length_ms=50,
                 frame_shift_ms=12.5,
                 preemphasis=0.98,
                 min_level_db=-100,
                 ref_level_db=20,
                 power=1.5,
                 mel_fmin=0.0,
                 mel_fmax=8000.0,
                 griffin_lim_iters=60):
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.preemphasis = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.power = power
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.griffin_lim_iters = griffin_lim_iters

        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()

        print('AudioProcessor')
        for key, value in vars(self).items():
            print('  {}:{}'.format(key, value))

    def load_wav(self, filename, sr=None):
        if sr is None:
            y, sr = sf.read(filename)
        else:
            y, sr = librosa.load(filename, sr=sr)
        assert self.sample_rate == sr, \
            'WARNING: sample_rate mismatch: %d <=> %d' % (self.sample_rate, sr)
        return y

    def save_wav(self, y, path):
        wav_norm = y * (32767 / max(0.01, np.max(np.abs(y))))
        scipy.io.wavfile.write(path, self.sample_rate,
                               wav_norm.astype(np.int16))

    def apply_preemphasis(self, y):
        if self.preemphasis == 0:
            raise RuntimeError(
                'ERROR: Preemphasis is applied with factor 0.0!')
        return scipy.signal.lfilter([1, -self.preemphasis], [1], y)

    def apply_inv_preemphasis(self, y):
        if self.preemphasis == 0:
            raise RuntimeError(
                'ERROR: Preemphasis is applied with factor 0.0!')
        return scipy.signal.lfilter([1], [1, -self.preemphasis], y)

    def spectrogram(self, y):
        D = self._stft(self.apply_preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)
        y = self._griffin_lim(S ** self.power)
        return self.apply_inv_preemphasis(y)

    def melspectrogram(self, y):
        D = self._stft(self.apply_preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def inv_melspectrogram(self, melspectrogram):
        D = self._denormalize(melspectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)
        y = self._griffin_lim(S ** self.power)
        return self.apply_inv_preemphasis(y)

    def _stft_parameters(self):
        n_fft = (self.num_freq - 1) * 2
        factor = self.frame_length_ms / self.frame_shift_ms
        assert factor.is_integer(), \
            "WARNING: frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return n_fft, hop_length, win_length

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)

    def _istft(self, stft):
        return librosa.istft(
            stft,
            hop_length=self.hop_length,
            win_length=self.win_length)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _build_mel_basis(self):
        n_fft = (self.num_freq - 1) * 2
        return librosa.filters.mel(
            self.sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, melspec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, melspec))

    def _normalize(self, S, max_norm=4):
        S_norm = ((S - self.min_level_db) / -self.min_level_db)
        S_norm = ((2 * max_norm) * S_norm) - max_norm
        S_norm = np.clip(S_norm, -max_norm, max_norm)
        return S_norm

    def _denormalize(self, S, max_norm=4):
        S_denorm = np.clip(S, -max_norm, max_norm)
        S_denorm = ((S_denorm + max_norm) * -self.min_level_db /
                    (2 * max_norm)) + self.min_level_db
        return S_denorm

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y


def plot_spectrogram(linear_output, ap, figsize=(8, 5)):
    spectrogram = ap._denormalize(linear_output)
    fig = plt.figure(figsize=figsize)
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    return fig
