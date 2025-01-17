import numpy as np
import numba as nb
from numpy.fft import rfft, rfftfreq
from scipy.sparse import spdiags
from spectrum import dpss
from time import time


class Beamforming:

    def __init__(self):
        pass

    def delaysum(self, data):
        # Number of stations, number of time sampling points
        N_d, N_t = data.shape
        # Recursion depth
        N_rec = int(np.log2(N_d))

        # Buffers for the time shifts and waveform stacks
        shifts = np.zeros(N_d, dtype=int)
        stack = data.copy() / data.std()

        # Perform loop over N_rec recursion steps
        for n in range(N_rec):
            # Number of stacks
            N_i = stack.shape[0] // 2
            # Buffer for new stack
            new_stack = np.zeros((N_i, N_t))
            # Loop over station pairs
            for i in range(N_i):
                # Select waveform (stacks)
                x = stack[2 * i]
                y = stack[2 * i + 1]
                # Cross-correlate waveforms
                shift, R = self.xcorr(x, y)
                # Shift and stack
                new_stack[i] = x + np.roll(y, -shift)
                # Indices of cumulative shifts performd on a given waveform
                start = (2 * i + 1) * 2**n
                stop = (i + 1) * 2**(n + 1)
                # Increment cumulative shift
                shifts[start:stop] += shift
            # Replace old stack with new stack
            stack = new_stack

        # Remove unused dimensions
        stack = stack.ravel()
        # Compute norm of stack
        norm = np.linalg.norm(stack / stack.shape[0])

        # Return cumulative shifts, stacked waveform, norm
        return shifts, stack, norm

    def precompute_A(self, freqs):
        dt = self.dt
        Ns = dt.shape[-1]
        dt_r = dt.reshape((np.prod(dt.shape[:-1]), Ns))
        fdt = np.einsum("f,nk->fnk", freqs, dt_r, optimize=False)
        A = np.exp(-2j * np.pi * fdt)
        self.A = A
        return A

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
    def compute_Cxy_jit(Xf, weights, scale):
        tapers, N_stations, N_t = Xf.shape
        Cxy = np.zeros((N_stations, N_stations, N_t), dtype=nb.complex64)
        Xfc = Xf.conj()

        for i in nb.prange(N_stations):
            for j in nb.prange(i + 1):
                for k in range(tapers):
                    Cxy[i, j] += weights[k] * Xf[k, i] * Xfc[k, j]
                Cxy[i, j] = Cxy[i, j] * (scale[i] * scale[j])

        return Cxy

    def CMTM(self, X, Nw, freq_band=None, fsamp=None, scale=True, jit=False):

        # Number of tapers
        K = 2 * Nw
        # Number of stations (m), time sampling points (Nx)
        m, Nf = X.shape

        # Next power of 2 (for FFT)
        # NFFT = 2**int(np.log2(Nf))
        NFFT = 2**int(np.log2(Nf) + 1) + 1

        # Subtract mean (over time axis) for each station
        X_mean = np.mean(X, axis=1)
        X_mean = np.tile(X_mean, [Nf, 1]).T
        X = X - X_mean


        # Compute taper weight coefficients
        tapers, eigenvalues = dpss(N=Nf, NW=Nw, k=K)


        # Compute weights from eigenvalues
        weights = eigenvalues / (np.arange(K) + 1).astype(float)

        # Align tapers with X
        tapers = np.tile(tapers.T, [m, 1, 1])
        tapers = np.swapaxes(tapers, 0, 1)



        # Compute tapered FFT of X.0
        # Note that X is assumed to be real, so that the negative frequencies can be discarded
        #
        Xf = rfft(np.multiply(tapers, X), NFFT, axis=-1)

        # Multitaper power spectrum (not scaled by weights.sum()!)
        Pk = np.abs(Xf)**2
        Pxx = np.sum(Pk.T * weights, axis=-1).T
        inv_Px = 1 / np.sqrt(Pxx)

        inv_sum_weights = 1.0 / weights.sum()

        # If a specific frequency band is given
        if freq_band is not None:
            # Check if the sampling frequency is specified
            if fsamp is None:
                print("When a frequency band is selected, fsamp must be provided")
                return False
            # Compute the frequency range
            freqs = rfftfreq(n=NFFT, d=1.0 / fsamp)
            # Select the frequency band indices
            inds = (freqs >= freq_band[0]) & (freqs < freq_band[1])
            # print(freqs[inds])

            # Slice the vectors
            Xf = Xf[:, :, inds]

            inv_Px = inv_Px[:, inds]


        # Buffer for covariance matrix
        if jit:
            Ns = Xf.shape[1]
            if scale:
                # Vector for scaling
                scale_vec = inv_Px
                # Compute covariance matrix
                Cxy = self.compute_Cxy_jit(Xf, weights, scale_vec)
                # Make Cxy Hermitian
                Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
                # Add ones to diagonal
                for i in range(Ns):
                    Cxy[i, i] = 1
            else:
                # Vector for scaling
                scale_vec = np.sqrt(np.ones(Ns) * inv_sum_weights / Xf.shape[2])
                # Compute covariance matrix
                Cxy = self.compute_Cxy_jit(Xf, weights, scale_vec)
                # Make Cxy Hermitian
                Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
                # Correct diagonal
                for i in range(Ns):
                    Cxy[i, i] *= 0.5

        else:
            Cxy = np.zeros((m, m, Xf.shape[2]), dtype=complex)

            # Loop over all stations
            for i in range(m):
                # Do only lower triangle
                for j in range(i):
                    # Compute SUM[w_k . X_k . Y*_k] using Einstein notation
                    Pxy = np.einsum("k,kt,kt->t", weights, Xf[:, i], Xf[:, j].conj(), optimize=True)
                    # Store result in covariance matrix
                    if scale:
                        Cxy[i, j] = Pxy * (inv_Px[i] * inv_Px[j])
                    else:
                        Cxy[i, j] = Pxy * inv_sum_weights / Xf.shape[2]
                if not scale:
                    Cxy[i, i] = 0.5 * np.einsum("k,kt,kt->t", weights, Xf[:, i], Xf[:, i].conj(), optimize=True) * inv_sum_weights / Xf.shape[2]
            # Make Cxy Hermitian
            Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
            # Add ones to diagonal
            if scale:
                Cxy = Cxy + np.tile(np.eye(m), [Cxy.shape[2], 1, 1]).T
        return Cxy

    def noise_space_projection(self, Rxx, sources=3, mode="MUSIC"):
        # Number of source locations (Nx, Ny), number of stations (m)
        grid_size = self.grid["grid_size"]
        A = self.A
        Nf, _, m = A.shape #
        scale = 1.0 / (m * Nf)

        # Total projection onto noise space
        Pm = np.zeros(np.prod(grid_size), dtype=complex)

        # Loop over frequencies
        for f in range(Nf):
            # Select steering vectors for frequency f
            Af = A[f]

            # Traditional beamforming: maximise projection of steering vector onto covariance matrix
            if mode == "beam":
                # Cast to complex, which dramatically reduces overhead. No clue why, because Rxx is already complex...
                Un = Rxx[:, :, f].astype(complex)
                # Project steering vector onto subspace
                Pm += np.einsum("sn, nk, sk->s", Af.conj(), Un, Af, optimize=True)
            # MUSIC: minimise projection of steering vector onto noise space
            elif mode == "MUSIC":
                # Compute eigenvalues/vectors assuming Rxx is complex Hermitian (conjugate symmetric)
                # Eigenvalues appear in ascending order
                l, v = np.linalg.eigh(Rxx[:, :, f]) # 特征值升序排列
                M = sources
                # Extract noise space (size n-M)
                # NOTE: in original code, un was labelled "signal space"!
                un = v[:, :m - M]
                # Precompute un.un*
                Un = np.dot(un, un.conj().T)
                # Project steering vector onto subspace
                Pm += np.einsum("sn, nk, sk->s", Af.conj(), Un, Af, optimize=True)

            else:
                print("Mode '%s' not recognised. Aborting...")
                return


        return np.real(Pm) * scale

    def SAMV3(self, Rxx, p_prev, sigma_prev):

        # Number of source locations (Nx, Ny), number of stations (m)
        Nx, Ny = self.grid["grid_size"]
        Nz = Nx * Ny
        # Maximum number of iterations
        Nit = self.Nit
        A = self.A
        Nf, _, m = A.shape
        """
        A = (Nf, Nz, m)? --> Check
        """
        scale = 1.0 / (m * Nf)

        # Total projection onto noise space
        Pm = np.zeros(Nz, dtype=complex)

        p_vec_prev = p_prev.copy()
        sigma = sigma_prev.copy()

        for f in range(Nf):
            Af = A[f].T
            print(Af.shape)
            for i in range(Nit):
                """
                Af = (m, Nz)
                P = (Nz, Nz)
                R, R_inv = (m, m)
                R_inv_A = (m, m) x (m, Nz) = (m, Nz)
                A_Rinv_A = (Nz, m) x (m, Nz) = (Nz, Nz)
                enum = (Nz, m) x (m, m) x (m, Nz) = (Nz, Nz)

                - Convergence check
                - Initialisation of p, sigma
                """
                R = np.einsum("mi,ii,ni->mn", Af, p_vec_prev, Af.conj(), optimize=True) + np.diagflat(sigma)
                R_inv = np.linalg.inv(R)
                A_Rinv_A = np.einsum("mk,mn,nk->k", Af.conj(), R_inv, Af, optimize=True)
                A_RRR_A = np.einsum("mk,mn,no,op,pk->k", Af.conj(), R_inv, Rxx, R_inv, Af, optimize=True)

                p_vec = p_vec_prev * (A_RRR_A / A_Rinv_A)
                R_inv_sq = R_inv.dot(R_inv)
                sigma = np.real(np.trace(R_inv_sq.dot(Rxx))) / np.real(np.trace(R_inv_sq))

                p_vec_prev = p_vec.copy()

            Pm += p_vec

        return Pm, sigma

    def do_backprojection(self, data, freq_band, sources=3, win=10., stride=1.,
                          mode="MUSIC", duration=None, jit=True):

        print("Start back-projection")

        Ns, Nt = data.shape
        grid_size = self.grid["grid_size"]
        fsamp = self.fsamp

        if duration is None:
            duration = Nt / fsamp # 采样时间

        win = int(win * fsamp) # 每个窗口的点数
        stride = int(stride * fsamp)

        Nwin = int(min(duration * fsamp / stride, (Nt - win) / stride)) # 步数

        P = np.zeros((Nwin, np.prod(grid_size))) # 储存P，分多个时间段

        NFFT = 2**int(np.log2(win) + 1) + 1
        freqs = rfftfreq(n=NFFT, d=1./fsamp)
        inds = (freqs >= freq_band[0]) & (freqs < freq_band[1])
        freqs_select = freqs[inds]

        self.precompute_A(freqs_select)
        t0 = time()

        for n in range(Nwin):
            running_time = time() - t0 + 1e-6
            win_per_sec = (n + 1) / running_time
            win_to_do = Nwin - n
            ETA = win_to_do / win_per_sec

            print("Processing %d / %d [ETA: %.0f s]" % (n + 1, Nwin, ETA))

            start = n * stride
            stop = start + win

            sub_data = data[:, start:stop]
            Rxx = self.CMTM(
                sub_data, Nw=2, freq_band=freq_band, fsamp=fsamp, jit=jit
            )
            Rxx[np.isnan(Rxx)] = 0
            P[n] = self.noise_space_projection(Rxx, sources=sources, mode=mode)

        P_shape = (Nwin,) + grid_size
        P = P.reshape(P_shape)
        return P
