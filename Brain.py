from __future__ import annotations

import os
import sys
import json
import time
import math
import queue
import hashlib
import re
import sqlite3
import threading
import traceback
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any, Callable

import numpy as np

# Optional compatibility imports 
HybridQubit = None
QuantumEmulator = None
try:
    from Qubit import HybridQubit
except Exception:
    pass
try:
    from Cubit import QuantumEmulator
except Exception:
    pass
try:
    import requests
except Exception:
    requests = None
_QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    _QISKIT_AVAILABLE = True
except Exception:
    try:
        from qiskit import QuantumCircuit
        from qiskit.providers.aer import AerSimulator
        _QISKIT_AVAILABLE = True
    except Exception:
        QuantumCircuit = None
        AerSimulator = None
        _QISKIT_AVAILABLE = False

_QISKIT_RUNTIME_AVAILABLE = False
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    _QISKIT_RUNTIME_AVAILABLE = True
except Exception:
    QiskitRuntimeService = None
    Sampler = None
    _QISKIT_RUNTIME_AVAILABLE = False
_TK_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
    _TK_AVAILABLE = True
except Exception:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    scrolledtext = None
    simpledialog = None
    _TK_AVAILABLE = False

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    x = logits.astype(np.float64) / t
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if not np.isfinite(s) or s <= 0:
        out = np.ones_like(ex) / float(len(ex))
        return out.astype(np.float32)
    return (ex / s).astype(np.float32)


def _hash_to_unit_interval(data: bytes) -> float:
    h = hashlib.sha256(data).digest()
    x = int.from_bytes(h[:8], "little", signed=False)
    return (x % 10_000_000) / 9_999_999.0


def _hash_to_phase(data: bytes) -> float:
    h = hashlib.sha256(data).digest()
    x = int.from_bytes(h[8:16], "little", signed=False)
    return (x % 10_000_000) / 9_999_999.0 * (2.0 * math.pi) - math.pi


def _sm_safe_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    s = s.replace("\x00", " ")
    try:
        s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        s = "".join(ch if ord(ch) < 128 else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# This will be replaced in future updates.....
class ByteTokenizer:
    START = 256
    END = 257
    PAD = 258
    SEP = 259
    VOCAB_SIZE = 260

    def encode(self, text: str, add_start_end: bool = True) -> List[int]:
        b = (text or "").encode("utf-8", errors="replace")
        ids = list(b)
        return ([self.START] + ids + [self.END]) if add_start_end else ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for i in ids:
            ii = int(i)
            if 0 <= ii <= 255:
                out.append(ii)
        return out.decode("utf-8", errors="replace")



class QuantumBackendConfig:
    mode: str = "aer"  
    ibm_backend_name: str = ""
    ibm_instance: str = ""
    ibm_token: str = ""
    shots: int = 2048

class BrainConfig:
    # model
    embed_dim: int = 64
    context_len: int = 128
    temperature: float = 0.9
    top_k: int = 80
    max_reply_bytes: int = 420

    # learning
    continual_learning: bool = True
    spsa_steps_per_turn: int = 1
    spsa_a: float = 0.05
    spsa_c: float = 0.02
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    grad_clip: float = 5.0
    train_positions_per_step: int = 128

    # fast learning (parameter-efficient)
    adapter_rank: int = 8
    replay_buffer_size: int = 256
    replay_batch: int = 2
    consolidate_every_turns: int = 0  # 0 disables slow consolidation of base weights
    slow_spsa_steps: int = 1
    slow_lr_scale: float = 0.15
    amp_sharpen_iters: int = 0  
    qmem_channels: int = 8
    qmem_dimension: int = 8
    qmem_noise: float = 0.00
    vqc_enabled: bool = False
    vqc_qubits: int = 3
    vqc_layers: int = 2
    vqc_scale: float = 0.20
    internet_enabled: bool = True
    auto_web: bool = False
    semantic_memory: bool = True
    semantic_topk: int = 3
    semantic_store_turns: bool = True
    autosave_every_turns: int = 2
    qbackend: QuantumBackendConfig = field(default_factory=QuantumBackendConfig)

class SubspaceDefinitionError(Exception):
    pass


class SubBitMemoryChannel:

    def __init__(self, dimension: int = 8, logical_zero_idx: int = 0, logical_one_idx: int = 1):
        self.dimension = int(dimension)
        if self.dimension <= 2 or (self.dimension & (self.dimension - 1)) != 0:
            raise SubspaceDefinitionError("dimension must be a power-of-two and > 2")
        self.logical_zero_idx = int(logical_zero_idx)
        self.logical_one_idx = int(logical_one_idx)
        if self.logical_zero_idx == self.logical_one_idx:
            raise SubspaceDefinitionError("logical indices must differ")
        if not (0 <= self.logical_zero_idx < self.dimension and 0 <= self.logical_one_idx < self.dimension):
            raise SubspaceDefinitionError("logical indices out of range")

        self.sv = np.zeros((self.dimension,), dtype=np.complex64)
        self.sv[self.logical_zero_idx] = 1.0 + 0.0j
        self._normalize()

    def _normalize(self) -> None:
        n = float(np.linalg.norm(self.sv))
        if n <= 1e-12:
            self.sv[:] = 0
            self.sv[self.logical_zero_idx] = 1.0 + 0.0j
            return
        self.sv = (self.sv / n).astype(np.complex64)

    def _hidden_indices(self) -> List[int]:
        return [i for i in range(self.dimension) if i not in (self.logical_zero_idx, self.logical_one_idx)]

    def store_amplitude_in_subspace(self, target_hidden_idx: int) -> None:
        t = int(target_hidden_idx)
        if t in (self.logical_zero_idx, self.logical_one_idx) or not (0 <= t < self.dimension):
            raise SubspaceDefinitionError("target must be a hidden index")
        amp1 = self.sv[self.logical_one_idx]
        self.sv[self.logical_one_idx] = 0.0 + 0.0j
        self.sv[t] = self.sv[t] + amp1
        self._normalize()

    def apply_noise_to_hidden(self, noise_level: float) -> None:
        nl = float(noise_level)
        if nl <= 0:
            return
        hidden = self._hidden_indices()
        if not hidden:
            return
        noise = (np.random.normal(0, nl, size=(len(hidden),)) + 1j * np.random.normal(0, nl, size=(len(hidden),))).astype(
            np.complex64
        )
        for idx, n in zip(hidden, noise):
            self.sv[idx] += n
        self._normalize()

    def inject_signal(self, strength: float, phase: float, target_hidden_idx: int) -> None:

        s = float(np.clip(strength, 0.0, 1.0))
        phi = float(phase)
        alpha = math.sqrt(max(1.0 - s, 0.0))
        beta_mag = math.sqrt(max(s, 0.0))
        beta = np.complex64(beta_mag * (math.cos(phi) + 1j * math.sin(phi)))
        self.sv[self.logical_zero_idx] += np.complex64(alpha * 0.15)
        self.sv[self.logical_one_idx] += np.complex64(beta * 0.15)
        self._normalize()
        self.store_amplitude_in_subspace(target_hidden_idx)

    def features(self) -> np.ndarray:

        p0 = float(np.abs(self.sv[self.logical_zero_idx]) ** 2)
        p1 = float(np.abs(self.sv[self.logical_one_idx]) ** 2)
        hid = self._hidden_indices()
        hs = float(np.sum(np.abs(self.sv[hid]) ** 2)) if hid else 0.0
        return np.array([p0, p1, hs], dtype=np.float32)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "logical_zero_idx": self.logical_zero_idx,
            "logical_one_idx": self.logical_one_idx,
            "sv_re": self.sv.real.astype(np.float32).tolist(),
            "sv_im": self.sv.imag.astype(np.float32).tolist(),
        }

    def from_jsonable(d: Dict[str, Any]) -> "SubBitMemoryChannel":
        ch = SubBitMemoryChannel(
            dimension=int(d["dimension"]),
            logical_zero_idx=int(d.get("logical_zero_idx", 0)),
            logical_one_idx=int(d.get("logical_one_idx", 1)),
        )
        re_ = np.array(d["sv_re"], dtype=np.float32)
        im_ = np.array(d["sv_im"], dtype=np.float32)
        sv = (re_ + 1j * im_).astype(np.complex64)
        if sv.shape[0] != ch.dimension:
            raise ValueError("Saved quantum memory statevector size mismatch")
        ch.sv = sv
        ch._normalize()
        return ch


class QuantumBackendManager:

    def __init__(self, cfg: QuantumBackendConfig):
        self.cfg = cfg
        self._service = None
        self._backend_name = ""
        self._sampler = None
        self._lock = threading.RLock()
        self._aer = AerSimulator(method="statevector") if _QISKIT_AVAILABLE and AerSimulator else None

    def is_ibm_ready(self) -> bool:
        return _QISKIT_RUNTIME_AVAILABLE and (self._service is not None) and bool(self._backend_name)

    def connect_ibm(self, token: str, instance: str = "") -> List[str]:
        if not _QISKIT_RUNTIME_AVAILABLE:
            raise RuntimeError("qiskit_ibm_runtime is not installed")
        token = (token or "").strip() or os.environ.get("QISKIT_IBM_TOKEN", "").strip()
        if not token:
            raise RuntimeError("No IBM token provided (and QISKIT_IBM_TOKEN not set)")
        kwargs: Dict[str, Any] = {}
        if instance.strip():
            kwargs["instance"] = instance.strip()
        try:
            QiskitRuntimeService.save_account(token=token, overwrite=True, **kwargs)
        except Exception:
            pass
        service = QiskitRuntimeService(token=token, **kwargs)
        backends = service.backends()
        names = sorted({b.name for b in backends})
        with self._lock:
            self._service = service
        return names

    def select_ibm_backend(self, backend_name: str) -> None:
        if not _QISKIT_RUNTIME_AVAILABLE:
            raise RuntimeError("IBM runtime unavailable")
        with self._lock:
            if self._service is None:
                raise RuntimeError("IBM service not connected")
            self._backend_name = backend_name.strip()
            self._sampler = Sampler(backend=self._service.backend(self._backend_name))

    def measure_populations(self, sv: np.ndarray, shots: int = 2048) -> np.ndarray:
        if not _QISKIT_AVAILABLE or QuantumCircuit is None:
            p = np.abs(sv.astype(np.complex128)) ** 2
            p = p / max(float(np.sum(p)), 1e-12)
            return p.astype(np.float32)

        n = int(round(math.log2(len(sv))))
        if 2**n != len(sv):
            raise ValueError("Statevector length must be power-of-two")
        qc = QuantumCircuit(n, n)
        qc.initialize(sv.astype(np.complex128), list(range(n)))
        qc.measure(list(range(n)), list(range(n)))

        if self.cfg.mode == "cubit" and QuantumEmulator is not None:
            try:
                em = QuantumEmulator(num_qubits=n)
                rho = np.outer(sv.astype(np.complex128), np.conjugate(sv.astype(np.complex128)))
                em.state = rho
                p = np.real(np.diag(em.state)).astype(np.float32)
                s = float(np.sum(p))
                if s > 1e-12:
                    p /= s
                return p
            except Exception:
                pass

        if self.cfg.mode == "ibm" and self.is_ibm_ready():
            with self._lock:
                sampler = self._sampler
            if sampler is None:
                raise RuntimeError("IBM sampler not initialized")
            job = sampler.run([qc], shots=int(shots))
            result = job.result()
            quasi = result.quasi_dists[0]
            p = np.zeros((len(sv),), dtype=np.float32)
            for k, v in quasi.items():
                p[int(k)] = float(v)
            s = float(np.sum(p))
            if s > 1e-12:
                p /= s
            return p

        # Aer fallback
        if self._aer is None:
            p = np.abs(sv.astype(np.complex128)) ** 2
            p = p / max(float(np.sum(p)), 1e-12)
            return p.astype(np.float32)

        try:
            job = self._aer.run(qc, shots=int(shots))
            counts = job.result().get_counts()
            p = np.zeros((len(sv),), dtype=np.float32)
            for bitstr, c in counts.items():
                idx = int(str(bitstr).replace(" ", "")[::-1], 2)
                p[idx] = float(c) / float(shots)
            s = float(np.sum(p))
            if s > 1e-12:
                p /= s
            return p
        except Exception:
            p = np.abs(sv.astype(np.complex128)) ** 2
            p = p / max(float(np.sum(p)), 1e-12)
            return p.astype(np.float32)

def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]], dtype=np.complex128)


def _kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _apply_1q(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    mats = [np.eye(2, dtype=np.complex128) for _ in range(n)]
    mats[q] = gate
    U = _kron_n(mats)
    return (U @ state).astype(np.complex128)


def _apply_cz(state: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    dim = 2**n
    st = state.copy().astype(np.complex128)
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 == 1 and b2 == 1:
            st[idx] *= -1.0
    return st


def _expect_z(state: np.ndarray, q: int, n: int) -> float:
    dim = 2**n
    p = np.abs(state) ** 2
    s = 0.0
    for idx in range(dim):
        b = (idx >> q) & 1
        s += (1.0 if b == 0 else -1.0) * float(p[idx])
    return float(s)


class TinyVQC:

    def __init__(self, n_qubits: int, n_layers: int, rng: np.random.Generator):
        self.n = int(n_qubits)
        self.layers = int(n_layers)
        if self.n < 1 or self.n > 4:
            raise ValueError("TinyVQC supports 1..4 qubits")
        if self.layers < 1:
            raise ValueError("TinyVQC layers must be >= 1")
        self.theta = (rng.normal(0, 0.1, size=(self.layers, self.n, 3))).astype(np.float32)

    def forward(self, in_angles: np.ndarray) -> np.ndarray:
        """in_angles shape: (n, 3) for (rx, ry, rz)."""
        if in_angles.shape != (self.n, 3):
            raise ValueError("in_angles must have shape (n_qubits, 3)")
        state = np.zeros((2**self.n,), dtype=np.complex128)
        state[0] = 1.0

        for q in range(self.n):
            state = _apply_1q(state, _rx(float(in_angles[q, 0])), q, self.n)
            state = _apply_1q(state, _ry(float(in_angles[q, 1])), q, self.n)
            state = _apply_1q(state, _rz(float(in_angles[q, 2])), q, self.n)

        for l in range(self.layers):
            for q in range(self.n):
                th = self.theta[l, q]
                state = _apply_1q(state, _rx(float(th[0])), q, self.n)
                state = _apply_1q(state, _ry(float(th[1])), q, self.n)
                state = _apply_1q(state, _rz(float(th[2])), q, self.n)

            for q in range(self.n):
                q2 = (q + 1) % self.n
                state = _apply_cz(state, q, q2, self.n)

        return np.array([_expect_z(state, q, self.n) for q in range(self.n)], dtype=np.float32)


class BrainModel:

    def __init__(self, cfg: BrainConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.tok = ByteTokenizer()
        self.vocab_size = self.tok.VOCAB_SIZE
        self.embed_dim = int(cfg.embed_dim)
        self.context_len = int(cfg.context_len)
        self._rng = rng or np.random.default_rng(1234)

        # Base parameters
        self.E = (self._rng.normal(0, 0.02, size=(self.vocab_size, self.embed_dim))).astype(np.float32)
        self.W = (self._rng.normal(0, 0.02, size=(self.embed_dim, self.vocab_size))).astype(np.float32)
        self.b = np.zeros((self.vocab_size,), dtype=np.float32)
        self._seed_base_bias()

        r = max(int(cfg.adapter_rank), 0)
        self.adapter_rank = r
        if r > 0:
            self.A_U = np.zeros((self.embed_dim, r), dtype=np.float32)
            self.A_V = (self._rng.normal(0, 0.02, size=(r, self.vocab_size))).astype(np.float32)
            self.A_b = np.zeros((self.vocab_size,), dtype=np.float32)
        else:
            self.A_U = np.zeros((self.embed_dim, 0), dtype=np.float32)
            self.A_V = np.zeros((0, self.vocab_size), dtype=np.float32)
            self.A_b = np.zeros((self.vocab_size,), dtype=np.float32)

        qfeat_dim = 3 * int(cfg.qmem_channels)
        self.Q = (self._rng.normal(0, 0.05, size=(qfeat_dim, self.embed_dim))).astype(np.float32)

        self.qmem: List[SubBitMemoryChannel] = [SubBitMemoryChannel(dimension=cfg.qmem_dimension) for _ in range(int(cfg.qmem_channels))]

        self.vqc: Optional[TinyVQC] = None
        if bool(cfg.vqc_enabled):
            n = int(cfg.vqc_qubits)
            L = int(cfg.vqc_layers)
            self.vqc = TinyVQC(n_qubits=n, n_layers=L, rng=self._rng)
            self.vqc_in = (self._rng.normal(0, 0.08, size=(self.embed_dim, n * 3))).astype(np.float32)
            self.vqc_out = (self._rng.normal(0, 0.08, size=(n, self.embed_dim))).astype(np.float32)
        else:
            self.vqc_in = np.zeros((self.embed_dim, 0), dtype=np.float32)
            self.vqc_out = np.zeros((0, self.embed_dim), dtype=np.float32)

        self._replay: List[List[int]] = []
        self._train_step_fast = 0
        self._train_step_slow = 0

    def _seed_base_bias(self) -> None:
        base = np.ones((self.vocab_size,), dtype=np.float32) * 1e-6

        def p(ch: str, val: float):
            if not ch:
                return
            o = ord(ch)
            if 0 <= o < 256:
                base[o] = float(val)

        p(" ", 0.18)
        common = [
            ("e", 0.102), ("t", 0.075), ("a", 0.065), ("o", 0.062), ("n", 0.057),
            ("i", 0.056), ("s", 0.053), ("r", 0.050), ("h", 0.049), ("l", 0.033),
            ("d", 0.033), ("u", 0.022), ("c", 0.022), ("m", 0.020), ("f", 0.018),
            ("w", 0.018), ("g", 0.016), ("y", 0.016), ("p", 0.015), ("b", 0.012),
            ("v", 0.008), ("k", 0.006), ("x", 0.0015), ("j", 0.0010), ("q", 0.0010), ("z", 0.0007),
        ]
        for ch, val in common:
            p(ch, val)
            p(ch.upper(), val * 0.35)

        for ch, val in [
            (".", 0.012), (",", 0.012), ("\n", 0.010), ("?", 0.004), ("!", 0.004),
            (":", 0.003), (";", 0.003), ("-", 0.003), ("(", 0.002), (")", 0.002),
            ("'", 0.003), ('"', 0.003)
        ]:
            p(ch, val)

        base_sum = float(np.sum(base[:256]))
        if base_sum > 0:
            base[:256] /= base_sum
        self.b[:256] = np.log(np.maximum(base[:256], 1e-9)).astype(np.float32)


    def _context_embedding(self, ids: List[int]) -> np.ndarray:
        if not ids:
            return np.zeros((self.embed_dim,), dtype=np.float32)
        ids2 = ids[-self.context_len :]
        emb = self.E[np.array(ids2, dtype=np.int32)]
        return np.mean(emb, axis=0).astype(np.float32)

    def _quantum_memory_vector(self) -> np.ndarray:
        if not self.qmem:
            return np.zeros((self.embed_dim,), dtype=np.float32)
        feats = np.concatenate([ch.features() for ch in self.qmem], axis=0).astype(np.float32)  
        return (feats @ self.Q).astype(np.float32)

    def _vqc_vector(self, h: np.ndarray) -> np.ndarray:
        if self.vqc is None:
            return np.zeros((self.embed_dim,), dtype=np.float32)
        n = self.vqc.n
        angles = (h @ self.vqc_in).astype(np.float32)
        angles = angles.reshape((n, 3))
        z = self.vqc.forward(angles)  # (n,)
        v = (z @ self.vqc_out).astype(np.float32)
        return v * float(self.cfg.vqc_scale)

    def next_token_probs(self, context_ids: List[int], temperature: Optional[float] = None) -> np.ndarray:
        h = self._context_embedding(context_ids)
        q = self._quantum_memory_vector()
        v = self._vqc_vector(h)
        h2 = np.tanh(h + q + v).astype(np.float32)

        dW = (self.A_U @ self.A_V).astype(np.float32) if self.adapter_rank > 0 else 0.0
        logits = (h2 @ (self.W + dW) + (self.b + self.A_b)).astype(np.float32)
        return _stable_softmax(logits, temperature if temperature is not None else self.cfg.temperature)

    def _sanitize_text(self, s: str) -> str:
        if not s:
            return ""
        out = []
        for ch in s:
            o = ord(ch)
            if ch in ("\n", "\t"):
                out.append(ch)
            elif 32 <= o <= 126:
                out.append(ch)
        return "".join(out).strip()

    def _filter_probs_to_printable(self, probs: np.ndarray) -> np.ndarray:
        p = probs.copy().astype(np.float32)
        # Never emit these directly
        p[self.tok.START] = 0.0
        p[self.tok.PAD] = 0.0
        p[self.tok.SEP] = 0.0
        for b in range(0, 256):
            if b in (9, 10, 13):
                continue
            if b < 32 or b >= 127:
                p[b] = 0.0
        s = float(np.sum(p))
        if s > 1e-12:
            p /= s
            return p
        p[:] = 0.0
        for b in range(32, 127):
            p[b] = 1.0
        p[10] = 1.0
        p /= float(np.sum(p))
        return p

    def generate(self, prompt: str, max_bytes: Optional[int] = None) -> str:
        maxb = int(max_bytes or self.cfg.max_reply_bytes)
        context = self.tok.encode(prompt, add_start_end=True)
        out_ids: List[int] = []
        temperature = float(self.cfg.temperature)
        top_k = int(self.cfg.top_k)
        sharpen = max(int(self.cfg.amp_sharpen_iters), 0)

        for _ in range(maxb):
            probs = self.next_token_probs(context + out_ids, temperature=temperature)
            probs = self._filter_probs_to_printable(probs)

            # top-k sampling
            if top_k > 0 and top_k < len(probs):
                idx = np.argpartition(-probs, top_k)[:top_k]
                p = probs[idx].astype(np.float64)
                for _it in range(sharpen):
                    p = p * p
                p = p / max(float(np.sum(p)), 1e-12)
                nxt = int(np.random.choice(idx, p=p))
            else:
                p = probs.astype(np.float64)
                for _it in range(sharpen):
                    p = p * p
                p = p / max(float(np.sum(p)), 1e-12)
                nxt = int(np.random.choice(np.arange(len(probs)), p=p))

            if nxt == self.tok.END:
                break
            out_ids.append(nxt)
            if len(out_ids) >= 2 and out_ids[-2:] == [10, 10]:
                break

        return self._sanitize_text(self.tok.decode(out_ids))

    def write_to_memory(self, text: str) -> None:
        data = (text or "").encode("utf-8", errors="replace")
        strength = _hash_to_unit_interval(data)
        phase = _hash_to_phase(data)
        h = hashlib.sha256(data).digest()
        if not self.qmem:
            return
        ch_idx = int(h[0]) % len(self.qmem)
        dim = self.qmem[ch_idx].dimension
        hid = 2 + (int(h[1]) % max(dim - 2, 1))
        self.qmem[ch_idx].inject_signal(strength=strength, phase=phase, target_hidden_idx=hid)
        if len(self.qmem) >= 2:
            ch2 = (ch_idx + 1) % len(self.qmem)
            hid2 = 2 + (int(h[2]) % max(self.qmem[ch2].dimension - 2, 1))
            self.qmem[ch2].inject_signal(strength=strength * 0.65, phase=-phase, target_hidden_idx=hid2)
        if self.cfg.qmem_noise > 0:
            for ch in self.qmem:
                ch.apply_noise_to_hidden(self.cfg.qmem_noise)

    def add_replay(self, ids: List[int]) -> None:
        if not ids:
            return
        self._replay.append(list(ids))
        maxn = int(self.cfg.replay_buffer_size)
        if maxn > 0 and len(self._replay) > maxn:
            self._replay = self._replay[-maxn:]

    def sample_replay(self, k: int) -> List[List[int]]:
        if not self._replay or k <= 0:
            return []
        k = min(k, len(self._replay))
        idx = self._rng.choice(np.arange(len(self._replay)), size=k, replace=False)
        return [self._replay[int(i)] for i in idx]

    def loss_on_sequence(self, ids: List[int], sample_positions: int) -> float:
        if len(ids) < 4:
            return 0.0
        L = len(ids)
        npos = min(int(sample_positions), max(L - 1, 1))
        lo = max(1, L - 4096)
        positions = self._rng.choice(np.arange(lo, L), size=npos, replace=True)
        total = 0.0
        for i in positions:
            ctx = ids[max(0, i - self.context_len) : i]
            tgt = int(ids[i])
            probs = self.next_token_probs(ctx, temperature=1.0)
            p = float(probs[tgt]) if 0 <= tgt < len(probs) else 1e-9
            total += -math.log(max(p, 1e-9))
        return total / float(len(positions))

    def loss_on_batch(self, seqs: List[List[int]], sample_positions: int) -> float:
        if not seqs:
            return 0.0
        return float(np.mean([self.loss_on_sequence(s, sample_positions) for s in seqs]))


    #SPSA increases speed, but isn't necessarily the best overall..
    def _fast_param_arrays(self) -> List[np.ndarray]:
        arrs: List[np.ndarray] = [self.A_U, self.A_V, self.A_b, self.Q]
        if self.vqc is not None:
            arrs.extend([self.vqc.theta, self.vqc_in, self.vqc_out])
        return arrs

    def _slow_param_arrays(self) -> List[np.ndarray]:
        # base weights (b is included but its seeded prior)
        return [self.E, self.W, self.b]

    def _spsa_update(self, arrays: List[np.ndarray], seqs: List[List[int]], *, a_scale: float = 1.0, step_counter_attr: str = "_train_step_fast") -> Dict[str, float]:
        # SPSA hyperparameters
        if step_counter_attr == "_train_step_fast":
            self._train_step_fast += 1
            k = self._train_step_fast
        else:
            self._train_step_slow += 1
            k = self._train_step_slow

        a0 = float(self.cfg.spsa_a) * float(a_scale)
        c0 = float(self.cfg.spsa_c)
        alpha = float(self.cfg.spsa_alpha)
        gamma = float(self.cfg.spsa_gamma)
        ak = a0 / ((k + 10.0) ** alpha)
        ck = c0 / ((k + 10.0) ** gamma)

        base = [a.copy() for a in arrays]
        deltas = [self._rng.choice([-1.0, 1.0], size=a.shape).astype(np.float32) for a in arrays]

        for a, a0v, d in zip(arrays, base, deltas):
            a[:] = (a0v + ck * d).astype(np.float32)
        Lp = self.loss_on_batch(seqs, sample_positions=int(self.cfg.train_positions_per_step))

        for a, a0v, d in zip(arrays, base, deltas):
            a[:] = (a0v - ck * d).astype(np.float32)
        Lm = self.loss_on_batch(seqs, sample_positions=int(self.cfg.train_positions_per_step))

        for a, a0v in zip(arrays, base):
            a[:] = a0v

        gscale = float((Lp - Lm) / max(2.0 * ck, 1e-12))
        grads = [(gscale * d).astype(np.float32) for d in deltas]

        gn = float(np.sum([np.linalg.norm(g) for g in grads]))
        if gn > float(self.cfg.grad_clip):
            s = float(self.cfg.grad_clip / max(gn, 1e-12))
            grads = [(g * s).astype(np.float32) for g in grads]

        for a, g in zip(arrays, grads):
            a[:] = (a - ak * g).astype(np.float32)

        return {"k": float(k), "ak": float(ak), "ck": float(ck), "Lp": float(Lp), "Lm": float(Lm), "gscale": float(gscale)}

    def spsa_step(self, ids: List[int]) -> Dict[str, float]:

        batch = [ids] + self.sample_replay(int(self.cfg.replay_batch))
        info = self._spsa_update(self._fast_param_arrays(), batch, a_scale=1.0, step_counter_attr="_train_step_fast")
        return info

    def maybe_consolidate(self) -> Optional[Dict[str, float]]:
        every = int(self.cfg.consolidate_every_turns)
        if every <= 0:
            return None
        if len(self._replay) == 0:
            return None
        if (len(self._replay) % every) != 0:
            return None
        batch = self.sample_replay(max(4, int(self.cfg.replay_batch) * 2))
        if not batch:
            return None
        return self._spsa_update(self._slow_param_arrays(), batch, a_scale=float(self.cfg.slow_lr_scale), step_counter_attr="_train_step_slow")


    def to_npz(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "E": self.E,
            "W": self.W,
            "b": self.b,
            "A_U": self.A_U,
            "A_V": self.A_V,
            "A_b": self.A_b,
            "Q": self.Q,
            "adapter_rank": np.array([self.adapter_rank], dtype=np.int32),
        }
        if self.vqc is not None:
            out["vqc_theta"] = self.vqc.theta
            out["vqc_in"] = self.vqc_in
            out["vqc_out"] = self.vqc_out
            out["vqc_meta"] = np.array([self.vqc.n, self.vqc.layers], dtype=np.int32)
        return out

    def load_npz(self, d: Dict[str, np.ndarray]) -> None:
        self.E = d.get("E", self.E).astype(np.float32)
        self.W = d.get("W", self.W).astype(np.float32)
        self.b = d.get("b", self.b).astype(np.float32)

        if "A_U" in d and "A_V" in d and "A_b" in d:
            self.A_U = d["A_U"].astype(np.float32)
            self.A_V = d["A_V"].astype(np.float32)
            self.A_b = d["A_b"].astype(np.float32)
            self.adapter_rank = int(self.A_U.shape[1])
        if "Q" in d:
            self.Q = d["Q"].astype(np.float32)

        if bool(self.cfg.vqc_enabled) and ("vqc_theta" in d) and ("vqc_meta" in d):
            meta = d["vqc_meta"].astype(np.int32).tolist()
            n, L = int(meta[0]), int(meta[1])
            self.vqc = TinyVQC(n_qubits=n, n_layers=L, rng=self._rng)
            self.vqc.theta = d["vqc_theta"].astype(np.float32)
            self.vqc_in = d.get("vqc_in", self.vqc_in).astype(np.float32)
            self.vqc_out = d.get("vqc_out", self.vqc_out).astype(np.float32)


class BrainPersistence:
    def __init__(self, root_dir: str):
        self.root = os.path.abspath(root_dir)
        os.makedirs(self.root, exist_ok=True)

    def list_sessions(self) -> List[str]:
        out = []
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "brain_state.json")):
                out.append(name)
        return sorted(out)

    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.root, session_id)

    def save(self, session_id: str, cfg: BrainConfig, model: BrainModel, history: List[Dict[str, Any]]) -> None:
        sdir = self.session_dir(session_id)
        os.makedirs(sdir, exist_ok=True)
        meta = {
            "session_id": session_id,
            "saved_at": _now_iso(),
            "cfg": asdict(cfg),
            "history": history,
            "qmem": [ch.to_jsonable() for ch in model.qmem],
            "format_version": 13,
        }
        with open(os.path.join(sdir, "brain_state.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.savez_compressed(os.path.join(sdir, "brain_params.npz"), **model.to_npz())

    def load(self, session_id: str) -> Tuple[BrainConfig, Dict[str, np.ndarray], List[Dict[str, Any]], List[SubBitMemoryChannel]]:
        sdir = self.session_dir(session_id)
        with open(os.path.join(sdir, "brain_state.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = self._cfg_from_dict(meta.get("cfg", {}))
        params = dict(np.load(os.path.join(sdir, "brain_params.npz")))
        history = meta.get("history", [])
        qmem = [SubBitMemoryChannel.from_jsonable(x) for x in meta.get("qmem", [])]
        return cfg, params, history, qmem

    def _cfg_from_dict(self, d: Dict[str, Any]) -> BrainConfig:
        base = BrainConfig()
        qb = d.get("qbackend", {}) if isinstance(d, dict) else {}
        qbcfg = QuantumBackendConfig(**{k: qb.get(k, getattr(QuantumBackendConfig(), k)) for k in asdict(QuantumBackendConfig()).keys()})

        def g(key: str, default: Any) -> Any:
            return d.get(key, default)

        # Construct with explicit casting to tolerate partial loads..
        return BrainConfig(
            embed_dim=int(g("embed_dim", base.embed_dim)),
            context_len=int(g("context_len", base.context_len)),
            temperature=float(g("temperature", base.temperature)),
            top_k=int(g("top_k", base.top_k)),
            max_reply_bytes=int(g("max_reply_bytes", base.max_reply_bytes)),

            continual_learning=bool(g("continual_learning", base.continual_learning)),
            spsa_steps_per_turn=int(g("spsa_steps_per_turn", base.spsa_steps_per_turn)),
            spsa_a=float(g("spsa_a", base.spsa_a)),
            spsa_c=float(g("spsa_c", base.spsa_c)),
            spsa_alpha=float(g("spsa_alpha", base.spsa_alpha)),
            spsa_gamma=float(g("spsa_gamma", base.spsa_gamma)),
            grad_clip=float(g("grad_clip", base.grad_clip)),
            train_positions_per_step=int(g("train_positions_per_step", base.train_positions_per_step)),

            adapter_rank=int(g("adapter_rank", base.adapter_rank)),
            replay_buffer_size=int(g("replay_buffer_size", base.replay_buffer_size)),
            replay_batch=int(g("replay_batch", base.replay_batch)),
            consolidate_every_turns=int(g("consolidate_every_turns", base.consolidate_every_turns)),
            slow_spsa_steps=int(g("slow_spsa_steps", base.slow_spsa_steps)),
            slow_lr_scale=float(g("slow_lr_scale", base.slow_lr_scale)),

            amp_sharpen_iters=int(g("amp_sharpen_iters", base.amp_sharpen_iters)),

            qmem_channels=int(g("qmem_channels", base.qmem_channels)),
            qmem_dimension=int(g("qmem_dimension", base.qmem_dimension)),
            qmem_noise=float(g("qmem_noise", base.qmem_noise)),

            vqc_enabled=bool(g("vqc_enabled", base.vqc_enabled)),
            vqc_qubits=int(g("vqc_qubits", base.vqc_qubits)),
            vqc_layers=int(g("vqc_layers", base.vqc_layers)),
            vqc_scale=float(g("vqc_scale", base.vqc_scale)),

            internet_enabled=bool(g("internet_enabled", base.internet_enabled)),
            auto_web=bool(g("auto_web", base.auto_web)),

            semantic_memory=bool(g("semantic_memory", base.semantic_memory)),
            semantic_topk=int(g("semantic_topk", base.semantic_topk)),
            semantic_store_turns=bool(g("semantic_store_turns", base.semantic_store_turns)),

            autosave_every_turns=int(g("autosave_every_turns", base.autosave_every_turns)),
            qbackend=qbcfg,
        )

# Internet class, used common llm wiring
class InternetTools:
    def __init__(self, enabled: bool = True, timeout_s: float = 6.0):
        self.enabled = enabled
        self.timeout_s = float(timeout_s)

    def ddg_instant_answer(self, query: str) -> Optional[str]:
        if not self.enabled or requests is None:
            return None
        q = (query or "").strip()
        if not q:
            return None
        try:
            r = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": q, "format": "json", "no_redirect": "1", "no_html": "1"},
                timeout=self.timeout_s,
                headers={"User-Agent": "Brain/1.0 (QELM)"},
            )
            if r.status_code != 200:
                return None
            j = r.json()
            abs_ = (j.get("AbstractText") or "").strip()
            if abs_:
                src = (j.get("AbstractSource") or "").strip()
                return f"{abs_}\n(Source: {src})" if src else abs_
        except Exception:
            return None
        return None

    def wikipedia_summary(self, query: str) -> Optional[str]:
        if not self.enabled or requests is None:
            return None
        q = (query or "").strip()
        if not q:
            return None
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": q, "format": "json"},
                timeout=self.timeout_s,
                headers={"User-Agent": "Brain/1.0 (QELM)"},
            )
            if r.status_code != 200:
                return None
            hits = (((r.json().get("query") or {}).get("search")) or [])
            if not hits:
                return None
            title = hits[0].get("title")
            if not title:
                return None
            rs = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(str(title)),
                timeout=self.timeout_s,
                headers={"User-Agent": "Brain/1.0 (QELM)"},
            )
            if rs.status_code != 200:
                return None
            extract = (rs.json().get("extract") or "").strip()
            if extract:
                return f"{extract}\n(Source: Wikipedia)"
        except Exception:
            return None
        return None

    def ddg_html_snippets(self, query: str, max_results: int = 5) -> List[Tuple[str, str, str]]:
        if not self.enabled or requests is None:
            return []
        q = (query or "").strip()
        if not q:
            return []
        try:
            url = "https://duckduckgo.com/html/"
            resp = requests.post(url, data={"q": q}, timeout=self.timeout_s, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return []
            html = resp.text
            items: List[Tuple[str, str, str]] = []
            for m in re.finditer(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                html,
                flags=re.S,
            ):
                href = m.group(1)
                title = re.sub(r"<.*?>", "", m.group(2))
                snip = re.sub(r"<.*?>", "", m.group(3))
                title = _sm_safe_text(title)
                snip = _sm_safe_text(snip)
                if title:
                    items.append((title, href, snip))
                if len(items) >= int(max_results):
                    break
            return items
        except Exception:
            return []

    def lookup(self, query: str) -> Optional[str]:
        q = (query or "").strip()
        if not q:
            return None
        items = self.ddg_html_snippets(q, max_results=5)
        if items:
            lines = ["Web lookup results:"]
            for i, (title, url, snip) in enumerate(items, 1):
                lines.append(f"{i}. {title}\n   {url}" + (f"\n   {snip}" if snip else ""))
            return "\n".join(lines)
        res = self.ddg_instant_answer(q)
        if res:
            return res
        return self.wikipedia_summary(q)


_WORD_RE_SM = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _sm_terms(text: str, max_terms: int = 10) -> List[str]:
    toks = _WORD_RE_SM.findall((text or "").lower())
    toks = [t for t in toks if len(t) >= 2]
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _sm_build_fts_query(text: str, max_terms: int = 10) -> str:
    terms = _sm_terms(text, max_terms=max_terms)
    if not terms:
        return ""
    return " ".join([t + "*" for t in terms])


def _sm_split_chunks(text: str, max_chars: int = 800) -> List[str]:
    text = _sm_safe_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    chunks: List[str] = []
    buf: List[str] = []
    n = 0
    for p in parts:
        if not p:
            continue
        if n + len(p) + 1 > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = [p]
            n = len(p)
        else:
            buf.append(p)
            n += len(p) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    # hard split
    out: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) > max_chars:
            for i in range(0, len(c), max_chars):
                out.append(c[i : i + max_chars])
        else:
            out.append(c)
    return out


class SemanticMemory:

    def __init__(self, db_path: str, log_cb: Optional[Callable[[str], None]] = None):
        self.db_path = db_path
        self.log_cb = log_cb
        self._lock = threading.Lock()
        self._fts_ok = False
        self._init_db()

    def _log(self, msg: str) -> None:
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = self._connect()
        try:
            con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS mem USING fts5(session, kind, q, a, text);""")
            self._fts_ok = True
            self._log("[SEM] FTS5 enabled.")
        except Exception as e:
            self._fts_ok = False
            self._log(f"[SEM] FTS5 not available; using plain table. ({e})")
            con.execute(
                """CREATE TABLE IF NOT EXISTS mem_plain (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       session TEXT,
                       kind TEXT,
                       q TEXT,
                       a TEXT,
                       text TEXT
                   );"""
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_plain_session_kind ON mem_plain(session, kind);")
        con.commit()
        con.close()

    def add_rows(self, rows: List[Tuple[str, str, str, str, str]]) -> None:
        if not rows:
            return
        clean = [(_sm_safe_text(s), _sm_safe_text(k), _sm_safe_text(q), _sm_safe_text(a), _sm_safe_text(t)) for (s, k, q, a, t) in rows]
        with self._lock:
            con = self._connect()
            try:
                if self._fts_ok:
                    con.executemany("INSERT INTO mem(session, kind, q, a, text) VALUES (?,?,?,?,?)", list(clean))
                else:
                    con.executemany("INSERT INTO mem_plain(session, kind, q, a, text) VALUES (?,?,?,?,?)", list(clean))
                con.commit()
            finally:
                con.close()

    def add_qa(self, session: str, q: str, a: str) -> None:
        self.add_rows([(session, "qa", q, a, f"Q: {q}\nA: {a}")])

    def add_corpus_chunk(self, session: str, source: str, chunk: str) -> None:
        self.add_rows([(session, "corpus", source, "", chunk)])

    def add_turn(self, session: str, who_kind: str, text: str) -> None:
        self.add_rows([(session, who_kind, "", "", text)])

    def stats(self, session: str) -> str:
        con = self._connect()
        try:
            cur = con.cursor()
            if self._fts_ok:
                cur.execute("SELECT kind, COUNT(*) FROM mem WHERE session=? GROUP BY kind ORDER BY kind", (session,))
            else:
                cur.execute("SELECT kind, COUNT(*) FROM mem_plain WHERE session=? GROUP BY kind ORDER BY kind", (session,))
            rows = cur.fetchall()
        finally:
            con.close()
        if not rows:
            return "No semantic memory rows for this session."
        lines = ["Semantic memory counts:"]
        for k, c in rows:
            lines.append(f"  {k}: {c}")
        return "\n".join(lines)

    def _query_fts(self, session: str, kind: str, query: str, limit: int) -> List[Dict[str, str]]:
        q = _sm_build_fts_query(query)
        if not q:
            return []
        con = self._connect()
        try:
            cur = con.cursor()
            try:
                cur.execute(
                    "SELECT session, kind, q, a, text FROM mem WHERE mem MATCH ? AND session=? AND kind=? ORDER BY bm25(mem) LIMIT ?",
                    (q, session, kind, int(limit)),
                )
            except Exception:
                cur.execute(
                    "SELECT session, kind, q, a, text FROM mem WHERE mem MATCH ? AND session=? AND kind=? LIMIT ?",
                    (q, session, kind, int(limit)),
                )
            rows = cur.fetchall()
            return [{"session": r[0], "kind": r[1], "q": r[2] or "", "a": r[3] or "", "text": r[4] or ""} for r in rows]
        finally:
            con.close()

    def _query_plain(self, session: str, kind: str, query: str, limit: int) -> List[Dict[str, str]]:
        terms = _sm_terms(query, max_terms=6)
        if not terms:
            return []
        like = "%" + "%".join(terms) + "%"
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT session, kind, q, a, text FROM mem_plain WHERE session=? AND kind=? AND (q LIKE ? OR a LIKE ? OR text LIKE ?) LIMIT ?",
                (session, kind, like, like, like, int(limit)),
            )
            rows = cur.fetchall()
            return [{"session": r[0], "kind": r[1], "q": r[2] or "", "a": r[3] or "", "text": r[4] or ""} for r in rows]
        finally:
            con.close()

    def find(self, session: str, kind: str, query: str, limit: int = 3) -> List[Dict[str, str]]:
        query = _sm_safe_text(query)
        if not query:
            return []
        if self._fts_ok:
            return self._query_fts(session, kind, query, limit)
        return self._query_plain(session, kind, query, limit)

    def find_qa(self, session: str, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        hits = self.find(session, "qa", query, limit)
        out: List[Tuple[str, str]] = []
        for h in hits:
            q = h.get("q", "")
            a = h.get("a", "")
            if q and a:
                out.append((q, a))
        return out

    def find_corpus(self, session: str, query: str, limit: int = 3) -> List[str]:
        hits = self.find(session, "corpus", query, limit)
        return [h.get("text", "") for h in hits if h.get("text")]


_GREETINGS = {"hi", "hello", "hey", "yo", "sup", "hiya", "howdy"}


def _brain_norm(s: str) -> str:
    s = _sm_safe_text(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _brain_builtin_answer(s: str) -> Optional[str]:
    n = _brain_norm(s)
    if not n:
        return "Hello. What should we work on? Use /help for commands."
    if n in _GREETINGS:
        return "Hello. What should we work on? Use /help for commands."
    if n in {"whats your name", "what is your name", "your name"}:
        return "My name is Brain."
    if n in {"who are you", "what are you"}:
        return (
            "I am Brain — a standalone hybrid system that learns via (1) persistent semantic memory, "
            "(2) sub-bit quantum memory channels, and (3) background SPSA updates on fast adapter parameters. "
            "Use /help for commands, /teach to store a fact, /web for live lookup, and /trainfile to ingest a local corpus."
        )
    if n in {"how do you learn", "how do you learn"}:
        return (
            "I learn in three layers: (1) semantic memory for exact recall (taught Q/A and corpus chunks), "
            "(2) sub-bit quantum memory writes each turn (a persistent bias signal), and "
            "(3) background SPSA updates on small adapter parameters (fast on-the-fly learning). "
            "If you want me to learn a specific answer, use /teach <question> => <answer>."
        )
    if n in {"what is qelm", "qelm"}:
        return (
            "QELM (Quantum‑Enhanced Language Model) is your hybrid quantum/classical language-model architecture. "
            "It uses quantum channels (parameterized circuits) as computational units, a SubBitDecoder to convert quantum state outputs into classical features, "
            "and quantum-enhanced transformer blocks (quantum attention + quantum feed-forward). "
            "Training typically uses parameter-shift gradients with SPSA as a practical optimizer."
        )
    return None


class BrainEngine:
    def __init__(self, root_dir: str = "brain_sessions", session_id: Optional[str] = None, cfg: Optional[BrainConfig] = None, log_cb=None):
        self.root_dir = os.path.abspath(root_dir)
        self.persistence = BrainPersistence(self.root_dir)
        self.session_id = session_id or time.strftime("session_%Y%m%d_%H%M%S")

        self.cfg = cfg or BrainConfig()
        self.model = BrainModel(self.cfg)
        self.history: List[Dict[str, Any]] = []
        self._turns = 0

        self.log_cb = log_cb
        self.qbackend = QuantumBackendManager(self.cfg.qbackend)
        self.internet = InternetTools(enabled=self.cfg.internet_enabled)
        self.semantic: Optional[SemanticMemory] = None
        if self.cfg.semantic_memory:
            self.semantic = SemanticMemory(os.path.join(self.persistence.session_dir(self.session_id), "semantic.sqlite"), log_cb=self._log)

        self._lock = threading.RLock()
        self._train_q: "queue.Queue[List[int]]" = queue.Queue(maxsize=512)
        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

        self._hf_cancel = threading.Event()
        self._hf_thread: Optional[threading.Thread] = None
        self._ingest_cancel = threading.Event()

        # Altering this will make it not learn from previous conversations, please leave..
        try:
            if session_id and session_id in self.persistence.list_sessions():
                self._load_session(session_id)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    def update_config(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            for k, v in (updates or {}).items():
                if k == "qbackend" and isinstance(v, dict):
                    for kk, vv in v.items():
                        if hasattr(self.cfg.qbackend, kk):
                            setattr(self.cfg.qbackend, kk, vv)
                    self.qbackend.cfg = self.cfg.qbackend
                elif hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)

            self.internet.enabled = bool(self.cfg.internet_enabled)
            if self.cfg.semantic_memory and self.semantic is None:
                self.semantic = SemanticMemory(os.path.join(self.persistence.session_dir(self.session_id), "semantic.sqlite"), log_cb=self._log)
            if (not self.cfg.semantic_memory) and self.semantic is not None:
                self.semantic = None

    def _load_session(self, session_id: str) -> None:
        cfg, params, history, qmem = self.persistence.load(session_id)
        self.session_id = session_id
        self.cfg = cfg
        self.model = BrainModel(self.cfg)
        self.model.load_npz(params)
        if qmem:
            self.model.qmem = qmem
        self.history = history
        self.qbackend = QuantumBackendManager(self.cfg.qbackend)
        self.internet = InternetTools(enabled=self.cfg.internet_enabled)
        self.semantic = SemanticMemory(os.path.join(self.persistence.session_dir(self.session_id), "semantic.sqlite"), log_cb=self._log) if self.cfg.semantic_memory else None
        self._log(f"[SYS] Loaded session {session_id}")

    def save(self) -> None:
        with self._lock:
            self.persistence.save(self.session_id, self.cfg, self.model, self.history)


    def _train_worker(self) -> None:
        while True:
            ids = self._train_q.get()
            if ids is None:
                return
            try:
                self.model.add_replay(ids)
                steps = max(int(self.cfg.spsa_steps_per_turn), 0)
                for _ in range(steps):
                    info = self.model.spsa_step(ids)
                    self._log(f"[LEARN] SPSA fast k={int(info['k'])} L+={info['Lp']:.3f} L-={info['Lm']:.3f} ak={info['ak']:.4f} ck={info['ck']:.4f}")
                info2 = self.model.maybe_consolidate()
                if info2:
                    self._log(f"[LEARN] Consolidate k={int(info2['k'])} L+={info2['Lp']:.3f} L-={info2['Lm']:.3f} ak={info2['ak']:.4f}")
            except Exception as e:
                self._log(f"[LEARN] Error: {e}\n{traceback.format_exc()}")

    def _enqueue_learning(self, ids: List[int]) -> None:
        if not self.cfg.continual_learning:
            return
        try:
            self._train_q.put_nowait(list(ids))
        except Exception:
            pass

    def _cmd_help(self) -> str:
        return (
            "Commands:\n"
            "  /help                     Show this help\n"
            "  /teach <q> => <a>          Store a Q/A pair in semantic memory\n"
            "  /web <query>               Perform a live web lookup (no API key)\n"
            "  /hf <ds>|<cfg>|<split>|<col>  Stream a HuggingFace dataset into semantic memory (background)\n"
            "  /cancel                   Cancel active /trainfile or /hf ingestion\n"
            "  /stats                    Show semantic memory counts\n"
            "  /save                     Save the current session\n"
            "  /trainfile <path>         Ingest a local text file into semantic memory (background)\n"
            "  /reset                    Clear conversation history (does not wipe saved model params)\n"
        )

    # HF connectors/adaptors
    def start_hf_ingest(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        max_rows: int = 0,
    ) -> None:

        if self.semantic is None:
            self._log("[HF] Semantic memory is OFF; ingestion skipped.")
            return

        try:
            import datasets
        except Exception:
            self._log("[HF] Python package 'datasets' not installed. Install with: pip install datasets")
            return

        self._hf_cancel.clear()

        def worker() -> None:
            try:
                ds_label = dataset_name if not dataset_config else f"{dataset_name} ({dataset_config})"
                self._log(f"[HF] Streaming dataset: {ds_label} | split={split} | column={text_column}")

                ds = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=True)
                n = 0
                buf: List[str] = []
                for row in ds:
                    if self._hf_cancel.is_set():
                        self._log("[HF] Cancelled.")
                        return
                    txt = _sm_safe_text(str(row.get(text_column, "")))
                    if not txt:
                        continue
                    buf.append(txt)
                    if len(buf) >= 8:
                        joined = "\n".join(buf)
                        buf = []
                        for chunk in _sm_split_chunks(joined, max_chars=800):
                            self.semantic.add_corpus_chunk(self.session_id, ds_label, chunk)
                        # Queue occasional learning on raw text (optional)
                        ids = self.model.tok.encode(joined, add_start_end=True)
                        self._enqueue_learning(ids)
                    n += 1
                    if max_rows and n >= max_rows:
                        break
                    if n % 200 == 0:
                        self._log(f"[HF] Rows ingested: {n}")

                if buf:
                    joined = "\n".join(buf)
                    for chunk in _sm_split_chunks(joined, max_chars=800):
                        self.semantic.add_corpus_chunk(self.session_id, ds_label, chunk)
                self._log(f"[HF] Done. Rows ingested: {n}")
            except Exception as e:
                self._log(f"[HF] Error: {e}\n{traceback.format_exc()}")

        self._hf_thread = threading.Thread(target=worker, daemon=True)
        self._hf_thread.start()

    def cancel_ingest(self) -> None:
        """Signal cancellation for background /trainfile and /hf ingestion."""
        try:
            self._ingest_cancel.set()
        except Exception:
            pass
        try:
            self._hf_cancel.set()
        except Exception:
            pass


    def start_corpus_training(self, file_path: str, max_lines: int = 0) -> None:

        fp = os.path.abspath(file_path)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

        self._ingest_cancel.clear()

        def worker() -> None:
            try:
                self._log(f"[CORP] Ingesting file: {fp}")
                if self.semantic is None:
                    self._log("[CORP] Semantic memory is OFF; ingestion skipped.")
                    return
                n = 0
                buf: List[str] = []
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if self._ingest_cancel.is_set():
                            self._log("[CORP] Cancelled.")
                            return
                        line = _sm_safe_text(line)
                        if not line:
                            continue
                        buf.append(line)
                        if len(buf) >= 8:
                            chunk = " ".join(buf)
                            for c in _sm_split_chunks(chunk, max_chars=800):
                                self.semantic.add_corpus_chunk(self.session_id, os.path.basename(fp), c)
                            buf = []
                        n += 1
                        if max_lines and n >= int(max_lines):
                            break
                        if self.cfg.continual_learning and (n % 200 == 0):
                            ids = self.model.tok.encode("\n".join(buf[-3:]), add_start_end=True)
                            self._enqueue_learning(ids)
                if buf:
                    chunk = " ".join(buf)
                    for c in _sm_split_chunks(chunk, max_chars=800):
                        self.semantic.add_corpus_chunk(self.session_id, os.path.basename(fp), c)
                self._log(f"[CORP] Ingest complete. Lines read: {n}")
            except Exception as e:
                self._log(f"[CORP] Error: {e}\n{traceback.format_exc()}")

        threading.Thread(target=worker, daemon=True).start()


    def _build_prompt(self, user_text: str, retrieved: str = "", web: str = "") -> str:
        turns = []
        for t in self.history[-8:]:
            role = t.get("role", "").upper()
            content = t.get("content", "")
            if role and content:
                turns.append(f"{role}: {content}")

        blocks: List[str] = []
        if retrieved:
            blocks.append("[MEMORY]\n" + retrieved.strip())
        if web:
            blocks.append("[WEB]\n" + web.strip())
        if turns:
            blocks.append("\n".join(turns))
        blocks.append(f"USER: {user_text}\nBRAIN:")
        return "\n\n".join([b for b in blocks if b.strip()])

    def respond(self, user_text: str) -> str:
        user_text = _sm_safe_text(user_text)
        if not user_text:
            return ""

        with self._lock:
            # Commands
            if user_text.strip().startswith("/"):
                cmd = user_text.strip().split(" ", 1)[0].lower()
                rest = user_text.strip()[len(cmd) :].strip()
                if cmd == "/help":
                    return self._cmd_help()
                if cmd == "/reset":
                    self.history = []
                    return "Conversation history cleared."
                if cmd == "/save":
                    self.save()
                    return "Saved."
                if cmd == "/stats":
                    if self.semantic is None:
                        return "Semantic memory is OFF."
                    return self.semantic.stats(self.session_id)
                if cmd == "/web":
                    if not rest:
                        return "Usage: /web <query>"
                    res = self.internet.lookup(rest)
                    return res or "No web results."
                if cmd == "/teach":
                    if self.semantic is None:
                        return "Semantic memory is OFF; cannot /teach."
                    if "=>" not in rest:
                        return "Usage: /teach <question> => <answer>"
                    q, a = [x.strip() for x in rest.split("=>", 1)]
                    if not q or not a:
                        return "Usage: /teach <question> => <answer>"
                    self.semantic.add_qa(self.session_id, q, a)
                    self.model.write_to_memory(f"TEACH: {q} => {a}")
                    return "Stored."
                if cmd == "/trainfile":
                    if not rest:
                        return "Usage: /trainfile <path>"
                    try:
                        self.start_corpus_training(rest, max_lines=0)
                        return f"Started ingest: {rest}"
                    except Exception as e:
                        return f"Error: {e}"
                if cmd == "/hf":
                    if not rest:
                        return "Usage: /hf <dataset>|<config>|<split>|<text_col>  (use '|' separators; omit trailing fields if not needed)"
                    try:
                        parts = [p.strip() for p in rest.split("|")]
                        ds = parts[0] if len(parts) > 0 else ""
                        cfg = parts[1] if len(parts) > 1 and parts[1] else None
                        split = parts[2] if len(parts) > 2 and parts[2] else "train"
                        col = parts[3] if len(parts) > 3 and parts[3] else "text"
                        if not ds:
                            return "Usage: /hf <dataset>|<config>|<split>|<text_col>"
                        self.start_hf_ingest(dataset_name=ds, dataset_config=cfg, split=split, text_column=col)
                        return f"Started HF ingest: {ds} | {cfg or '-'} | {split} | {col}"
                    except Exception as e:
                        return f"Error: {e}"
                if cmd == "/cancel":
                    self.cancel_ingest()
                    return "Cancel signaled."
                return "Unknown command. Use /help."

            builtin = _brain_builtin_answer(user_text)
            if builtin is not None:
                self.model.write_to_memory(user_text + "\n" + builtin)
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "brain", "content": builtin})
                return builtin

            retrieved = ""
            qa_best: Optional[Tuple[str, str]] = None
            if self.semantic is not None and self.cfg.semantic_memory:
                qa = self.semantic.find_qa(self.session_id, user_text, limit=int(self.cfg.semantic_topk))
                if qa:
                    qa_best = qa[0]
                    lines = []
                    for q, a in qa[: int(self.cfg.semantic_topk)]:
                        lines.append(f"Q: {q}\nA: {a}")
                    retrieved = "\n\n".join(lines)
                else:
                    corpus_hits = self.semantic.find_corpus(self.session_id, user_text, limit=max(1, int(self.cfg.semantic_topk)))
                    if corpus_hits:
                        retrieved = "\n\n".join(corpus_hits[: int(self.cfg.semantic_topk)])

            web = ""
            if self.cfg.internet_enabled and self.cfg.auto_web:
                if (not retrieved) and (len(_sm_terms(user_text, max_terms=12)) >= 2) and (len(user_text) >= 12):
                    web = self.internet.lookup(user_text) or ""

            self.history.append({"role": "user", "content": user_text})

            if qa_best is not None:
                q0, a0 = qa_best
                qt = set(_sm_terms(user_text, max_terms=12))
                tt = set(_sm_terms(q0, max_terms=12))
                score = (len(qt & tt) / max(1, len(qt))) if qt else 0.0
                if score >= 0.60 or _brain_norm(user_text) == _brain_norm(q0):
                    reply = a0.strip()
                    if not reply:
                        reply = "(Stored answer was empty.)"
                    self.history.append({"role": "brain", "content": reply})
                    if self.semantic is not None and self.cfg.semantic_store_turns:
                        try:
                            self.semantic.add_turn(self.session_id, "turn_user", user_text)
                            self.semantic.add_turn(self.session_id, "turn_brain", reply)
                        except Exception:
                            pass
                    self.model.write_to_memory(user_text + "\n" + reply)
                    ids = self.model.tok.encode("USER: " + user_text + "\nBRAIN: " + reply, add_start_end=True)
                    self._enqueue_learning(ids)
                    self._turns += 1
                    if int(self.cfg.autosave_every_turns) > 0 and (self._turns % int(self.cfg.autosave_every_turns) == 0):
                        try:
                            self.save()
                            self._log("[SYS] Autosaved.")
                        except Exception:
                            pass
                    return reply

            prompt = self._build_prompt(user_text, retrieved=retrieved, web=web)
            reply = self.model.generate(prompt, max_bytes=int(self.cfg.max_reply_bytes))
            if not reply:
                if retrieved:
                    reply = "I found relevant stored information, but generation is not trained yet.\n\n" + retrieved.strip()
                elif web:
                    reply = "I found relevant web information, but generation is not trained yet.\n\n" + web.strip()
                else:
                    reply = "I don't know yet. If you want me to remember a specific answer, use: /teach <question> => <answer>."

            self.history.append({"role": "brain", "content": reply})

            if self.semantic is not None and self.cfg.semantic_store_turns:
                try:
                    self.semantic.add_turn(self.session_id, "turn_user", user_text)
                    self.semantic.add_turn(self.session_id, "turn_brain", reply)
                except Exception:
                    pass

            self.model.write_to_memory(user_text + "\n" + reply)

            ids = self.model.tok.encode("USER: " + user_text + "\nBRAIN: " + reply, add_start_end=True)
            self._enqueue_learning(ids)

            self._turns += 1
            if int(self.cfg.autosave_every_turns) > 0 and (self._turns % int(self.cfg.autosave_every_turns) == 0):
                try:
                    self.save()
                    self._log("[SYS] Autosaved.")
                except Exception:
                    pass

            return reply
          
if _TK_AVAILABLE:

    class BrainApp(tk.Tk):
        def __init__(self, engine: BrainEngine):
            super().__init__()
            self.engine = engine
            self.title("Brain — Quantum-Enhanced Continual LLM (Standalone)")
            self.geometry("1100x720")

            self._bg = "#111111"
            self._fg = "#eaeaea"
            self._accent = "#2b7cff"
            try:
                self.configure(background=self._bg)
            except Exception:
                pass

            self._build_ui()
            self.after(500, self._refresh_status)

        def _build_ui(self):
            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True)

            chat_tab = ttk.Frame(nb)
            nb.add(chat_tab, text="Chat")

            top = ttk.Frame(chat_tab)
            top.pack(fill="x")
            self.status_lbl = ttk.Label(top, text="", justify="left")
            self.status_lbl.pack(side="left", padx=8, pady=6)

            self.var_continual = tk.BooleanVar(value=self.engine.cfg.continual_learning)
            self.var_inet = tk.BooleanVar(value=self.engine.cfg.internet_enabled)
            self.var_auto_web = tk.BooleanVar(value=self.engine.cfg.auto_web)
            qc = ttk.Frame(top)
            qc.pack(side="right", padx=8)
            ttk.Checkbutton(qc, text="Continual", variable=self.var_continual, command=self._apply_quick_controls).pack(side="left", padx=4)
            ttk.Checkbutton(qc, text="Internet", variable=self.var_inet, command=self._apply_quick_controls).pack(side="left", padx=4)
            ttk.Checkbutton(qc, text="Auto web", variable=self.var_auto_web, command=self._apply_quick_controls).pack(side="left", padx=4)

            self.chat_log = scrolledtext.ScrolledText(chat_tab, wrap=tk.WORD, height=24)
            self.chat_log.pack(fill="both", expand=True, padx=8, pady=8)
            try:
                self.chat_log.configure(background=self._bg, foreground=self._fg, insertbackground=self._fg)
            except Exception:
                pass
            self.chat_log.configure(state="disabled")

            bottom = ttk.Frame(chat_tab)
            bottom.pack(fill="x", padx=8, pady=(0, 8))
            self.input_var = tk.StringVar(value="")
            entry = ttk.Entry(bottom, textvariable=self.input_var)
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<Return>", lambda e: self._send())
            ttk.Button(bottom, text="Send", command=self._send).pack(side="left", padx=(8, 0))

            settings_tab = ttk.Frame(nb)
            nb.add(settings_tab, text="Backends / Training")

            frame = ttk.Frame(settings_tab)
            frame.pack(fill="both", expand=True, padx=10, pady=10)

            left = ttk.Frame(frame)
            left.pack(side="left", fill="y")
            right = ttk.Frame(frame)
            right.pack(side="right", fill="both", expand=True)


            model_box = ttk.LabelFrame(left, text="Model")
            model_box.pack(fill="x", pady=(0, 10))
            self.var_temp = tk.DoubleVar(value=float(self.engine.cfg.temperature))
            self.var_topk = tk.IntVar(value=int(self.engine.cfg.top_k))
            self.var_maxb = tk.IntVar(value=int(self.engine.cfg.max_reply_bytes))
            ttk.Label(model_box, text="Temperature").grid(row=0, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(model_box, textvariable=self.var_temp, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(model_box, text="Top-k").grid(row=1, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(model_box, textvariable=self.var_topk, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(model_box, text="Max reply bytes").grid(row=2, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(model_box, textvariable=self.var_maxb, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=4)

            learn_box = ttk.LabelFrame(left, text="Continual Learning (SPSA, CPU)")
            learn_box.pack(fill="x", pady=(0, 10))
            self.var_steps = tk.IntVar(value=int(self.engine.cfg.spsa_steps_per_turn))
            self.var_a = tk.DoubleVar(value=float(self.engine.cfg.spsa_a))
            self.var_c = tk.DoubleVar(value=float(self.engine.cfg.spsa_c))
            self.var_clip = tk.DoubleVar(value=float(self.engine.cfg.grad_clip))
            ttk.Label(learn_box, text="Steps per user turn").grid(row=0, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(learn_box, textvariable=self.var_steps, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(learn_box, text="a (learning rate scale)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(learn_box, textvariable=self.var_a, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(learn_box, text="c (perturb scale)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(learn_box, textvariable=self.var_c, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(learn_box, text="Grad clip").grid(row=3, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(learn_box, textvariable=self.var_clip, width=10).grid(row=3, column=1, sticky="w", padx=6, pady=4)

            qmem_box = ttk.LabelFrame(left, text="Quantum Sub-bit Memory")
            qmem_box.pack(fill="x")
            self.var_qnoise = tk.DoubleVar(value=float(self.engine.cfg.qmem_noise))
            ttk.Label(qmem_box, text="Hidden noise level").grid(row=0, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(qmem_box, textvariable=self.var_qnoise, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)

            ttk.Button(left, text="Apply Settings", command=self._apply_settings).pack(anchor="w", pady=(10, 0))

            inet_box = ttk.LabelFrame(right, text="Internet")
            inet_box.pack(fill="x", pady=(0, 10))
            self.var_inet2 = tk.BooleanVar(value=self.engine.cfg.internet_enabled)
            self.var_auto_web2 = tk.BooleanVar(value=self.engine.cfg.auto_web)
            ttk.Checkbutton(inet_box, text="Enable internet lookup", variable=self.var_inet2).pack(anchor="w", padx=6, pady=4)
            ttk.Checkbutton(inet_box, text="Auto lookup (heuristic)", variable=self.var_auto_web2).pack(anchor="w", padx=6, pady=4)

            qb_box = ttk.LabelFrame(right, text="Quantum Backend (Assist)")
            qb_box.pack(fill="x", pady=(0, 10))
            self.var_qmode = tk.StringVar(value=self.engine.cfg.qbackend.mode)
            ttk.Radiobutton(qb_box, text="Aer / Local", value="aer", variable=self.var_qmode).pack(anchor="w", padx=6, pady=2)
            ttk.Radiobutton(qb_box, text="IBM QPU", value="ibm", variable=self.var_qmode).pack(anchor="w", padx=6, pady=2)
            ttk.Radiobutton(qb_box, text="Off", value="off", variable=self.var_qmode).pack(anchor="w", padx=6, pady=2)
            self.var_shots = tk.IntVar(value=int(self.engine.cfg.qbackend.shots))
            ttk.Label(qb_box, text="Shots").pack(anchor="w", padx=6, pady=(6, 0))
            ttk.Entry(qb_box, textvariable=self.var_shots, width=12).pack(anchor="w", padx=6, pady=4)
            ttk.Button(qb_box, text="Configure IBM…", command=self._menu_configure_ibm).pack(fill="x", padx=6, pady=(6, 0))

            corp_box = ttk.LabelFrame(right, text="Corpus Training")
            corp_box.pack(fill="x")
            self.corpus_path_var = tk.StringVar(value="")
            ttk.Entry(corp_box, textvariable=self.corpus_path_var).pack(fill="x", padx=6, pady=(6, 2))
            ttk.Button(corp_box, text="Choose file…", command=self._choose_corpus).pack(fill="x", padx=6, pady=2)
            self.corpus_lines_var = tk.StringVar(value="(optional) max lines, blank = all")
            ttk.Entry(corp_box, textvariable=self.corpus_lines_var).pack(fill="x", padx=6, pady=2)
            ttk.Button(corp_box, text="Start training (background)", command=self._start_corpus_training).pack(fill="x", padx=6, pady=(2, 6))

            log_box = ttk.LabelFrame(frame, text="Training / System Log")
            log_box.pack(fill="both", expand=True, pady=(12, 0))
            self.log = scrolledtext.ScrolledText(log_box, wrap=tk.WORD, height=12)
            self.log.pack(fill="both", expand=True)
            try:
                self.log.configure(background=self._bg, foreground=self._fg, insertbackground=self._fg)
            except Exception:
                pass
            self.log.configure(state="disabled")

        def _append_chat(self, role: str, text: str):
            self.chat_log.configure(state="normal")
            self.chat_log.insert(tk.END, f"{role}: {text}\n\n")
            self.chat_log.see(tk.END)
            self.chat_log.configure(state="disabled")

        def _log(self, msg: str):
            self.log.configure(state="normal")
            self.log.insert(tk.END, msg.rstrip() + "\n")
            self.log.see(tk.END)
            self.log.configure(state="disabled")

        def _refresh_status(self):
            try:
                cfg = self.engine.cfg
                s = [
                    f"Session: {self.engine.session_id}",
                    f"Turns: {len(self.engine.history)//2}",
                    f"Continual learning: {'ON' if cfg.continual_learning else 'OFF'}",
                    f"Internet: {'ON' if cfg.internet_enabled else 'OFF'} (auto={cfg.auto_web})",
                    f"Quantum assist: {cfg.qbackend.mode.upper()}",
                    f"Embed dim: {cfg.embed_dim} | Context: {cfg.context_len}",
                    f"SPSA steps/turn: {cfg.spsa_steps_per_turn}",
                    f"Replay: {len(self.engine.model._replay)} / {cfg.replay_buffer_size}",
                ]
                self.status_lbl.config(text="\n".join(s))
            except Exception:
                pass
            self.after(800, self._refresh_status)

        def _apply_quick_controls(self):
            self.engine.update_config({
                "continual_learning": bool(self.var_continual.get()),
                "internet_enabled": bool(self.var_inet.get()),
                "auto_web": bool(self.var_auto_web.get()),
            })

        def _send(self):
            txt = self.input_var.get().strip()
            if not txt:
                return
            self.input_var.set("")
            self._append_chat("USER", txt)

            def work():
                try:
                    reply = self.engine.respond(txt)
                except Exception as e:
                    reply = f"[Error] {e}\n{traceback.format_exc()}"
                self.after(0, lambda: self._append_chat("BRAIN", reply))

            threading.Thread(target=work, daemon=True).start()

        def _apply_settings(self):
            updates = {
                "temperature": float(self.var_temp.get()),
                "top_k": int(self.var_topk.get()),
                "max_reply_bytes": int(self.var_maxb.get()),
                "spsa_steps_per_turn": int(self.var_steps.get()),
                "spsa_a": float(self.var_a.get()),
                "spsa_c": float(self.var_c.get()),
                "grad_clip": float(self.var_clip.get()),
                "qmem_noise": float(self.var_qnoise.get()),
                "internet_enabled": bool(self.var_inet2.get()),
                "auto_web": bool(self.var_auto_web2.get()),
                "qbackend": {"mode": str(self.var_qmode.get()), "shots": int(self.var_shots.get())},
            }
            self.engine.update_config(updates)
            self._log("[UI] Applied settings.")

        def _choose_corpus(self):
            if filedialog is None:
                return
            fp = filedialog.askopenfilename(title="Select corpus text file")
            if fp:
                self.corpus_path_var.set(fp)

        def _start_corpus_training(self):
            fp = self.corpus_path_var.get().strip()
            if not fp:
                self._log("[UI] No corpus file selected.")
                return
            ml = self.corpus_lines_var.get().strip()
            max_lines = 0
            try:
                if ml and ml.isdigit():
                    max_lines = int(ml)
            except Exception:
                max_lines = 0
            try:
                self.engine.start_corpus_training(fp, max_lines=max_lines)
                self._log(f"[UI] Started corpus ingest: {fp}")
            except Exception as e:
                self._log(f"[UI] Corpus ingest error: {e}")

        def _menu_configure_ibm(self):
            if simpledialog is None:
                self._log("[IBM] Tk dialogs unavailable.")
                return
            token = simpledialog.askstring("IBM Quantum", "Enter IBM Quantum token (or leave blank to use QISKIT_IBM_TOKEN env var):", show="*")
            instance = simpledialog.askstring("IBM Quantum", "Optional instance (hub/group/project) or blank:")
            try:
                backends = self.engine.qbackend.connect_ibm(token or "", instance or "")
                if not backends:
                    messagebox.showinfo("IBM Quantum", "Connected, but no backends listed.")
                    return
                name = simpledialog.askstring("IBM Quantum", "Choose backend name:\n" + "\n".join(backends[:30]))
                if name:
                    self.engine.qbackend.select_ibm_backend(name)
                    self._log(f"[IBM] Selected backend: {name}")
            except Exception as e:
                self._log(f"[IBM] Error: {e}")


def run_cli() -> None:
    eng = BrainEngine()
    print("Brain (CLI). Type /help for commands. Ctrl+C to exit.")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            try:
                eng.save()
            except Exception:
                pass
            return
        if not user:
            continue
        try:
            print(eng.respond(user))
        except Exception as e:
            print(f"Error: {e}\n{traceback.format_exc()}")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    if "--cli" in argv or (not _TK_AVAILABLE):
        run_cli()
        return 0

    app_engine = BrainEngine(log_cb=None)
    app = BrainApp(app_engine)
    try:
        app_engine.log_cb = app._log
        if app_engine.semantic is not None:
            app_engine.semantic.log_cb = app._log
    except Exception:
        pass
    app.mainloop()
    try:
        app_engine.save()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
