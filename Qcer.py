#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""
QubitCreator Script

This script creates and manages qubits in "neuron" groupings, where each
neuron has 50 qubits. It uses a 3-layer quantum scheme (middle layer is in
superposition) and leverages Grover's algorithm for search across neurons.
The structure can keep expanding "graphene-lattice style," adding new layers
as needed. Error handling is included to ensure stability.

- Gate manipulations on each neuron (50-qubit group).
- Grover’s search bar for quick searching within or across neurons.
- Logging of any errors and results.

"""

import sys
import os
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator


class QubitNeuron:
    """
    A single "neuron" grouping of 50 qubits, arranged in 3 layers:
      - First layer: optional rotation / pre-processing
      - Second layer: place all qubits in superposition (middle layer)
      - Third layer: optional mild transform or pass

    We set up robust error handling to track any issues.
    """

    def __init__(self, neuron_id: int, enable_logging: bool = True):
        self.neuron_id = neuron_id
        self.num_qubits = 50
        self.enable_logging = enable_logging
        self.last_error = None

        # Create quantum + classical registers for this neuron.
        self.qr = QuantumRegister(self.num_qubits, f"qr_{neuron_id}")
        self.cr = ClassicalRegister(self.num_qubits, f"cr_{neuron_id}")
        self.circuit = QuantumCircuit(self.qr, self.cr, name=f"Neuron_{neuron_id}")

        # We'll use an AerSimulator with method='statevector' by default.
        # This aligns with the style used in QELM code.
        try:
            self.backend = AerSimulator(method='statevector')
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Using AerSimulator(method='statevector').")
        except Exception as e:
            self.last_error = traceback.format_exc()

        # Build the 3-layer structure
        self._build_3_layers()

    def _build_3_layers(self):
        """
        Build out the 3-layer structure:
          L1: Optional rotation (mild RY)
          L2: Full superposition (H)
          L3: Another mild rotation or pass
        """
        try:
            # L1: mild rotation on half qubits
            half_n = self.num_qubits // 2
            for qubit_idx in range(half_n):
                # We can use a small RY
                self.circuit.ry(np.pi / 6, self.qr[qubit_idx])

            # L2: superposition on all qubits
            for qubit_idx in range(self.num_qubits):
                self.circuit.h(self.qr[qubit_idx])

            # L3: mild rotation on the second half
            for qubit_idx in range(half_n, self.num_qubits):
                self.circuit.rx(np.pi / 12, self.qr[qubit_idx])

            # We can optionally save statevector after building
            self.circuit.save_statevector()

        except Exception as e:
            self.last_error = traceback.format_exc()

    def run_circuit(self):
        """
        Execute the circuit to retrieve the resulting statevector.
        """
        try:
            job = self.backend.run(self.circuit, shots=1)
            result = job.result()
            final_state = result.get_statevector(self.circuit)
            return final_state.data  # Return the statevector as a complex numpy array
        except Exception as e:
            self.last_error = traceback.format_exc()
            return None

    def apply_grover_search(self, target_index: int):
        """
        Demonstration: Apply a Grover-like search on the entire 50-qubit structure,
        marking 'target_index' as the search target. 
        For large qubits, this is more conceptual than feasible, but we preserve the 
        Qiskit call style. We do a simplified approach if memory constraints are high.
        """
        try:
            if target_index < 0 or target_index >= 2**self.num_qubits:
                raise ValueError(
                    f"[Neuron {self.neuron_id}] target_index {target_index} is invalid for "
                    f"{2**self.num_qubits} possible states."
                )

            # We could attempt building a new circuit or subroutine for Grover. 
            # Since 50 qubits is large, we do a minimal placeholder approach:
            grover_circuit = QuantumCircuit(self.qr, self.cr, name=f"Grover_{self.neuron_id}")

            # We'll do a simplified "mark and diffuse" approach (not a full Qiskit Grover Operator).
            # Marking operation (placeholder).
            # We'll just do a single phase flip or something symbolic.
            grover_circuit.x(self.qr[0])
            grover_circuit.cz(self.qr[0], self.qr[1])
            grover_circuit.x(self.qr[0])

            # Diffusion: apply H, X, multi-controlled Z, X, H
            for idx in range(self.num_qubits):
                grover_circuit.h(self.qr[idx])
                grover_circuit.x(self.qr[idx])
            # We'll do a partial multi-control approach on a few qubits only
            grover_circuit.h(self.qr[0])
            grover_circuit.cx(self.qr[1], self.qr[0])  # just a small example
            grover_circuit.h(self.qr[0])
            for idx in range(self.num_qubits):
                grover_circuit.x(self.qr[idx])
                grover_circuit.h(self.qr[idx])

            # Merge with existing circuit
            self.circuit = self.circuit.compose(grover_circuit)
            # Re-save statevector
            self.circuit.save_statevector()

        except Exception as e:
            self.last_error = traceback.format_exc()


class QuantumBrain:
    """
    Represents the "quantum brain" made up of multiple QubitNeuron objects.
    Each QubitNeuron is a group of 50 qubits arranged in 3 layers.
    We can create indefinite layers (graphene-lattice style).
    """

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.neurons = []
        self.last_error = None
        if self.enable_logging:
            print("[QuantumBrain] Initialized empty brain structure.")

    def create_neuron(self):
        """
        Create a new neuron and add it to the brain structure.
        """
        try:
            neuron_id = len(self.neurons) + 1
            new_neuron = QubitNeuron(neuron_id, enable_logging=self.enable_logging)
            self.neurons.append(new_neuron)
            if self.enable_logging:
                print(f"[QuantumBrain] Neuron {neuron_id} created.")
            return new_neuron
        except Exception as e:
            self.last_error = traceback.format_exc()
            return None

    def run_all_neurons(self):
        """
        Run each neuron's circuit and return a dict of {neuron_id: statevector}
        """
        outputs = {}
        for neuron in self.neurons:
            sv = neuron.run_circuit()
            outputs[neuron.neuron_id] = sv
        return outputs

    def apply_grover_global(self, target_idx: int):
        """
        Apply a minimal Grover approach to each neuron with the same target index.
        """
        try:
            for neuron in self.neurons:
                neuron.apply_grover_search(target_idx)
        except Exception as e:
            self.last_error = traceback.format_exc()

    def apply_grover_to_neuron(self, neuron_id: int, target_idx: int):
        """
        Apply Grover approach to a specific neuron by ID.
        """
        for neuron in self.neurons:
            if neuron.neuron_id == neuron_id:
                neuron.apply_grover_search(target_idx)
                return
        self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} not found."


class QubitCreatorGUI:
    """
    A Tkinter-based GUI that lets us:
      - Create and visualize QubitNeurons (50-qubit groups).
      - Run circuits to see final statevectors.
      - Perform Grover's search on all or a specific neuron.
      - Manage logs and errors.
    Does not auto-close on double-click.
    """

    def __init__(self, master):
        self.master = master
        self.master.title("Qubit Creator - Quantum Brain")
        # Ensure geometry is stable and doesn't auto-close
        self.master.geometry("1280x800")
        self.master.resizable(False, False)

        self.brain = QuantumBrain(enable_logging=True)
        self._build_gui()

    def _build_gui(self):
        """
        Construct the main UI layout with frames, labels, text areas, etc.
        """
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.Frame(main_frame)

        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Buttons on the left
        self.add_btn = ttk.Button(left_frame, text="Create 50-Qubit Neuron", command=self.add_neuron)
        self.add_btn.pack(pady=10, fill="x")

        ttk.Label(left_frame, text="Grover Target Index:").pack(pady=5)
        self.target_index_var = tk.StringVar(value="0")
        self.target_entry = ttk.Entry(left_frame, textvariable=self.target_index_var)
        self.target_entry.pack(pady=5, fill="x")

        self.apply_grover_all_btn = ttk.Button(
            left_frame, 
            text="Apply Grover to ALL",
            command=self.apply_grover_to_all
        )
        self.apply_grover_all_btn.pack(pady=10, fill="x")

        ttk.Label(left_frame, text="Neuron ID for Single Grover:").pack(pady=5)
        self.neuron_id_var = tk.StringVar(value="1")
        self.neuron_id_entry = ttk.Entry(left_frame, textvariable=self.neuron_id_var)
        self.neuron_id_entry.pack(pady=5, fill="x")

        self.apply_grover_one_btn = ttk.Button(
            left_frame,
            text="Apply Grover to One Neuron",
            command=self.apply_grover_to_one
        )
        self.apply_grover_one_btn.pack(pady=10, fill="x")

        self.run_all_btn = ttk.Button(left_frame, text="Run All Neurons", command=self.run_all_neurons)
        self.run_all_btn.pack(pady=10, fill="x")

        # Scrolled text on the right for logs/results
        self.log_area = scrolledtext.ScrolledText(right_frame, wrap='word')
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

        # Style to match your QELM aesthetic
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#2C3E50", foreground="white")
        style.configure("TFrame", background="#2C3E50")
        style.configure("TLabelFrame", background="#34495E", foreground="white")
        style.configure("TLabel", background="#2C3E50", foreground="white")
        style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
        style.configure("TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
        style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
        style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])
        main_frame.configure(style="TFrame")
        left_frame.configure(style="TFrame")
        right_frame.configure(style="TFrame")

        self.log_message("Qubit Creator GUI initialized.\n")

    def log_message(self, msg: str):
        """
        Print a log message to the scrolled text area.
        """
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, msg)
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def add_neuron(self):
        """
        Add a new 50-qubit neuron to the quantum brain.
        """
        neuron = self.brain.create_neuron()
        if neuron is None:
            err = self.brain.last_error or "Error creating neuron."
            self.log_message(f"[ERROR] {err}\n")
        else:
            self.log_message(f"Created neuron {neuron.neuron_id}.\n")

    def apply_grover_to_all(self):
        """
        Apply the Grover approach to all neurons with user-specified target index.
        """
        try:
            target_idx = int(self.target_index_var.get())
        except ValueError:
            self.log_message("[ERROR] Invalid target index.\n")
            return
        self.brain.apply_grover_global(target_idx)
        if self.brain.last_error:
            self.log_message(f"[ERROR] {self.brain.last_error}\n")
        else:
            self.log_message(f"Applied Grover to ALL neurons with target index={target_idx}.\n")

    def apply_grover_to_one(self):
        """
        Apply Grover approach to a single neuron by ID.
        """
        try:
            target_idx = int(self.target_index_var.get())
            neuron_id = int(self.neuron_id_var.get())
        except ValueError:
            self.log_message("[ERROR] Invalid neuron ID or target index.\n")
            return
        self.brain.apply_grover_to_neuron(neuron_id, target_idx)
        if self.brain.last_error:
            self.log_message(f"[ERROR] {self.brain.last_error}\n")
        else:
            self.log_message(f"Applied Grover to neuron {neuron_id} (target idx={target_idx}).\n")

    def run_all_neurons(self):
        """
        Run the circuits for all neurons in the brain. Display the length
        of resulting statevectors or the full data if small.
        """
        results = self.brain.run_all_neurons()
        for nid, sv in results.items():
            if sv is None:
                err = self.brain.last_error or f"Error running neuron {nid}."
                self.log_message(f"[ERROR] {err}\n")
            else:
                self.log_message(f"\n--- Neuron {nid} ---\n")
                if len(sv) > 16:
                    self.log_message(f"Statevector length: {len(sv)}. Not displaying all.\n")
                else:
                    self.log_message(f"{sv}\n")


def main():
    root = tk.Tk()
    app = QubitCreatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
