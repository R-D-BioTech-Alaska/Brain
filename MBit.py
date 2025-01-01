#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""
Qubit Script

This script enables users to encode multiple instructions within qubit states
by manipulating their amplitude and phase. It offers a Tkinter-based GUI for
managing qubits, encoding instructions, running simulations, and viewing results.

Key Features:
    - Configure the number of qubits per neuron.
    - Select simulation methods to manage memory usage.
    - Encode and decode instructions using amplitude and phase.
    - Create and delete neurons.
    - Visualize qubit states on the Bloch sphere.
    - Robust error handling and real-time logging.

Requirements:
    - Python 3.11+
    - Qiskit >= 0.39.0
    - qiskit-aer
    - tkinter
"""

import sys
import os
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt


class InstructionEncoder:
    """
    Encodes and decodes instructions into qubit states using amplitude and phase encoding.
    """
    
    def __init__(self):
        self.instruction_map = {}
        self.inverse_map = {}
        self.define_instruction_map()
    
    def define_instruction_map(self):
        """
        Define a mapping from instructions to (amplitude, phase) tuples.
        """
        # Example: Mapping 10 instructions for demonstration.
        # Expand this as needed up to 999 instructions.
        for i in range(1, 11):
            instruction = f"Inst_{i}"
            amplitude = np.sqrt(i * 0.09)  # Example amplitude scaling: ensures sum < 1
            phase = (i * np.pi) / 5  # Example phase scaling
            self.instruction_map[instruction] = (amplitude, phase)
            self.inverse_map[(amplitude, phase)] = instruction
    
    def encode_instruction(self, instruction: str):
        """
        Given an instruction, return the (theta, phi) angles to prepare the qubit.
        """
        if instruction not in self.instruction_map:
            raise ValueError(f"Instruction '{instruction}' not defined.")
        amplitude, phase = self.instruction_map[instruction]
        theta = 2 * np.arccos(amplitude)  # Polar angle for RY gate
        phi = phase  # Phase angle for RZ gate
        return theta, phi
    
    def decode_statevector(self, statevector: np.ndarray):
        """
        Given a statevector, decode it to the corresponding instruction.
        """
        # Extract alpha and beta
        alpha = statevector[0]
        beta = statevector[1]
        amplitude_alpha = np.abs(alpha)
        amplitude_beta = np.abs(beta)
        phase_beta = np.angle(beta)
        
        # Find the closest matching instruction
        min_distance = float('inf')
        closest_instruction = None
        for (amp, ph), instr in self.inverse_map.items():
            distance = np.sqrt((amplitude_alpha - amp)**2 + (phase_beta - ph)**2)
            if distance < min_distance:
                min_distance = distance
                closest_instruction = instr
        return closest_instruction


class QubitNeuron:
    """
    Represents a single qubit with the ability to encode and decode instructions.
    """
    
    def __init__(self, neuron_id: int, encoder: InstructionEncoder, sim_method: str = 'statevector', enable_logging: bool = True):
        self.neuron_id = neuron_id
        self.encoder = encoder
        self.sim_method = sim_method
        self.enable_logging = enable_logging
        self.last_error = None
        
        # Initialize registers
        self.qr = QuantumRegister(1, f"qr_{neuron_id}")
        self.cr = ClassicalRegister(1, f"cr_{neuron_id}")
        self.circuit = QuantumCircuit(self.qr, self.cr, name=f"Neuron_{neuron_id}")
        
        # Initialize simulator
        try:
            self.backend = Aer.get_backend('aer_simulator')
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Initialized with simulator '{self.sim_method}'.")
        except Exception:
            self.last_error = traceback.format_exc()
        
    def prepare_instruction(self, instruction: str):
        """
        Prepare the qubit state corresponding to the given instruction.
        """
        try:
            theta, phi = self.encoder.encode_instruction(instruction)
            self.circuit.reset(self.qr[0])
            self.circuit.ry(theta, self.qr[0])
            self.circuit.rz(phi, self.qr[0])
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Prepared state for '{instruction}' with theta={theta:.4f}, phi={phi:.4f}.")
        except Exception:
            self.last_error = traceback.format_exc()
    
    def run_circuit(self):
        """
        Execute the circuit and return the statevector.
        """
        try:
            # Simulate the circuit
            qc = self.circuit.copy()
            qc.save_statevector()
            job = self.backend.run(qc)
            result = job.result()
            statevector = result.get_statevector(qc)
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Circuit run successfully.")
            return statevector.data
        except Exception:
            self.last_error = traceback.format_exc()
            return None
    
    def decode_instruction(self, statevector: np.ndarray):
        """
        Decode the statevector back to an instruction.
        """
        try:
            instruction = self.encoder.decode_statevector(statevector)
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Decoded instruction: '{instruction}'.")
            return instruction
        except Exception:
            self.last_error = traceback.format_exc()
            return None


class QuantumBrain:
    """
    Manages multiple QubitNeurons, allowing creation, deletion,
    encoding, decoding, and simulation of instructions.
    """
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.neurons = {}
        self.encoder = InstructionEncoder()
        self.last_error = None
        if self.enable_logging:
            print("[QuantumBrain] Initialized empty brain structure.")
    
    def create_neuron(self, sim_method: str = 'statevector'):
        """
        Create a new neuron with a unique ID.
        """
        try:
            neuron_id = len(self.neurons) + 1
            neuron = QubitNeuron(neuron_id, self.encoder, sim_method, self.enable_logging)
            self.neurons[neuron_id] = neuron
            if self.enable_logging:
                print(f"[QuantumBrain] Neuron {neuron_id} created.")
            return neuron_id
        except Exception:
            self.last_error = traceback.format_exc()
            return None
    
    def delete_neuron(self, neuron_id: int):
        """
        Delete a neuron by its ID.
        """
        try:
            if neuron_id in self.neurons:
                del self.neurons[neuron_id]
                if self.enable_logging:
                    print(f"[QuantumBrain] Neuron {neuron_id} deleted.")
                return True
            else:
                self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} does not exist."
                return False
        except Exception:
            self.last_error = traceback.format_exc()
            return False
    
    def encode_instruction(self, neuron_id: int, instruction: str):
        """
        Encode an instruction into a specific neuron.
        """
        try:
            if neuron_id in self.neurons:
                self.neurons[neuron_id].prepare_instruction(instruction)
                return True
            else:
                self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} does not exist."
                return False
        except Exception:
            self.last_error = traceback.format_exc()
            return False
    
    def run_neuron(self, neuron_id: int):
        """
        Run a specific neuron and decode its instruction.
        """
        try:
            if neuron_id in self.neurons:
                statevector = self.neurons[neuron_id].run_circuit()
                if statevector is not None:
                    instruction = self.neurons[neuron_id].decode_instruction(statevector)
                    return instruction
                else:
                    return None
            else:
                self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} does not exist."
                return None
        except Exception:
            self.last_error = traceback.format_exc()
            return None
    
    def run_all_neurons(self):
        """
        Run all neurons and return their decoded instructions.
        """
        results = {}
        for neuron_id, neuron in self.neurons.items():
            statevector = neuron.run_circuit()
            if statevector is not None:
                instruction = neuron.decode_instruction(statevector)
                results[neuron_id] = instruction
            else:
                results[neuron_id] = None
        return results
    
    def update_neuron_selection(self):
        """
        Update the neuron selection in the GUI.
        """
        return list(self.neurons.keys())


class QubitCreatorGUI:
    """
    A robust Tkinter-based GUI for managing QuantumBrain, allowing users to:
      - Create and delete neurons.
      - Encode instructions into neurons.
      - Run simulations and view decoded instructions.
      - Edit various parameters such as qubit count, simulation method, and rotation angles.
      - Visualize qubit states.
    """
    
    def __init__(self, master):
        self.master = master
        self.master.title("Qubit Creator - Quantum Brain")
        self.master.geometry("1600x900")
        self.master.resizable(False, False)
        
        self.brain = QuantumBrain(enable_logging=True)
        
        # GUI Variables
        self.selected_neuron_var = tk.IntVar()
        self.instruction_var = tk.StringVar()
        self.run_status_var = tk.StringVar(value="Ready")
        
        # Build the GUI Layout
        self.build_gui()
    
    def build_gui(self):
        """
        Constructs the main GUI layout.
        """
        # Main Frames
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        log_frame = ttk.Frame(self.master)
        log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Control Panel Components
        # 1. Neuron Management
        neuron_mgmt = ttk.LabelFrame(control_frame, text="Neuron Management")
        neuron_mgmt.pack(fill="x", pady=5)
        
        create_btn = ttk.Button(neuron_mgmt, text="Create Neuron", command=self.create_neuron)
        create_btn.pack(fill="x", padx=5, pady=5)
        
        delete_lbl = ttk.Label(neuron_mgmt, text="Delete Neuron ID:")
        delete_lbl.pack(padx=5, pady=5, anchor="w")
        self.delete_entry = ttk.Entry(neuron_mgmt, textvariable=self.selected_neuron_var)
        self.delete_entry.pack(fill="x", padx=5, pady=5)
        delete_btn = ttk.Button(neuron_mgmt, text="Delete Neuron", command=self.delete_neuron)
        delete_btn.pack(fill="x", padx=5, pady=5)
        
        # 2. Instruction Encoding
        instr_frame = ttk.LabelFrame(control_frame, text="Instruction Encoding")
        instr_frame.pack(fill="x", pady=5)
        
        neuron_select_lbl = ttk.Label(instr_frame, text="Neuron ID:")
        neuron_select_lbl.pack(padx=5, pady=5, anchor="w")
        self.neuron_select_var = tk.IntVar()
        self.neuron_select_combo = ttk.Combobox(instr_frame, textvariable=self.neuron_select_var, state='readonly')
        self.neuron_select_combo.pack(fill="x", padx=5, pady=5)
        self.update_neuron_selection()
        
        instruction_lbl = ttk.Label(instr_frame, text="Instruction:")
        instruction_lbl.pack(padx=5, pady=5, anchor="w")
        self.instruction_entry = ttk.Entry(instr_frame, textvariable=self.instruction_var)
        self.instruction_entry.pack(fill="x", padx=5, pady=5)
        encode_btn = ttk.Button(instr_frame, text="Encode Instruction", command=self.encode_instruction)
        encode_btn.pack(fill="x", padx=5, pady=5)
        
        # 3. Simulation Controls
        sim_frame = ttk.LabelFrame(control_frame, text="Simulation Controls")
        sim_frame.pack(fill="x", pady=5)
        
        run_all_btn = ttk.Button(sim_frame, text="Run All Neurons", command=self.run_all_neurons)
        run_all_btn.pack(fill="x", padx=5, pady=5)
        
        run_status_lbl = ttk.Label(sim_frame, textvariable=self.run_status_var)
        run_status_lbl.pack(padx=5, pady=5)
        
        # 4. Parameter Editing
        param_frame = ttk.LabelFrame(control_frame, text="Parameter Editing")
        param_frame.pack(fill="x", pady=5)
        
        # Editing Rotation Angles
        ttk.Label(param_frame, text="Neuron ID to Edit:").pack(padx=5, pady=5, anchor="w")
        self.edit_neuron_id_var = tk.IntVar()
        self.edit_neuron_id_entry = ttk.Entry(param_frame, textvariable=self.edit_neuron_id_var)
        self.edit_neuron_id_entry.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Layer1 RY Angle (rad):").pack(padx=5, pady=5, anchor="w")
        self.edit_angle_l1_var = tk.DoubleVar()
        self.edit_angle_l1_entry = ttk.Entry(param_frame, textvariable=self.edit_angle_l1_var)
        self.edit_angle_l1_entry.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Layer3 RX Angle (rad):").pack(padx=5, pady=5, anchor="w")
        self.edit_angle_l3_var = tk.DoubleVar()
        self.edit_angle_l3_entry = ttk.Entry(param_frame, textvariable=self.edit_angle_l3_var)
        self.edit_angle_l3_entry.pack(fill="x", padx=5, pady=5)
        
        update_param_btn = ttk.Button(param_frame, text="Update Neuron Parameters", command=self.update_neuron_parameters)
        update_param_btn.pack(fill="x", padx=5, pady=5)
        
        # 5. Visualization
        viz_frame = ttk.LabelFrame(control_frame, text="Qubit Visualization")
        viz_frame.pack(fill="x", pady=5)
        
        viz_neuron_lbl = ttk.Label(viz_frame, text="Neuron ID to Visualize:")
        viz_neuron_lbl.pack(padx=5, pady=5, anchor="w")
        self.viz_neuron_id_var = tk.IntVar()
        self.viz_neuron_id_entry = ttk.Entry(viz_frame, textvariable=self.viz_neuron_id_var)
        self.viz_neuron_id_entry.pack(fill="x", padx=5, pady=5)
        
        viz_btn = ttk.Button(viz_frame, text="Visualize Qubit State", command=self.visualize_qubit)
        viz_btn.pack(fill="x", padx=5, pady=5)
        
        # Log Frame Components
        log_label = ttk.Label(log_frame, text="Logs and Outputs:")
        log_label.pack(anchor="w")
        
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap='word', state='disabled')
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Style Configuration
        self.configure_styles()
    
    def configure_styles(self):
        """
        Configures the styles for the GUI elements.
        """
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#2C3E50", foreground="white")
        style.configure("TLabelFrame", background="#34495E", foreground="white")
        style.configure("TLabel", background="#2C3E50", foreground="white")
        style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
        style.configure("TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
        style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
        style.map("TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#1F2A36')])
    
    def log_message(self, msg: str):
        """
        Logs a message to the scrolled text area.
        """
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, msg)
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
    
    def update_neuron_selection(self):
        """
        Updates the neuron selection dropdown based on existing neurons.
        """
        neuron_ids = self.brain.update_neuron_selection()
        self.neuron_select_combo['values'] = neuron_ids
        if neuron_ids:
            self.neuron_select_combo.current(0)
        else:
            self.neuron_select_combo.set('')
    
    def create_neuron(self):
        """
        Handles the creation of a new neuron.
        """
        neuron_id = self.brain.create_neuron()
        if neuron_id is not None:
            self.log_message(f"Created Neuron {neuron_id}.\n")
            self.update_neuron_selection()
        else:
            self.log_message(f"Failed to create neuron: {self.brain.last_error}\n")
    
    def delete_neuron(self):
        """
        Handles the deletion of an existing neuron.
        """
        try:
            neuron_id = self.selected_neuron_var.get()
            if not neuron_id:
                raise ValueError("No Neuron ID entered.")
            success = self.brain.delete_neuron(neuron_id)
            if success:
                self.log_message(f"Deleted Neuron {neuron_id}.\n")
                self.update_neuron_selection()
            else:
                self.log_message(f"Failed to delete Neuron {neuron_id}: {self.brain.last_error}\n")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")
    
    def encode_instruction(self):
        """
        Encodes an instruction into the selected neuron.
        """
        try:
            neuron_id = self.neuron_select_var.get()
            instruction = self.instruction_var.get().strip()
            if not instruction:
                raise ValueError("No instruction entered.")
            success = self.brain.encode_instruction(neuron_id, instruction)
            if success:
                self.log_message(f"Encoded instruction '{instruction}' into Neuron {neuron_id}.\n")
            else:
                self.log_message(f"Failed to encode instruction: {self.brain.last_error}\n")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")
    
    def run_all_neurons(self):
        """
        Runs all neurons and displays their decoded instructions.
        """
        self.run_status_var.set("Running...")
        self.master.update_idletasks()
        try:
            results = self.brain.run_all_neurons()
            for neuron_id, instruction in results.items():
                if instruction:
                    self.log_message(f"Neuron {neuron_id}: '{instruction}'\n")
                else:
                    self.log_message(f"Neuron {neuron_id}: Failed to decode instruction.\n")
            self.run_status_var.set("Run Completed.")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")
            self.run_status_var.set("Run Failed.")
    
    def update_neuron_parameters(self):
        """
        Updates rotation angles for a specific neuron.
        """
        try:
            neuron_id = self.edit_neuron_id_var.get()
            angle_l1 = self.edit_angle_l1_var.get()
            angle_l3 = self.edit_angle_l3_var.get()
            if neuron_id not in self.brain.neurons:
                raise ValueError(f"Neuron ID {neuron_id} does not exist.")
            neuron = self.brain.neurons[neuron_id]
            neuron.angle_l1 = angle_l1
            neuron.angle_l3 = angle_l3
            # Rebuild the circuit with new angles
            neuron._build_3_layers()
            self.log_message(f"Updated Neuron {neuron_id} parameters: angle_l1={angle_l1}, angle_l3={angle_l3}.\n")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")
    
    def visualize_qubit(self):
        """
        Visualizes the state of a specific qubit on the Bloch sphere.
        """
        try:
            neuron_id = self.viz_neuron_id_var.get()
            if neuron_id not in self.brain.neurons:
                raise ValueError(f"Neuron ID {neuron_id} does not exist.")
            neuron = self.brain.neurons[neuron_id]
            statevector = neuron.run_circuit()
            if statevector is not None:
                sv = Statevector(statevector)
                plot_bloch_multivector(sv)
                plt.title(f"Neuron {neuron_id} Qubit State on Bloch Sphere")
                plt.show()
                self.log_message(f"Visualized Neuron {neuron_id} state on Bloch sphere.\n")
            else:
                self.log_message(f"Failed to visualize Neuron {neuron_id}: {neuron.last_error}\n")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")
