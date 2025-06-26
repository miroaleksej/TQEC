# tqec_enhanced.py
import numpy as np
import cupy as cp
from mpi4py import MPI
from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from numba import cuda
import matplotlib.pyplot as plt
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from qiskit import Aer, IBMQ, QuantumCircuit, execute, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import QuantumInstance
from qiskit_nature.drivers import Molecule
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Конфигурация системы
@dataclass
class TQECConfig:
    quantum_units: int = 512
    tunnel_units: int = 256
    max_steps: int = 1000
    energy_threshold: float = 1e-6
    use_gpu: bool = True
    use_braket: bool = False
    use_tfq: bool = False
    use_qiskit: bool = False
    use_real_quantum: bool = False
    quantum_backend: str = 'aer_simulator'
    visualize: bool = False
    save_results: bool = True
    verbose: bool = True
    qiskit_token: Optional[str] = None
    error_correction_type: str = 'enhanced'
    distributed_computing: bool = False

class QuantumAlgorithms:
    @staticmethod
    def grover_search(oracle, num_qubits, iterations=None):
        """Реализация алгоритма Гровера"""
        circuit = QuantumCircuit(num_qubits)
        # Инициализация равномерной суперпозиции
        circuit.h(range(num_qubits))
        
        # Определение оптимального числа итераций
        if iterations is None:
            iterations = int(np.pi/4 * np.sqrt(2**num_qubits))
        
        # Применение оракула и диффузионного оператора
        for _ in range(iterations):
            circuit.append(oracle, range(num_qubits))
            circuit.h(range(num_qubits))
            circuit.x(range(num_qubits))
            circuit.h(num_qubits-1)
            circuit.mct(list(range(num_qubits-1)), num_qubits-1)
            circuit.h(num_qubits-1)
            circuit.x(range(num_qubits))
            circuit.h(range(num_qubits))
        
        return circuit

    @staticmethod
    def qft(num_qubits):
        """Квантовое преобразование Фурье"""
        circuit = QuantumCircuit(num_qubits)
        
        for j in range(num_qubits):
            circuit.h(j)
            for k in range(j+1, num_qubits):
                circuit.cp(np.pi/2**(k-j), k, j)
        
        # Добавляем переворот битов для правильного порядка
        for qubit in range(num_qubits//2):
            circuit.swap(qubit, num_qubits-qubit-1)
            
        return circuit

class ExtendedGates:
    @staticmethod
    def add_to_circuit(circuit, gate_name, qubits, params=None):
        """Добавление расширенных гейтов в схему"""
        if gate_name == 'crx':
            circuit.crx(params[0], qubits[0], qubits[1])
        elif gate_name == 'cry':
            circuit.cry(params[0], qubits[0], qubits[1])
        elif gate_name == 'crz':
            circuit.crz(params[0], qubits[0], qubits[1])
        elif gate_name == 'cu1':
            circuit.cu1(params[0], qubits[0], qubits[1])
        elif gate_name == 'cu3':
            circuit.cu3(*params, qubits[0], qubits[1])
        elif gate_name == 'iswap':
            circuit.iswap(qubits[0], qubits[1])
        elif gate_name == 'ms':
            circuit.ms(*params, qubits)
        else:
            raise ValueError(f"Неизвестный гейт: {gate_name}")

class EnhancedErrorCorrection:
    def __init__(self, code_type='surface', error_rates=None):
        self.code_type = code_type
        self.redundancy = 3 if code_type == 'surface' else 5
        self.error_rates = error_rates or {'readout': 0.01, 'gate': 0.005, 'thermal': 0.001}
        
    def __call__(self, quantum_state, tunnel_probs):
        return (
            self._correct_quantum(quantum_state),
            self._correct_tunnel(tunnel_probs)
        )

    def _correct_quantum(self, state):
        if isinstance(state, cp.ndarray):
            correction_matrix = self._build_correction_matrix()
            corrected = cp.einsum('ij,kj->ki', correction_matrix, state)
            norms = cp.linalg.norm(corrected, axis=1)
            return corrected / norms[:, cp.newaxis]
        else:
            correction_matrix = self._build_correction_matrix()
            corrected = np.einsum('ij,kj->ki', correction_matrix, state)
            norms = np.linalg.norm(corrected, axis=1)
            return corrected / norms[:, np.newaxis]
    
    def _build_correction_matrix(self):
        p_read = self.error_rates['readout']
        p_gate = self.error_rates['gate']
        p_therm = self.error_rates['thermal']
        
        diag = 1 - p_read - 2*p_gate - p_therm
        off_diag = p_read/3 + p_gate + p_therm/2
        
        return np.array([
            [diag, off_diag, off_diag, off_diag],
            [off_diag, diag, off_diag, off_diag],
            [off_diag, off_diag, diag, off_diag],
            [off_diag, off_diag, off_diag, diag]
        ])

    def _correct_tunnel(self, probs):
        if isinstance(probs, cp.ndarray):
            reshaped = probs.reshape(self.redundancy, -1)
            return cp.median(reshaped, axis=0)
        reshaped = probs.reshape(self.redundancy, -1)
        return np.median(reshaped, axis=0)

class RealDeviceManager:
    def __init__(self, config):
        self.config = config
        self.backends = {}
        self._init_providers()
        
    def _init_providers(self):
        if self.config.use_qiskit and self.config.qiskit_token:
            IBMQ.save_account(self.config.qiskit_token)
            IBMQ.load_account()
            self.backends['ibm'] = IBMQ.get_provider()
            
        if self.config.use_braket:
            self.backends['braket'] = AwsDevice.get_devices()
            
    def get_device(self, provider, requirements=None):
        if provider == 'ibm':
            return self._select_ibm_device(requirements)
        elif provider == 'braket':
            return self._select_braket_device(requirements)
        else:
            raise ValueError(f"Неизвестный провайдер: {provider}")
            
    def _select_ibm_device(self, requirements):
        backend = None
        if requirements.get('real', False):
            backends = self.backends['ibm'].backends(
                simulator=False,
                operational=True,
                min_qubits=requirements.get('min_qubits', 5)
            )
            backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
        else:
            backend = Aer.get_backend('aer_simulator')
        return backend
        
    def _select_braket_device(self, requirements):
        if requirements.get('real', False):
            return AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        else:
            return AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-11")

class QMLModels:
    @staticmethod
    def quantum_neural_network(num_qubits, num_layers, observables):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        circuit = cirq.Circuit()
        
        for _ in range(num_layers):
            circuit += cirq.H.on_each(qubits)
            for i in range(num_qubits-1):
                circuit += cirq.CNOT(qubits[i], qubits[i+1])
            for i, qubit in enumerate(qubits):
                circuit += cirq.rz(0.1).on(qubit)
                circuit += cirq.rx(0.2).on(qubit)
        
        return tfq.layers.PQC(circuit, observables)
        
    @staticmethod
    def quantum_embedding_layer(input_dim, num_qubits):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        circuit = cirq.Circuit()
        
        for i in range(num_qubits):
            circuit += cirq.rx(np.pi/input_dim).on(qubits[i])
            circuit += cirq.rz(np.pi/input_dim).on(qubits[i])
        
        return tfq.layers.PQC(circuit, operators=[])

class QuantumMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics_history = {
            'energy': [],
            'fidelity': [],
            'tunnel_mean': [],
            'tunnel_std': []
        }
        
    def log_step(self, metrics):
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
                
        if self.config.verbose and metrics.get('step', 0) % 10 == 0:
            self._print_status(metrics)
            
    def _print_status(self, metrics):
        print(f"\nШаг {metrics['step']}:")
        print(f"  Энергия: {metrics['energy']:.4f}")
        print(f"  Фидельность: {metrics['fidelity']:.4f}")
        print(f"  Туннелирование (среднее): {metrics['tunnel_mean']:.4f}")
        
    def save_logs(self, filename='quantum_logs.json'):
        with open(filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def plot_history(self):
        plt.figure(figsize=(12, 8))
        
        for i, (metric, values) in enumerate(self.metrics_history.items()):
            plt.subplot(2, 2, i+1)
            plt.plot(values)
            plt.title(metric)
            plt.xlabel('Шаг')
            
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

class QuantumVisualizer:
    @staticmethod
    def plot_quantum_state(state, title="Квантовое состояние"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if isinstance(state, cp.ndarray):
            state = cp.asnumpy(state)
            
        theta = 2 * np.arccos(state[:, 0])
        phi = np.angle(state[:, 0] + 1j*state[:, 1])
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        ax.scatter(x, y, z, alpha=0.6)
        ax.set_title(title)
        plt.savefig('quantum_state.png')
        plt.close()
        
    @staticmethod
    def plot_circuit(circuit, filename="circuit.png"):
        if isinstance(circuit, QuantumCircuit):
            circuit.draw(output='mpl').savefig(filename)
        elif isinstance(circuit, cirq.Circuit):
            svg = cirq.Circuit.to_text_diagram(circuit)
            with open(filename.replace('.png', '.txt'), 'w') as f:
                f.write(svg)

class DistributedQuantum:
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
    def distribute_circuit(self, circuit, num_qubits):
        qubits_per_node = num_qubits // self.size
        my_qubits = range(
            self.rank * qubits_per_node,
            (self.rank + 1) * qubits_per_node
        )
        
        subcircuit = circuit.copy()
        subcircuit.remove_qubits(*[q for q in range(num_qubits) if q not in my_qubits])
        
        return subcircuit
        
    def gather_results(self, local_results):
        all_results = self.comm.gather(local_results, root=0)
        
        if self.rank == 0:
            combined = {}
            for res in all_results:
                combined.update(res)
            return combined
        return None

class TQECSystem:
    def __init__(self, config: TQECConfig):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.config = config
        
        self.last_energy = float('inf')
        self.convergence_history = []
        self.current_params = {}
        
        self._init_hardware()
        
        # Инициализация улучшенных компонентов
        self.quantum_algorithms = QuantumAlgorithms()
        self.device_manager = RealDeviceManager(config) if config.use_real_quantum else None
        self.error_corrector = EnhancedErrorCorrection(config.error_correction_type)
        self.optimizer = AdaptiveAllocator(self.config)
        self.monitor = QuantumMonitor(config)
        self.distributor = DistributedQuantum(self.comm) if config.distributed_computing else None
        
        if self.config.use_braket:
            self.braket_executor = BraketExecutor()
        
        if self.config.use_tfq:
            self.tfq_model = TFQHybridModel()
            self.qml_models = QMLModels()
            
        if self.config.use_qiskit:
            self._init_qiskit()

    def _init_hardware(self):
        self.local_quantum_size = self.config.quantum_units // self.size
        self.local_tunnel_size = self.config.tunnel_units // self.size
        
        if self.config.use_gpu:
            self.quantum_state = cp.zeros((self.local_quantum_size, 2), dtype=cp.complex128)
            self.tunnel_probs = cp.random.uniform(0, 0.2, size=self.local_tunnel_size)
        else:
            self.quantum_state = np.zeros((self.local_quantum_size, 2), dtype=np.complex128)
            self.tunnel_probs = np.random.uniform(0, 0.2, size=self.local_tunnel_size)
        
        self._sync_states()

    def _sync_states(self):
        if self.rank == 0:
            if self.config.use_gpu:
                ref_quantum = cp.random.randn(self.config.quantum_units, 2) + 1j*cp.random.randn(self.config.quantum_units, 2)
                ref_quantum /= cp.linalg.norm(ref_quantum, axis=1)[:, cp.newaxis]
                ref_tunnel = cp.random.uniform(0, 0.2, size=self.config.tunnel_units)
            else:
                ref_quantum = np.random.randn(self.config.quantum_units, 2) + 1j*np.random.randn(self.config.quantum_units, 2)
                ref_quantum /= np.linalg.norm(ref_quantum, axis=1)[:, np.newaxis]
                ref_tunnel = np.random.uniform(0, 0.2, size=self.config.tunnel_units)
        else:
            ref_quantum = None
            ref_tunnel = None
        
        self.comm.Scatter(ref_quantum, self.quantum_state, root=0)
        self.comm.Scatter(ref_tunnel, self.tunnel_probs, root=0)

    def _init_qiskit(self):
        if self.config.use_real_quantum:
            if self.config.qiskit_token:
                IBMQ.save_account(self.config.qiskit_token)
            if not IBMQ.active_account():
                IBMQ.load_account()
            self.provider = IBMQ.get_provider(hub='ibm-q')
            try:
                self.quantum_backend = self.provider.get_backend(self.config.quantum_backend)
                if self.rank == 0 and self.config.verbose:
                    print(f"Подключено к реальному квантовому устройству: {self.quantum_backend}")
            except Exception as e:
                if self.rank == 0:
                    print(f"Ошибка подключения к {self.config.quantum_backend}: {str(e)}, используем симулятор")
                self.quantum_backend = AerSimulator()
        else:
            self.quantum_backend = Aer.get_backend('aer_simulator')
            if self.rank == 0 and self.config.verbose:
                print("Инициализирован Qiskit симулятор")

    def solve(self, problem: Dict) -> Optional[Dict]:
        start_time = time.time()
        
        if self.rank == 0 and
print(f"Запуск решения задачи {problem.get('type', 'unknown')}")
            print(f"Конфигурация: {self.config}")
        
        for step in range(self.config.max_steps):
            # 1. Оптимизация параметров
            params = self.optimizer.optimize(problem, step)
            self._update_params(params)
            
            # 2. Гибридное вычисление с улучшенными алгоритмами
            self._enhanced_quantum_step(problem)
            self._tunnel_step()
            
            # 3. Улучшенная коррекция ошибок
            self.quantum_state, self.tunnel_probs = self.error_corrector(
                self.quantum_state, 
                self.tunnel_probs
            )
            
            # 4. Сбор результатов
            step_result = self._collect_step_results()
            
            # 5. Обработка на главном узле
            if self.rank == 0:
                metrics = self._calculate_metrics(step_result)
                self.convergence_history.append(metrics)
                self.monitor.log_step(metrics)
                
                if self.config.visualize and step % 10 == 0:
                    self._enhanced_visualization(step, step_result)
                
                if self._check_convergence(step_result):
                    if self.config.save_results:
                        self._save_checkpoint(step)
                    break
        
        return self._finalize_results() if self.rank == 0 else None
        
    def _enhanced_quantum_step(self, problem):
        """Улучшенный квантовый шаг с поддержкой новых алгоритмов"""
        if problem.get('algorithm') == 'grover':
            oracle = problem.get('oracle')
            if oracle:
                circuit = self.quantum_algorithms.grover_search(oracle, problem.get('num_qubits', 2))
                problem['qiskit_circuit'] = circuit
                self._run_qiskit_quantum_step(problem)
        elif problem.get('algorithm') == 'qft':
            circuit = self.quantum_algorithms.qft(problem.get('num_qubits', 3))
            problem['qiskit_circuit'] = circuit
            self._run_qiskit_quantum_step(problem)
        elif problem.get('algorithm') == 'vqe':
            self._run_vqe_algorithm(problem)
        else:
            super()._quantum_step(problem)

    def _run_vqe_algorithm(self, problem):
        """Запуск алгоритма VQE для квантовой химии"""
        if 'molecule' not in problem:
            raise ValueError("Для VQE необходимо указать молекулу в problem['molecule']")
            
        molecule = problem['molecule']
        driver = Molecule(
            geometry=[(molecule['atom'], molecule['coords'])],
            charge=molecule.get('charge', 0),
            multiplicity=molecule.get('multiplicity', 1)
        )
        
        es_problem = ElectronicStructureProblem(driver)
        qubit_converter = QubitConverter(JordanWignerMapper())
        quantum_instance = QuantumInstance(self.quantum_backend)
        
        vqe = VQE(
            quantum_instance=quantum_instance,
            optimizer=SPSA(maxiter=100)
        )
        
        result = vqe.compute_minimum_eigenvalue(qubit_converter.convert(es_problem.second_q_ops()[0]))
        
        if self.rank == 0 and self.config.verbose:
            print(f"VQE результат: {result}")

    def _enhanced_visualization(self, step, results):
        """Улучшенная визуализация с новыми методами"""
        QuantumVisualizer.plot_quantum_state(results['quantum'], f"Шаг {step}")
        super()._visualize(step, results)

    def _quantum_step(self, problem: Dict):
        if self.config.use_qiskit and problem.get('use_qiskit', False):
            self._run_qiskit_quantum_step(problem)
        elif self.config.use_braket and problem.get('use_braket', False):
            braket_result = self.braket_executor.run_hybrid_task(
                problem.get('quantum_circuit', []),
                problem.get('classical_params', {})
            )
            self._update_from_braket(braket_result)
        else:
            if self.config.use_gpu:
                threads = 32
                blocks = (self.quantum_state.shape[0] + threads - 1) // threads
                self._quantum_kernel[blocks, threads](self.quantum_state)
            else:
                for i in range(self.quantum_state.shape[0]):
                    self._sx_gate(self.quantum_state[i])
            
            if self.config.use_tfq:
                self._apply_tfq_operations(problem)

    @cuda.jit
    def _quantum_kernel(state):
        i = cuda.grid(1)
        if i < state.shape[0]:
            a = (state[i,0] + 1j*state[i,1]) * (1+1j)/2
            state[i,0], state[i,1] = a.real, a.imag

    def _sx_gate(self, qubit_state):
        sx_matrix = np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2
        qubit_state[:] = sx_matrix.dot(qubit_state)

    def _run_qiskit_quantum_step(self, problem: Dict):
        if 'qiskit_circuit' not in problem:
            raise ValueError("Для использования Qiskit необходимо предоставить qiskit_circuit в problem")
            
        circuit = problem['qiskit_circuit']
        transpiled_circ = transpile(circuit, self.quantum_backend)
        job = execute(transpiled_circ, self.quantum_backend, shots=1024)
        result = job.result()
        
        if hasattr(result, 'get_statevector'):
            statevector = result.get_statevector()
            if self.config.use_gpu:
                self.quantum_state[:len(statevector), 0] = cp.asarray(statevector.real)
                self.quantum_state[:len(statevector), 1] = cp.asarray(statevector.imag)
            else:
                self.quantum_state[:len(statevector), 0] = statevector.real
                self.quantum_state[:len(statevector), 1] = statevector.imag
        elif hasattr(result, 'get_counts'):
            counts = result.get_counts()
            norm = sum(counts.values())**0.5
            for state, count in counts.items():
                idx = int(state, 2)
                if idx < self.quantum_state.shape[0]:
                    prob = (count / 1024)**0.5
                    angle = 2 * np.pi * np.random.random()
                    if self.config.use_gpu:
                        self.quantum_state[idx, 0] = cp.array(prob * np.cos(angle))
                        self.quantum_state[idx, 1] = cp.array(prob * np.sin(angle))
                    else:
                        self.quantum_state[idx, 0] = prob * np.cos(angle)
                        self.quantum_state[idx, 1] = prob * np.sin(angle)

    def _tunnel_step(self):
        if self.config.use_gpu:
            noise = cp.random.uniform(0.9, 1.1, size=self.tunnel_probs.shape)
            self.tunnel_probs = cp.clip(self.tunnel_probs * noise, 0, 1)
        else:
            noise = np.random.uniform(0.9, 1.1, size=self.tunnel_probs.shape)
            self.tunnel_probs = np.clip(self.tunnel_probs * noise, 0, 1)

    def _apply_tfq_operations(self, problem: Dict):
        tfq_input = tfq.convert_to_tensor([
            cirq.Circuit(
                cirq.X(cirq.GridQubit(0, i))**problem.get('tfq_params', 0.5)
                for i in range(min(4, self.quantum_state.shape[0]))
        ])
        
        updates = self.tfq_model(tfq_input).numpy()[0]
        
        if self.config.use_gpu:
            self.quantum_state[:len(updates), 0] = cp.asarray(updates.real)
            self.quantum_state[:len(updates), 1] = cp.asarray(updates.imag)
        else:
            self.quantum_state[:len(updates), 0] = updates.real
            self.quantum_state[:len(updates), 1] = updates.imag

    def _collect_step_results(self) -> Optional[Dict]:
        if self.config.use_gpu:
            quantum_gather = cp.zeros((self.config.quantum_units, 2), dtype=cp.complex128)
            local_quantum = cp.asnumpy(self.quantum_state)
            tunnel_gather = cp.zeros(self.config.tunnel_units, dtype=cp.float32)
            local_tunnel = cp.asnumpy(self.tunnel_probs)
        else:
            quantum_gather = np.zeros((self.config.quantum_units, 2), dtype=np.complex128)
            local_quantum = self.quantum_state
            tunnel_gather = np.zeros(self.config.tunnel_units, dtype=np.float32)
            local_tunnel = self.tunnel_probs
        
        self.comm.Gather(local_quantum, quantum_gather, root=0)
        self.comm.Gather(local_tunnel, tunnel_gather, root=0)
        
        if self.rank == 0:
            return {
                'quantum': quantum_gather,
                'tunnel': tunnel_gather
            }
        return None

    def _calculate_metrics(self, step_result: Dict) -> Dict:
        if self.config.use_gpu:
            energy = float(cp.linalg.norm(step_result['quantum']))
            fidelity = float(cp.mean(cp.abs(step_result['quantum']**2)))
            tunnel_mean = float(cp.mean(step_result['tunnel']))
            tunnel_std = float(cp.std(step_result['tunnel']))
        else:
            energy = float(np.linalg.norm(step_result['quantum']))
            fidelity = float(np.mean(np.abs(step_result['quantum']**2)))
            tunnel_mean = float(np.mean(step_result['tunnel']))
            tunnel_std = float(np.std(step_result['tunnel']))
        
        return {
            'step': len(self.convergence_history),
            'energy': energy,
            'fidelity': fidelity,
            'tunnel_mean': tunnel_mean,
            'tunnel_std': tunnel_std,
            'timestamp': time.time()
        }

    def _check_convergence(self, step_result: Optional[Dict]) -> bool:
        if len(self.convergence_history) < 2:
            return False
            
        current = self.convergence_history[-1]
        previous = self.convergence_history[-2]
        delta = abs(current['energy'] - previous['energy'])
        
        return delta < self.config.energy_threshold

    def _visualize(self, step: int, results: Dict):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        q = results['quantum']
        plt.scatter(q[:,0].real, q[:,1].real, alpha=0.5)
        plt.title(f'Квантовые состояния (шаг {step})')
        
        plt.subplot(1, 3, 2)
        t = results['tunnel']
        plt.hist(t, bins=20, range=(0, 1))
        plt.title('Туннельные вероятности')
        
        plt.subplot(1, 3, 3)
        steps = [m['step'] for m in self.convergence_history]
        energies = [m['energy'] for m in self.convergence_history]
        plt.plot(steps, energies)
        plt.title('Энергия системы')
        plt.xlabel('Шаг')
        plt.ylabel('Энергия')
        
        plt.tight_layout()
        plt.savefig(f'step_{step:04d}.png')
        plt.close()

    def _save_checkpoint(self, step: int):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        
        checkpoint = {
            'step': step,
            'quantum_state': self.quantum_state,
            'tunnel_probs': self.tunnel_probs,
            'convergence': self.convergence_history,
            'config': vars(self.config)
        }
        
        np.savez_compressed(
            f'checkpoints/step_{step:04d}.npz',
            **checkpoint
        )

    def _finalize_results(self) -> Dict:
        final_metrics = self.convergence_history[-1]
        final_state = self._collect_step_results()
        
        result = {
            'status': 'converged' if self._check_convergence(None) else 'max_steps_reached',
            'metrics': final_metrics,
            'states': final_state,
            'convergence_history': self.convergence_history,
            'config': vars(self.config)
        }
        
        if self.config.save_results:
            self._save_final_results(result)
            
        return result

    def _save_final_results(self, results: Dict):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        with open(f'results_{timestamp}.json', 'w') as f:
            json.dump({
                'status': results['status'],
                'metrics': results['metrics'],
                'config': results['config'],
                'convergence_history': results['convergence_history']
            }, f, indent=2)
        
        np.savez(
            f'states_{timestamp}.npz',
            quantum=results['states']['quantum'],
            tunnel=results['states']['tunnel']
        )

    def _validate_params(self, params: Dict):
        valid_ranges = {
            'quantum_lr': (0, 1),
            'tunnel_rate': (0.5, 2.0),
            'temperature': (0.001, 1.0)
        }
        
        for param, value in params.items():
            if param in valid_ranges:
                min_val, max_val = valid_ranges[param]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Параметр {param} вне диапазона [{min_val}, {max_val}]")

    def _reset_default_params(self):
        self._update_params({
            'quantum_lr': 0.05,
            'tunnel_rate': 1.0,
            'temperature': 0.5
        })

class AdaptiveAllocator:
    def __init__(self, config: TQECConfig):
        self.config = config
        self.param_history = []
        
    def optimize(self, problem: Dict, step: int) -> Dict:
        params = {
            'quantum_lr': self._calc_learning_rate(step),
            'tunnel_rate': self._calc_tunnel_rate(step),
            'temperature': self._calc_temperature(step)
        }
        self.param_history.append(params)
        return params
        
    def _calc_learning_rate(self, step):
        base_lr = 0.1
        decay_rate = 0.99
        return base_lr * (decay_rate ** step)
        
    def _calc_tunnel_rate(self, step):
        return 1.0 + 0.1 * np.sin(step / 10)
        
    def _calc_temperature(self, step):
        return max(0.01, 1.0 - step/self.config.max_steps)

class BraketExecutor:
    def __init__(self):
        self.device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        
    def run_hybrid_task(self, circuit_config: List[Dict], classical_params: Dict) -> Dict:
        circuit = Circuit()
        for op in circuit_config:
            gate = getattr(gates, op['gate'])
            qubits = [int(q) for q in op['qubits']]
            circuit.add_instruction(gate(*qubits))
            
        task = self.device.run(circuit, shots=1024)
        result = task.result()
        
        return {
            'counts': result.measurement_counts,
            'expectations': {
                'energy': sum(result.measurement_counts.values()) / 1024
            },
            'classical': classical_params
        }

class TFQHybridModel(tf.keras.Model):
    def __init__(self, num_qubits=4):
        super().__init__()
        self.quantum_layer = tfq.layers.PQC(
            self._build_circuit(num_qubits),
            operators=self._build_observables(num_qubits)
        )
        self.classical = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def _build_circuit(self, num_qubits):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        circuit = cirq.Circuit()
        
        circuit += cirq.H.on_each(qubits)
        for i in range(num_qubits-1):
            circuit += cirq.CNOT(qubits[i], qubits[i+1])
            
        return circuit
        
    def _build_observables(self, num_qubits):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        return [
            cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
            cirq.X(qubits[0]) * cirq.X(qubits[1])
        ]
        
    def call(self, inputs):
        quantum_out = self.quantum_layer(inputs)
        return self.classical(quantum_out)

if __name__ == "__main__":
    config = TQECConfig(
        quantum_units=512,
        tunnel_units=256,
        use_gpu=True,
        use_qiskit=True,
        visualize=True,
        error_correction_type='enhanced',
        distributed_com
distributed_computing=True
    )
    
    system = TQECSystem(config)
    
    # Пример 1: Задача с алгоритмом Гровера
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)  # Оракул для поиска состояния |11>
    
    grover_problem = {
        'type': 'grover_search',
        'algorithm': 'grover',
        'oracle': oracle,
        'num_qubits': 2,
        'use_qiskit': True
    }
    
    # Пример 2: Задача квантовой химии с VQE
    vqe_problem = {
        'type': 'vqe_chemistry',
        'algorithm': 'vqe',
        'molecule': {
            'atom': 'H',
            'coords': [0.0, 0.0, 0.0],
            'charge': 0,
            'multiplicity': 1
        },
        'use_qiskit': True
    }
    
    # Пример 3: Стандартная гибридная задача
    hybrid_problem = {
        'type': 'hybrid_example',
        'classical_params': {
            'param1': 0.5,
            'param2': 1.2
        },
        'tfq_params': 0.7
    }
    
    # Выбор задачи для решения
    problem = grover_problem  # Можно изменить на vqe_problem или hybrid_problem
    
    result = system.solve(problem)
    
    if system.rank == 0:
        print("\nФинальные результаты:")
        print(f"Статус завершения: {result['status']}")
        print(f"Финальная энергия: {result['metrics']['energy']:.6f}")
        print(f"Финальная фидельность: {result['metrics']['fidelity']:.4f}")
        
        # Сохранение и визуализация результатов
        system.monitor.plot_history()
        
        if config.visualize:
            QuantumVisualizer.plot_quantum_state(
                result['states']['quantum'],
                "Финальное квантовое состояние"
            )
            
            # Визуализация параметров оптимизации
            steps = range(len(system.optimizer.param_history))
            lr_values = [p['quantum_lr'] for p in system.optimizer.param_history]
            tunnel_values = [p['tunnel_rate'] for p in system.optimizer.param_history]
            temp_values = [p['temperature'] for p in system.optimizer.param_history]
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(steps, lr_values)
            plt.title('Квантовая скорость обучения')
            
            plt.subplot(1, 3, 2)
            plt.plot(steps, tunnel_values)
            plt.title('Коэффициент туннелирования')
            
            plt.subplot(1, 3, 3)
            plt.plot(steps, temp_values)
            plt.title('Температура')
            
            plt.tight_layout()
            plt.savefig('optimization_params.png')
            plt.close()

        print("\nРезультаты сохранены в файлы:")
        print("- training_history.png (графики метрик)")
        print("- quantum_state.png (визуализация состояния)")
        print("- optimization_params.png (параметры оптимизации)")
        print("- results_*.json (финальные метрики)")
        print("- states_*.npz (финальные состояния)")

# Тесты для новых функциональностей
def test_enhanced_features():
    """Тестирование расширенных возможностей системы"""
    test_config = TQECConfig(
        quantum_units=4,
        tunnel_units=4,
        use_gpu=False,
        max_steps=10,
        verbose=False
    )
    
    # Тест улучшенной коррекции ошибок
    ec = EnhancedErrorCorrection()
    test_state = np.random.rand(4, 2) + 1j*np.random.rand(4, 2)
    corrected_state = ec._correct_quantum(test_state)
    assert corrected_state.shape == test_state.shape, "Ошибка в коррекции состояний"
    
    # Тест алгоритма Гровера
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)
    grover_circ = QuantumAlgorithms.grover_search(oracle, 2)
    assert isinstance(grover_circ, QuantumCircuit), "Ошибка в генерации схемы Гровера"
    
    # Тест распределенных вычислений
    comm = MPI.COMM_WORLD
    distributor = DistributedQuantum(comm)
    test_circuit = QuantumCircuit(4)
    test_circuit.h(range(4))
    subcircuit = distributor.distribute_circuit(test_circuit, 4)
    assert subcircuit.num_qubits <= 4, "Ошибка в распределении схемы"
    
    print("Все тесты расширенных возможностей пройдены успешно!")

if __name__ == "__main__" and MPI.COMM_WORLD.Get_rank() == 0:
    test_enhanced_features()