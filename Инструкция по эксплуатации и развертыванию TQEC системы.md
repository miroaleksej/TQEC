### **Инструкция по эксплуатации и развертыванию TQEC системы**  
Для начинающих с пошаговыми примерами.

---

## **1. Установка**  
**Требования**:  
- Python 3.8+  
- CUDA 11.0+ (для GPU)  
- MPI (например, OpenMPI)  

```bash
# Установка зависимостей
pip install numpy cupy mpi4py qiskit cirq tensorflow-quantum matplotlib
pip install braket-sdk  # Для Amazon Braket
```

---

## **2. Запуск системы**  
### **Пример 1: Локальный симулятор (CPU/GPU)**  
Файл `simple_run.py`:
```python
from tqec_enhanced import TQECConfig, TQECSystem

# Конфигурация для 4 кубитов
config = TQECConfig(
    quantum_units=4,
    use_gpu=False,  # Для GPU измените на True
    visualize=True
)

system = TQECSystem(config)

# Задача: алгоритм Гровера
oracle = QuantumCircuit(2)
oracle.cz(0, 1)  # Оракул для |11⟩

problem = {
    "type": "grover",
    "algorithm": "grover",
    "oracle": oracle,
    "num_qubits": 2
}

result = system.solve(problem)
print(result["metrics"])
```
**Запуск**:  
```bash
python simple_run.py
```

---

### **Пример 2: Распределенные вычисления (MPI)**  
Файл `mpi_run.py`:
```python
from tqec_enhanced import TQECConfig, TQECSystem
from mpi4py import MPI

config = TQECConfig(
    quantum_units=8,
    distributed_computing=True
)

system = TQECSystem(config)

if MPI.COMM_WORLD.Get_rank() == 0:
    problem = {"type": "qft", "algorithm": "qft", "num_qubits": 3}
else:
    problem = None

result = system.solve(problem)
```
**Запуск**:  
```bash
mpiexec -n 4 python mpi_run.py  # 4 процесса MPI
```

---

## **3. Работа с реальными квантовыми устройствами**  
### **Пример 3: Запуск на IBM Quantum**  
```python
config = TQECConfig(
    quantum_units=5,
    use_qiskit=True,
    use_real_quantum=True,
    qiskit_token="ВАШ_ТОКЕН_IBMQ"
)

system = TQECSystem(config)
problem = {
    "type": "vqe",
    "algorithm": "vqe",
    "molecule": {
        "atom": "H",
        "coords": [0.0, 0.0, 0.0]
    }
}
result = system.solve(problem)
```

---

## **4. Визуализация результатов**  
После выполнения кода автоматически генерируются:  
- `quantum_state.png` — 3D-визуализация состояний  
- `training_history.png` — графики энергии и фидельности  
- `optimization_params.png` — параметры оптимизации  

**Пример просмотра результатов**:  
```python
from matplotlib import pyplot as plt
img = plt.imread("quantum_state.png")
plt.imshow(img)
plt.show()
```

---

## **5. Типовые сценарии**  
### **Сценарий 1: Квантовая химия**  
```python
problem = {
    "type": "vqe_chemistry",
    "algorithm": "vqe",
    "molecule": {
        "atom": "H2",
        "coords": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        "charge": 0,
        "multiplicity": 1
    }
}
```

### **Сценарий 2: Квантовый классификатор**  
```python
from qml_module import QMLConfig, QMLIntegration

qml_config = QMLConfig(num_qubits=4, use_qiskit=False)
qml_integration = QMLIntegration(system, qml_config)

X_train = np.random.rand(10, 4)  # 10 samples, 4 features
y_train = np.random.randint(0, 2, 10)  # Binary labels

history = qml_integration.train_model("vqc", X_train, y_train, epochs=10)
```

---

## **6. Решение проблем**  
| Ошибка | Решение |
|--------|---------|
| `CUDA out of memory` | Уменьшите `quantum_units` или используйте `use_gpu=False` |
| `MPI rank mismatch` | Проверьте количество процессов (`mpiexec -n N`) |
| `IBMQ credentials invalid` | Обновите токен на [IBM Quantum Experience](https://quantum-computing.ibm.com/) |

---

## **7. Дополнительные возможности**  
- **Интеграция с TensorFlow**:  
  ```python
  model = tf.keras.Sequential([
      tfq.layers.PQC(circuit, observables),
      tf.keras.layers.Dense(1)
  ])
  ```
- **Экспорт схем**:  
  ```python
  qiskit_circ = problem["qiskit_circuit"].qasm()
  ```

---

**Важно**: Для работы с GPU убедитесь, что установлены:  
```bash
pip install cupy-cuda11x  # Где 11x — ваша версия CUDA
```

Система готова к использованию в исследовательских и промышленных задачах. Для углубленного изучения см. [документацию](https://example.com/tqec-docs).