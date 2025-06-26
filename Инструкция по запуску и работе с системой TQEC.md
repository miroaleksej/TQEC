# Инструкция по запуску и работе с системой TQEC

## 1. Требования к системе

### Минимальные требования:
- ОС: Linux (Ubuntu 20.04+) или Windows 10/11 с WSL2
- Python 3.8+
- 8 ГБ оперативной памяти
- 4 ядра CPU
- 2 ГБ свободного места на диске

### Рекомендуемые требования (для GPU-ускорения):
- GPU NVIDIA с поддержкой CUDA 11.0+
- 16+ ГБ оперативной памяти
- 8+ ядер CPU

## 2. Установка и развертывание

### Установка зависимостей:

```bash
# Клонирование репозитория
git clone https://github.com/your-repo/tqec-system.git
cd tqec-system

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
# venv\Scripts\activate   # Для Windows

# Установка основных зависимостей
pip install -r requirements.txt

# Для GPU-ускорения (опционально)
pip install cupy-cuda11x  # Замените x на вашу версию CUDA
```

### Конфигурация системы:

Создайте файл `config.yaml` с основными параметрами:

```yaml
quantum_units: 512
tunnel_units: 256
use_gpu: false
use_qiskit: false
use_braket: false
verbose: true
error_correction_type: "enhanced"
```

## 3. Запуск системы в режиме эмуляции

Система может работать полностью на CPU без реального квантового железа, используя:
- Cirq для симуляции квантовых схем
- NumPy/CuPy для матричных операций
- MPI для распределенных вычислений (опционально)

### Пример запуска:

```python
from tqec_enhanced import TQECConfig, TQECSystem

# Конфигурация для эмуляции
config = TQECConfig(
    quantum_units=128,
    tunnel_units=64,
    use_gpu=False,
    use_qiskit=False,
    distributed_computing=False
)

# Инициализация системы
system = TQECSystem(config)

# Пример задачи - алгоритм Гровера
oracle = ...  # Определение оракула
problem = {
    'type': 'grover_search',
    'algorithm': 'grover',
    'num_qubits': 3,
    'oracle': oracle
}

# Запуск решения
result = system.solve(problem)
print(result)
```

## 4. Примеры работы

### Пример 1: Квантовая химия (VQE)

```python
from quantum_chemistry import QuantumChemistryConfig, MolecularGroundState

# Конфигурация
chem_config = QuantumChemistryConfig(basis_set='sto-3g', verbose=True)

# Создание экземпляра
calculator = MolecularGroundState(chem_config)

# Загрузка молекулы водорода
geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, 0.74])]
molecule = calculator.load_molecule(geometry, 'H2 molecule')

# Получение гамильтониана
hamiltonian = calculator.get_qubit_hamiltonian()

# Запуск VQE
vqe_result = calculator.solve_vqe(num_layers=3)
print(f"VQE Energy: {vqe_result['energy']}")
```

### Пример 2: Квантовое машинное обучение

```python
from qml_module import QMLConfig, VariationalQuantumClassifier

# Конфигурация
qml_config = QMLConfig(num_qubits=4, num_layers=2)

# Создание классификатора
vqc = VariationalQuantumClassifier(qml_config)

# Пример данных
X_train = np.random.rand(100, 4)  # 100 samples, 4 features
y_train = np.random.randint(0, 2, 100)  # Binary labels

# Обучение
history = vqc.fit(X_train, y_train, epochs=50)

# Предсказание
X_test = np.random.rand(5, 4)
predictions = vqc.predict(X_test)
print(predictions)
```

### Пример 3: Алгоритм Гровера (поиск)

```python
from tqec_enhanced import QuantumAlgorithms

# Создание оракула для поиска |11> состояния
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# Построение схемы Гровера
grover_circuit = QuantumAlgorithms.grover_search(oracle, num_qubits=2)

# Визуализация схемы
grover_circuit.draw(output='mpl')
plt.savefig('grover_circuit.png')
```

## 5. Описание работы эмуляции квантовых вычислений

### Принципы эмуляции:

1. **Представление состояний**:
   - Квантовые состояния представляются как комплексные векторы в Hilbert пространстве
   - Для n кубитов используется вектор размерности 2ⁿ

2. **Операции с кубитами**:
   - Квантовые гейты представляются как унитарные матрицы
   - Применение гейта = умножение матрицы на вектор состояния

3. **Измерения**:
   - Вероятности вычисляются как квадраты модулей амплитуд
   - Результаты измерений генерируются случайно согласно распределению вероятностей

4. **Шум и ошибки**:
   - Моделируются через добавление случайных возмущений
   - Коррекция ошибок применяется через специальные матрицы преобразования

### Особенности реализации:

1. **Оптимизации**:
   - Использование sparse матриц для больших систем
   - Пакетная обработка операций
   - Кэширование часто используемых гейтов

2. **Ограничения**:
   - Максимальное число кубитов ограничено памятью (∼30 кубитов на обычном ПК)
   - Время выполнения растет экспоненциально с числом кубитов

3. **Визуализация**:
   - Возможность отображения схем в виде диаграмм
   - 3D визуализация состояний на сфере Блоха
   - Графики динамики системы

## 6. Работа с результатами

### Сохранение результатов:

```python
# После выполнения расчета
system.save_final_results()  # Сохраняет JSON с метриками и NPZ с состояниями
system.monitor.plot_history()  # Сохраняет графики обучения
```

### Анализ результатов:

1. **Файлы результатов**:
   - `results_<timestamp>.json` - метаданные и метрики
   - `states_<timestamp>.npz` - квантовые состояния и туннельные вероятности
   - `training_history.png` - графики сходимости

2. **Пример анализа**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Загрузка результатов
data = np.load('states_20230515.npz')
quantum_states = data['quantum']
tunnel_probs = data['tunnel']

# Визуализация распределения вероятностей
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(np.abs(quantum_states[:, 0])**2, bins=50)
plt.title('Вероятности |0> состояния')

plt.subplot(1, 2, 2)
plt.hist(tunnel_probs, bins=50)
plt.title('Туннельные вероятности')
plt.show()
```

## 7. Советы по работе

1. **Для небольших систем** (до 10 кубитов):
   - Используйте режим без GPU для простоты отладки
   - Включите verbose=True для подробного лога

2. **Для больших систем** (10+ кубитов):
   - Активируйте use_gpu=True
   - Рассмотрите distributed_computing=True
   - Увеличьте memory_limit в конфигурации

3. **Оптимизация производительности**:
   - Используйте меньшую точность (float32 вместо float64)
   - Уменьшите число шагов max_steps для тестовых запусков
   - Отключите визуализацию для production-запусков

Система TQEC предоставляет гибкий инструмент для исследования гибридных квантово-классических алгоритмов без необходимости доступа к реальному квантовому оборудованию, с возможностью постепенного масштабирования от простых тестовых примеров до сложных вычислительных задач.