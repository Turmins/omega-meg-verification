# Собственные обученные веса проекта

Включены 18 `best.pt` state dictionaries без преобразования чисел:

| Каталог | Folds | Общий размер |
|---|---|---:|
| `v1/` | `fold_00.pt` ... `fold_05.pt` | 2,517,354 bytes |
| `eegnet/` | `fold_00.pt` ... `fold_05.pt` | 366,408 bytes |
| `graph/` | `fold_00.pt` ... `fold_05.pt` | 417,024 bytes |
| Все | 18 файлов | 3,300,786 bytes |

Это результаты обучения в данном проекте. Проверены все training summaries,
истории эпох, выбор checkpoint на DEV и соответствующие lock hashes. Загрузка
каждого файла через `torch.load(..., weights_only=True, map_location='cpu')` и
строгий `load_state_dict(..., strict=True)` в соответствующую архитектуру прошла.
Все значения тензоров конечны, текстовые ID и абсолютные пути не обнаружены.
V1 training-only classifier не сохранялся; Graph sensor embeddings относятся
к 260 каналам, а не к участникам. Отдельные participant embeddings/score NPZ/CSV
не включены. Проверка содержимого не является доказательством против memorization.

`manifest.json` связывает каждый файл с SHA-256 исходного checkpoint. Численные
формулы, frozen core и старые manifests не менялись. Для полного inference
через `scripts/14` и `18` по реальным MEG по-прежнему нужны разрешённые входы,
эталоны и совместимые frozen-протоколы/пороги согласно README.

Пример чтения только параметров:
```python
import torch
state = torch.load('weights/eegnet/fold_00.pt', map_location='cpu', weights_only=True)
```

Выбор лицензии собственных кода/весов остаётся за владельцем. Их включение не
означает разрешения распространять исходный датасет или индивидуальные scores.
GitHub push не выполнялся.
