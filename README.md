# MEG verification: публикационное ядро V1/V2

Проверка заявленной личности по короткой MEG-записи. V1 сравнивает фазовый метод
и Siamese CNN; V2 — фазовый контроль, EEGNet и Graph, включая обновление эталона.
Это исходная поставка для проверки этих экспериментов. Hybrid, E/L/F,
identification-прототип, старые презентации и инструменты миграции сюда не входят.
Их неизменённые материалы остаются в отдельном [историческом архиве](docs/ARCHIVE.md).

Все команды ниже выполняются из корня этой папки в Windows PowerShell.
Если буква SSD изменилась, замените её только в первом пути:

```powershell
Set-Location 'F:\codex\300826\omega-meg-verification-publication'
```

## 1. Посмотреть демо без установки научных зависимостей

```powershell
py -3.10 -B tools\prepare_saved_demo.py --archive-root ..\deep_learning_extension
Invoke-Item .\local_artifacts\verification_v1\report\demo.html
```

Импортёр использует только стандартный Python, проверяет хеши и копирует три
готовых файла. Повторно открывайте HTML, а не импортируйте его в занятую папку.
В браузере можно выбрать свой/чужой запрос, длительность и пример ошибки;
видны score, DEV-порог и сохранённое решение фазового метода и Siamese CNN.
**Это V1 saved TEST replay: сигнал не обрабатывается, модель не запускается.**
Это не показ EEGNet/Graph и не новая независимая оценка. После подготовки HTML
работает без Python, GPU и сети. Одних tracked-файлов Git для реального replay
недостаточно: необходим локальный архив с исходными HTML/CSV.

## 2. Установить среду и проверить код

Рекомендуется 64-битный Python 3.10. Историческая среда использовала 3.10.11 и
torch 2.5.1+cu121. Старая `.venv` другого компьютера непереносима.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Активация среды не требуется. Тесты проверяют V1/V2, разделение субъектов,
границы окон, метрики, freeze barriers, provenance и безопасное восстановление.
Они используют синтетические данные и не переобучают модели на MEG.
Внешний integration test явно пропускается, пока исторические runs не восстановлены.
Установка требует доступных пакетов; wheelhouse сюда не включён. CPU/CUDA и
численная совместимость проверяются на целевом компьютере отдельно.

## 3. Восстановить результаты и выполнить inference

Данные и веса приватные и не входят в Git. Ожидается соседнее расположение:

```text
parent/
  omega-meg-verification-publication/  # эта папка
  omega_student_broadband_T260_v1/     # разрешённый экспорт: 120 NPY и метаданные
  deep_learning_extension/            # неизменённый исторический архив
```

Сначала можно посмотреть точный план копирования. Основные runs занимают около
78 MB; три кешированные записи для примеров добавляют около 207 MB с graph cache.
Остальные гигабайты архива не копируются:

```powershell
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache --plan
.\.venv\Scripts\python.exe -B tools\restore_results.py --include-example-cache
.\.venv\Scripts\python.exe -B tools\verify_results.py
```

Восстановление проверяет pinned SHA-256, сохраняет старые manifests и environment
побайтово, не заменяет отличающиеся существующие файлы. Новая квитанция импорта
лежит отдельно в игнорируемом `local_artifacts/imports/`; это не provenance обучения.
`verify_results.py` воспроизводит метрики из сохранённых scores и последовательные
решения из сохранённых embeddings, не выполняя нейросетевой inference.

Настоящий inference на уже обученных моделях:

```powershell
.\.venv\Scripts\python.exe -B scripts\14_verify_claim.py --fold 0 --claimed sub-0006 --source sub-0006 --duration 30 --start 0
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model eegnet --fold 0 --claimed sub-0006 --source sub-0006 --enrollment 30 --duration 30 --start 0 --policy static
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model graph --fold 0 --claimed sub-0006 --source sub-0006 --enrollment 30 --duration 30 --start 0 --policy static
```

Для чужого запроса замените `--source sub-0006` на `--source sub-0016`.
Вывод — JSON со score, порогом и решением. Ошибочное принятие/отклонение возможно.
V1 обрабатывает исходные разрешённые окна; V2 запускает encoder на проверенных
кешированных окнах. Показаны только поддержанные примерные записи, а не полный
feature cache. Для других участников V2 нужны их совместимые кеши.
Профиль не сохраняется без `--profile`; `static` не обновляет эталон.

## 4. Новый эксперимент — отдельное действие

Стадия обязательна: простой запуск `11` или `17` больше не начинает обучение.
Номера файлов не задают цепочку запуска. До новых расчётов прочитайте
[протокол](docs/PROTOCOL.md) и [внешние артефакты](docs/EXTERNAL_ARTIFACTS.md).

```powershell
# Новый V1 output, не историческая папка outputs/verification_v1:
.\.venv\Scripts\python.exe -B scripts\11_run_verification.py --out outputs\v1_new --stage prepare
.\.venv\Scripts\python.exe -B scripts\11_run_verification.py --out outputs\v1_new --stage train
.\.venv\Scripts\python.exe -B scripts\11_run_verification.py --out outputs\v1_new --stage evaluate
.\.venv\Scripts\python.exe -B scripts\12_report_verification.py --out outputs\v1_new

# Новый V2: конфиг требует полноценный совместимый V1 в outputs/verification_v1.
# Трёх примерных кешей из предыдущего пункта недостаточно для обучения.
.\.venv\Scripts\python.exe -B scripts\17_run_adaptive_verification.py --out outputs\v2_new --stage all
```

Нужны исходные данные и полные соответствующие feature caches. Подготовка новой
публикационной версии не запускала эти обучения. Для изменённого `base_run`
используйте отдельный новый конфиг и новый output; manifests старых runs не правьте.
Новые вызовы получают отдельные invocation-записи; текущее окружение не приписывается
историческому обучению.

Проверка неизменности конкретных входов заменяет старый комплексный feasibility CLI:

```powershell
.\.venv\Scripts\python.exe -B tools\check_integrity.py snapshot --path ..\omega_student_broadband_T260_v1 --snapshot local_artifacts\inputs_before.json
# После расчёта:
.\.venv\Scripts\python.exe -B tools\check_integrity.py verify --snapshot local_artifacts\inputs_before.json
```

Полное хеширование датасета читает несколько GiB. Snapshot создаётся до расчёта,
не перезаписывается и не создаётся автоматически на стадии verify.

## Что проверяет публикация

Шесть subject-disjoint folds: 40 TRAIN / 10 DEV / 10 TEST, две сессии человека
остаются в одной части. Порог выбирается на DEV и фиксируется до TEST данного run.
Основная ошибка — HTER=(FAR+FRR)/2; ROC-AUC и диагностический EER усредняются внутри folds.
V1 240/30 s: CNN HTER 12.80%, phase 6.23%. V2 static 30/30 s: EEGNet 13.95%,
Graph 21.33%, phase 10.94%. V1 и V2 имеют разные сетки запросов.
Датасет уже изучался; это исследовательская cross-validation, а не внешняя
независимая или клиническая/продуктовая оценка. Попытки и TRAIN folds зависимы.

Подробности: [сохранённые агрегаты](results/README.md),
[frozen-совместимость](docs/FROZEN_CORE.md), [граница архива](docs/ARCHIVE.md),
[проверки этой поставки](docs/VALIDATION.md).

Это source-checkout workflow с соседними `src`, `configs`, `scripts` и reference
asset. Работа отдельно установленного wheel без этих файлов не обещается.
До внешней публикации владельцу нужно определить лицензию кода/reference и условия
доступа к данным: лицензия и публичные ссылки на приватные данные здесь не выдуманы.
Remote не задаётся; подготовка не выполняет push или установку внешних skills.
