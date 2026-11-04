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

## 1. Открыть публичное учебное демо сразу после скачивания

```powershell
Invoke-Item .\demo\index.html
```

Демо входит в Git и работает без Python, весов, GPU, приватного архива и сети.
В нём настоящие агрегированные результаты V1/V2 и отдельно синтетические примеры
правила `score >= threshold`: свой/чужой запрос, ложное принятие и ложный отказ.
Переключаются протокол, метод, длительность запроса/эталона и политика обновления.
**Браузер не запускает нейросеть. Синтетические score/порог не являются реальными
индивидуальными TEST-решениями или сохранёнными DEV-порогами.** В публичном HTML
нет идентификаторов участников и исходных MEG.

На странице файла GitHub виден исходный HTML. Для запуска скачайте/клонируйте
репозиторий и откройте `demo/index.html`. GitHub Pages не публиковался.

Полный исторический V1 saved replay остаётся отдельным приватным вариантом.
Для разрешённой локальной копии:

```powershell
py -3.10 -B tools\prepare_saved_demo.py --archive-root ..\deep_learning_extension
Invoke-Item .\local_artifacts\verification_v1\report\demo.html
```

Этот импортёр проверяет и копирует реальные индивидуальные HTML/CSV в игнорируемую
папку, а не в Git. Он не выполняет inference. Не включайте этот полный replay
в публичную поставку без подтверждения условий распространения. Повторный импорт
в занятую папку остановится; просто откройте уже созданный HTML.

Состав и точный вопрос о правах: [PUBLIC_ARTIFACTS.md](docs/PUBLIC_ARTIFACTS.md).

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

Исходные данные и индивидуальные score-таблицы не входят в Git. Все 18 собственных обученных
весов (3,30 МБ) включены в `weights/`: V1, EEGNet и Graph, по шесть folds.
Их SHA-256 совпадают с историческими checkpoint-файлами; это параметры моделей,
а не исходные записи или таблицы участников.
См. [weights/README.md](weights/README.md). Для приватного воспроизведения ожидается
соседнее расположение:

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

Настоящий inference на уже обученных моделях использует локальные артефакты.
Идентификаторы примеров берутся из приватного протокола, а не из публичного HTML:

```powershell
$fold0 = (Get-Content .\outputs\verification_v1\protocol.json -Raw | ConvertFrom-Json).folds[0].test_subjects
$claimed = $fold0[0]
$source = $fold0[0]  # свой запрос; для чужого используйте $fold0[1]
.\.venv\Scripts\python.exe -B scripts\14_verify_claim.py --fold 0 --claimed $claimed --source $source --duration 30 --start 0
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model eegnet --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
.\.venv\Scripts\python.exe -B scripts\18_verify_adaptive_claim.py --model graph --fold 0 --claimed $claimed --source $source --enrollment 30 --duration 30 --start 0 --policy static
```

Для чужого запроса присвойте `$source = $fold0[1]` и повторите нужную команду.
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

## Датасет: пользовательская ссылка Google Drive и размещение

- Ссылка пользователя на Google Drive: **[ДОБАВИТЬ_ССЫЛКУ_GOOGLE_DRIVE]**
- Исходное происхождение / точный релиз, если подтверждены: **[УКАЗАТЬ_ИСТОЧНИК_И_РЕЛИЗ]**
- Инструкция доступа к файлу Drive и применимые условия: **[ДОБАВИТЬ_ИНСТРУКЦИЮ_ДОСТУПА]**
- Цитирование / DOI: **[ДОБАВИТЬ_ОФИЦИАЛЬНОЕ_ЦИТИРОВАНИЕ]**

Поле Google Drive намеренно пустое: ссылку предоставляет пользователь. Репозиторий
не создаёт ссылку и не меняет права доступа. Размещайте только разрешённую копию;
наличие ссылки само по себе не подтверждает право распространять данные или веса.
Сведения об исходном архиве и цитировании заполняются отдельно от ссылки Drive.

После получения разрешённого архива распакуйте его рядом с репозиторием в
`../omega_student_broadband_T260_v1/`, без дополнительного вложенного уровня.
Внутри должны находиться `cache/` с 120 NPY, `recording_index.csv`,
`cache_manifest.csv`, `channels.txt`, `protocol_FROZEN.json` и README экспорта.
Содержимое данных не добавляйте в Git и не заменяйте в нём frozen-метаданные.
Непосредственный вход проекта — разрешённый локальный экспорт
`omega_student_broadband_T260_v1`: 120 NPY и фиксированные метаданные.
Для точного воспроизведения нужны именно совместимые экспорт, порядок каналов,
разбиения, веса и frozen-протоколы; новая загрузка иного upstream-релиза не обещает
совпадения с сохранёнными хешами. Ссылка на датасет не заменяет разрешения на
распространение обученных весов или индивидуальных результатов. Исходные MEG в Git
не добавляются. Автономное публичное демо не требует доступа к датасету.

Пересборка публичного HTML из уже включённых агрегатов: `python -B tools/build_public_demo.py`.
Проверка UI-контроллера при установленном Node: `node tests/public_demo_controls.cjs`;
это unit-проверка логики, не визуальный браузерный PASS.
