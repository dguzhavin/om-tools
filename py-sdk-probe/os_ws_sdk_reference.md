# os_ws_sdk 1.0: как SDK работает на самом деле

Справочник для технической команды ATOMS по Python SDK Optimacros (`os_ws_sdk` 1.0, Workspace 9.200).
Официальная документация описывает, как SDK задуман. Здесь записано, как он ведёт себя на живом
Workspace: что возвращает каждый метод, в каком виде, сколько длится, где падает и чем отличается
от документации.

**Как получены данные.** Стенд `sdk_probe.py` вызвал все публичные методы SDK на тестовой модели
`PY_SDK ##1.0` (ws125, storage 9.200.164.73) в четыре прогона 12.10.2026: чтение, запись с откатом,
управление моделью. Около 500 вызовов, отчёты `report.json` каждого прогона лежат рядом со стендом.
Окружение: Python 3.13.7, pandas 3.0.6, macOS.

## Как читать

| Метка | Что значит |
|---|---|
| **[стенд]** | Проверено вызовом на реальном Workspace |
| **[код]** | Проверено по исходному коду wheel, на стенде не воспроизводилось |
| **[док]** | Только из официальной документации |
| **[?]** | Предположение по косвенным признакам, нужна проверка |

Если [док] и [стенд] расходятся, верен [стенд]. Тестовая модель маленькая (3 мультикуба, 90–120
ячеек), поэтому время вызовов показывает накладные расходы API, а не скорость на больших объёмах.

---

## 0. Коротко: что нужно знать до первого скрипта

1. **`mc.export_df()` и `view.export_df()` сломаны.** Возвращают DataFrame из одной колонки.
   Обход: `await mc.export_df(sep=";")`. Подробности — в разделе 5.
2. **`catalog.import_file()` не добавляет элементы.** Ошибки нет, справочник остаётся пустым,
   с любым из проверенных форматов файла. Элементы создавайте через `add_elements`.
3. **`export_om_file` работает только для мультикубов и их вью.** У справочника и его вью он падает с 404.
4. **Экспорт в несуществующую папку ничего не делает и не сообщает об этом.** Создавайте папку сами.
5. **«Не найдено» обрабатывается по-разному:** одни методы возвращают `None`, другие бросают исключение.
   Таблица — в разделе 2.
6. **Права на макрос нельзя снять значением `'none'`** — SDK отвергает его на клиенте. Используйте `False`.
7. **Исходный код скрипта доступен:** `get_script_info(id)`, колонка `Macros`.
8. **Все формулы модели — одним запросом** через системный грид `CUBES_GRID`, колонка `Formula`.
   `get_formula` в цикле не нужен.
9. **`model.backup()` ничего не делает:** возвращается мгновенно, новый бэкап не появляется.

---

## 1. Установка и подключение

SDK распространяется только wheel-файлом, в PyPI его нет. Python 3.10+.

```bash
pip install ./os_ws_sdk-1.0-py3-none-any.whl pandas python-dotenv
```

Зависимости wheel (aiohttp, dataclasses_json и др.) pip ставит сам. Версия пакета через
`importlib.metadata.version("os_ws_sdk")` — `"1.0"` **[стенд]**.

```python
import asyncio, sys
from os_ws_sdk.api.connector_manager import ConnectorManager
from os_ws_sdk.api.ws_version import WsVersion
from os_ws_sdk.objects.workspace import Workspace

async def main():
    connector = ConnectorManager.create(version=WsVersion.VERSION_9_200, token="<токен>")
    ws = Workspace("https://wsNNN.optimacros.com", connector)
    model = await ws.get_model(id="<id модели>")
    print(model.info())

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())
```

- `WsVersion` содержит только `VERSION_9_200` **[стенд]**.
- `ConnectorManager.create` и `Workspace(...)` в сеть не ходят, ошибка в токене или URL проявится
  на первом запросе **[стенд]**.
- Логгер SDK `os_ws_sdk.logging_sdk.logger` без хендлеров: его сообщения видны только после
  `logging.basicConfig(...)` **[код]**.
- На Windows ставим `WindowsSelectorEventLoopPolicy`, как в тестах самого SDK **[код]**. На Windows
  стенд не запускался.

### Импорты

Импорт из корня пакета не работает: `from os_ws_sdk import ScriptRunType` (как в документации),
`ScriptStatus`, `Workspace`, `ConnectorManager` — все дают `AttributeError` **[стенд]**. Только полные пути:

| Имя | Модуль |
|---|---|
| `ConnectorManager` | `os_ws_sdk.api.connector_manager` |
| `WsVersion` | `os_ws_sdk.api.ws_version` |
| `Workspace` | `os_ws_sdk.objects.workspace` |
| `WorkspaceException`, `ModelNotFoundError` | `os_ws_sdk.common.exceptions` |
| `DownloadFormat`, `DownloadFormatOM` | `os_ws_sdk.components.download_format` |
| `EncodingTypeExport`, `EncodingTypeImport` | `os_ws_sdk.components.encoding_type` |
| `Position` | `os_ws_sdk.components.position` |
| `ScriptStatus`, `ScriptRunType`, `build_scripts_history_filter` | `os_ws_sdk.components.scripts_history_filter` |
| `SystemGridExporter`, `GridTypeIdentifiers` | `os_ws_sdk.objects.system_grid_exporter` |
| `CubeFormulaManager` | `os_ws_sdk.objects.cube_formula_manager` |
| `PermissionChangeManager` | `os_ws_sdk.objects.permission_change_manager` |
| `PermissionGridExporter`, `PermissionGridData` | `os_ws_sdk.objects.permission_grid_exporter` |
| `UamManager`, `UamGridData` | `os_ws_sdk.objects.uam_manager` |

Все пути проверены импортом **[стенд]**.

### Значения перечислений

| Enum | Значения **[стенд]** |
|---|---|
| `DownloadFormat` | `CSV="csv"`, `TXT="txt"`, `XLS="xls"`, `XLSX="xlsx"` |
| `DownloadFormatOM` | `OM_CSV="csv"`, `OM_TXT="txt"`, `OM_ZIP="gz"` |
| `EncodingTypeExport` | 8 значений: `WINDOWS_1251="windows-1251"`, `WINDOWS_1251_UPPER="WINDOWS-1251"`, `WIN="win"`, `UTF`, `UTF_UPPER`, `UTF_8`, `UTF_8_UPPER` и др. |
| `EncodingTypeImport` | 6 значений: `WINDOWS_1251="WINDOWS-1251"`, `WIN="win"`, `UTF="utf"`, `UTF_8_UPPER`, `AUTO`, `AUTO_UPPER` |
| `ScriptStatus` | `NONE=0`, `WAIT_IN_QUEUE=1`, `IN_PROGRESS=2`, `SUCCESS=3`, `FAILED`, `ABORTED` |
| `ScriptRunType` | `INTERFACE=0`, `SCHEDULER=1`, `API=2` |
| `Position` | `START="Start"`, `END="End"`, `BEFORE="Before"`, `AFTER="After"` |
| `GridTypeIdentifiers` | 17 системных гридов, см. раздел 4 |

### Устройство объектов

`Workspace`, `Model`, `Multicube`, `View`, `Catalog`, `CatalogView` — dataclasses на
`dataclasses_json`: у всех есть `to_dict`, `to_json`, `from_dict`, `from_json`, `schema` **[стенд]**.
Объект `Model` из `get_models()` несёт намного больше полей, чем `model.info()`: `storage_version`,
`auto_calc`, `created_at`/`updated_at` (unix-время), `workspace_folder_long_id`, режимы чтения и
записи хранилища, `restore_mode` и др. **[стенд]**.

У коннектора 17 публичных методов, которых нет в документации: `get_system_grid_data`,
`get_view_definition`, `get_dashboard_definition`, `get_entity_types`, `download_dataframe`,
`download_file_data`, `download_milticube_data` (опечатка в имени), `download_backup`, `get_lp_hash`,
`get_view_hash`, `get_optimizer_requests_table` и др. **[стенд]**. Сигнатуры не изучались —
это недокументированный путь на случай, если публичного метода не хватает.

---

## 2. Общее поведение

### Sync и async

| Вызов | Как вызывать | Ходит в сеть |
|---|---|---|
| `model.info()` | sync | нет |
| `mc.info()`, `view.info()`, `catalog.info()`, `catalog_view.info()` | `await` | нет: отдаёт поля, сохранённые в объекте |
| `ws.info()` | `await` | да |

Объекты SDK — снимки на момент получения. После `catalog.rename(...)` старый объект продолжает
отдавать старое имя; свежие данные — только через новый `get_catalog(...)` **[стенд]**.
Исключение: `model.stop()` меняет `is_offline` у того же объекта **[стенд]**.

### Что происходит, если объекта нет **[стенд]**

| Вызов | Результат |
|---|---|
| `ws.get_model(id)` | `ModelNotFoundError: Модель model='…' не найдена` |
| `ws.get_model_by_name(name)` | `None` |
| `model.get_multicube(id)`, `get_multicube_by_name(name)` | `None` |
| `model.get_id_multicube_by_name(name)` | `WorkspaceException(404, "Entity by query `…` not found")` |
| `model.get_catalog(id)`, `get_catalog_by_name(name)` | `WorkspaceException(404, "Каталог с id=… не найден")` |
| `model.get_catalog_view_by_name(...)`, `mc.get_view_by_name(...)`, `catalog.get_view_by_name(...)` | `None` |
| `model.get_script_info(id)` | `None` |
| `model.get_formula(id)` | `WorkspaceException(404, "Куб с id=… не найден")` |
| `model.get_dashboard_multicubes(id)` | `WorkspaceException(404, "В ответе GET_DASHBOARD_DEFINITION нет definitionJsonString…")` |
| `model.delete_catalogs([id])` | `WorkspaceException(404, …)` |
| `catalog.delete_elements(names=[...])` | ничего: несуществующие имена молча пропускаются |
| `model.get_backup(id)` | `WorkspaceException(500, …)` с HTML-страницей сервера в тексте |
| `model.run_script(id)` | `KeyError: 'output'` — не `WorkspaceException` |

### Ошибки

- `WorkspaceException` хранит в `args` пару `(код, текст)`: `e.args[0]` — HTTP-подобный код (400, 404, 500),
  `e.args[1]` — текст, часто по-русски **[стенд]**.
- Часть проверок SDK делает на клиенте, без запроса: недопустимое значение права, роль не числом,
  несуществующий файл для импорта (`FileNotFoundError`). Такие ошибки приходят за миллисекунды **[стенд]**.
- Неверные входные данные иногда дают не `WorkspaceException`, а
  `AttributeError: 'NoneType' object has no attribute 'items'` из `dataclasses_json`: так было при
  неверном id измерения в фильтре и при экспорте из остановленной модели **[стенд]**. Ловите `Exception`, а не только `WorkspaceException`.
- `action_info` для скрипта ошибку не бросает, а возвращает `dict` с `modalInfo.isError=True`
  («Action info disabled»). Ответы с `modalInfo` проверяйте вручную **[стенд]**.

### Время вызовов **[стенд]**

| Операция | Время |
|---|---|
| Метаданные: `get_models` (122 модели), `get_multicubes`, `get_catalogs`, `*_by_name` | 0,6–1 с |
| Системный грид | ~1 с |
| Любой экспорт в файл или DataFrame | 2,1–2,7 с |
| `import_om_file`, `import_file`, `clear` | 0,7–1 с |
| `get_formula` | ~1 с |
| `change_formula` | ~5 с |
| Изменение права | ~2 с |
| `change_uam` | ~4,6 с |
| `get_user_permissions` | ~5 с |
| `get_dashboard_multicube_links()` на 4 дашбордах | ~7 с |
| `stop()` / `start()` | 1,7 с / 22,6 с |

`*_by_name`, судя по времени (столько же, сколько полный список), каждый раз загружает весь список объектов **[?]**; у `ws.get_model` это проверено по коду **[код]**. Если
объект нужен несколько раз, сохраните его в переменную.

**Параллельность работает:** `asyncio.gather(mc.export_df(), mc.export_om_file(path))` занял 2,4 с,
столько же, сколько один экспорт **[стенд]**. В `ws.info()` видно `concurrency_max=30`.

### Файлы

- `path` во всех `export_*` — **папка**. Имя файла SDK строит сам: `<имя объекта>.<расширение>`,
  например `SRC_MC.csv`, `SRC_MC view 1.csv`. Путь методы не возвращают, результат `None` **[стенд]**.
- **Если папки нет, SDK не создаёт её, не пишет файл и не бросает ошибку** **[стенд]**. Всегда
  `Path(dir).mkdir(parents=True, exist_ok=True)` перед экспортом.
- Повторный экспорт того же объекта в ту же папку даст файл с тем же именем **[?]** (перезапись не
  проверялась). Для нескольких экспортов одного объекта используйте разные папки.

---

## 3. Workspace

| Метод | Возвращает **[стенд]** |
|---|---|
| `await ws.info()` | `WsInfo`: `workspace_name`, `workspace_id`, `workspace_size_allowance` / `current_workspace_size` (байты), `storage_version`, `tag_summaries` (список `TagMessage`: long_id, name, color…), `audit_settings_info`, `concurrency_max`, `check_license_status`, `active_model_only`. **`model_ids`, `model_names`, `model_summaries` приходят пустыми** |
| `await ws.get_models()` | `list[Model]`, все модели WS (122 за 0,9 с) |
| `await ws.get_model(id)` | `Model` или `ModelNotFoundError` |
| `await ws.get_model_by_name(name)` | `Model` или `None` |

Список моделей берите из `get_models()`, не из `ws.info()`.

---

## 4. Модель: структура и метаданные

| Метод | Возвращает **[стенд]** |
|---|---|
| `model.info()` (sync) | `ModelInfo`: model_name, cell_count, memory, model_id, model_definition_serial_number, is_offline |
| `get_multicubes()` | `list[Multicube]`, у объекта `name`, `id`, `model_id` |
| `get_multicube(id)`, `get_multicube_by_name(name)` | `Multicube` или `None` |
| `get_id_multicube_by_name(name)` | `int` или `WorkspaceException(404)` |
| `get_catalogs(start=0, count=-1)` | `list[Catalog]`. Включает системные справочники `Roles`, `Users`, `Access Levels`. Пагинация работает: `count=3` вернул 3 из 5 |
| `get_catalog(id)`, `get_catalog_by_name(name)` | `Catalog`. В объекте есть `unknown_things` — сырой ответ сервера: `itemCount`, `isFlat`, `isNumber`, `sortOrder`, `subsetEntityLongIds`, `propertiesInfo` и др. |
| `get_catalog_views(catalog_id)` | `list[CatalogView]`, ~1,3 с |
| `get_catalog_view_by_name(catalog_id, name)` | `CatalogView` или `None` |
| `action_info(action_id)` | `dict`. Для оптимизационного запроса: `name`, `savedActionLongId`, `objectives`, `variables`, `constraints`, `optimizerLongId`, `options`, `actionId="SAVED_ACTION_INFO"`. Для скрипта — ошибка внутри `dict` (`modalInfo.isError`) |

### Системные гриды

```python
from os_ws_sdk.objects.system_grid_exporter import SystemGridExporter, GridTypeIdentifiers
exporter = SystemGridExporter.from_model(model)
df = await exporter.export_df(GridTypeIdentifiers.CUBES_GRID)
```

| Грид | ID | Колонки **[стенд]** |
|---|---|---|
| `MULTICUBES_GRID` | 137000000002 | rowEntityLongId, rowLabel, Code, Folder, Comments, User Lists, Time Scale, Versions, Cube Subset, Clean Cells, Multicube Type, Cell Count, Size (GB), Id, Dev Complete |
| `FOLDERS_GRID` | 137000000003 | Parent Folder, Comments, Id (+ rowEntityLongId, rowLabel, если есть строки) |
| `CUBES_GRID` | 137000000004 | rowEntityLongId, rowLabel, Multicube, **Formula**, Format (JSON), Comments, Code, Summary, Time Summary, User Lists, Time Scale, Versions, Cube Subset, DCA Formula, Use Switchover, Use Workflow, Storage Format (JSON), Style, Referenced By, Cell Count, Size (GB), User Context Dimension, Omit Dimensions, Clean Cells, Repair Cells, Id, AutoCalculation, Last Update, Obfuscation Formula, Stable View Formula, User View Formula, Error Comment |
| `VERSIONS_GRID` | 137000000013 | rowEntityLongId, rowLabel, Item Name, Code, Debug (JSON), Id, Actual(s), SwitchOver, Comments |
| `VERSIONS_SUBSETS_GRID` | 137000000014 | Code, Comments, Element Count, Id, Production, Dev Complete |
| `DASHBOARDS_GRID` | 137000000022 | rowEntityLongId, rowLabel, Folder, Referred By, Comments, Production, Dev Complete, Id |
| `CONTEXT_TABLES_GRID` | 137000000024 | Folder, Comments, Production, Dev Complete, Id |
| `MODEL_USERS_GRID` | 137000000027 | rowEntityLongId, rowLabel (e-mail), First Name, Last Name, User Role, Modeler, Administrator, Service, Comments, Config, User Ban, User Ban Message, Id |
| `WORKSPACE_USERS_GRID` | 137000000028 | те же, что у MODEL_USERS_GRID |
| `BACKUPS_GRID` | 137000000030 | rowEntityLongId, rowLabel (`#47329`), User, Description, Date & Time, File Size MB, Memory GB, Comments, Model Version, Model Version Comment, Id |
| `DASHBOARD_MULTICUBES_GRID` | 137000000080 | rowEntityLongId, rowLabel, Folder, Comments, Namespace, Namespace Type, View Type, Saved View Owner, Referenced Objects Ids, Referenced Objects, View Id, Saved View Creator/Updater, Production, Dev Complete |
| `LIST_PROPERTIES_GRID` | 137000000032 | **не работает**: 400 `viewReference entity not valid`, в том числе с `context_entity_long_id` справочника |
| `TIME_DAYS/WEEKS/MONTHS/PERIODS/YEARS_SUBSETS_GRID` | 137000000040–44 | **не работают**: та же ошибка, с контекстом тоже |
| не из enum: оптимизационные запросы | 137000000051 | rowEntityLongId, rowLabel, Optimizer Engine, Objectives, Variables, Constraints, Comments, Id |

Особенности **[стенд]**:
- **Пустой грид приходит без колонок `rowEntityLongId` и `rowLabel`.** Код, который на них опирается,
  должен проверять наличие колонок.
- **`CUBES_GRID` смешивает строки мультикубов и кубов.** У строки мультикуба `Multicube` и `Formula` пустые,
  у куба в `Multicube` — имя родителя. Колонка `Formula` даёт все формулы модели одним запросом (~1 с).
- **`DASHBOARD_MULTICUBES_GRID` — не связи дашбордов**, а список сущностей модели и их вью
  (справочники, мультикубы, вью по умолчанию). Связи дашборд → мультикуб — через
  `get_dashboard_multicube_links()` (раздел 10).
- **`row_count` — размер страницы, а не лимит:** с `row_count=5` пришли все 11 строк.
- Числовой id работает вместо enum: `export_df(137000000002)`.
- **`exporter.export_many()` без аргументов всегда падает** через ~11 с: в список по умолчанию входят
  нерабочие гриды, а одна ошибка валит весь вызов. Передавайте список явно:
  `export_many([GridTypeIdentifiers.MULTICUBES_GRID, GridTypeIdentifiers.CUBES_GRID])` →
  `dict` с ключами-именами enum (`"MULTICUBES_GRID"`).
- `export_many_as_one_df([...])` склеивает гриды в один DataFrame, добавляя `_grid_name`, `_grid_id`;
  колонки — объединение колонок всех гридов.

---

## 5. Мультикубы

### Обычный экспорт и экспорт «формата ОМ»

| | `export_file` / `export_df` | `export_om_file` |
|---|---|---|
| Что выгружает | Грид в раскладке сохранённого вью | Плоскую таблицу напрямую из хранилища |
| Начало файла | Пустая строка; строка с именами измерений по осям (`Versions;Cubes;Months`); строка заголовков колонок (`;;"Jan 21";…`); дальше данные | Одна строка заголовка: измерения + кубы (`Versions;Months;куб_1;куб_2;куб_3;bool_cube`) |
| Агрегаты времени | Как в вью (в тесте — только месяцы) | **Включены**: строки `FY21`, `YTD` |
| Булевы значения | `true` / `false` | `1` / `0` |
| Форматы | `CSV` (по умолч.), `TXT` (то же содержимое), `XLSX`, `XLS` | `OM_CSV`, `OM_TXT`, `OM_ZIP` |
| Фильтры | `filter_boolean_cube=[cube_id, dim_id]`, `filter_formula=[formula, dim_id]` — см. ниже | `boolean_cube_filter_long_id=cube_id`, `filter_formula="formula"` — **работают** |

Подробности форматов **[стенд]**:
- `DownloadFormat.XLS` на деле даёт XLSX (zip внутри) с расширением `.xls`.
- `DownloadFormatOM.OM_ZIP` даёт файл `SRC_MC.gz`, внутри которого **zip** (не gzip) с CSV
  `SRC_MC--2026.10.12--01.52.csv`.
- `delimiter=","` и `encoding=EncodingTypeExport.WINDOWS_1251` работают. BOM в UTF-8 нет.
- Строковые значения заголовков в кавычках (`"Jan 21"`), числа без.

### `export_df` сломан: обход

`mc.export_df()` читает CSV грида через `pd.read_csv(skiprows=2)` без разделителя, а файл приходит
с `;`. Результат — DataFrame из **одной колонки**, где вся строка `Actual;куб_1;1;1;…` — одна ячейка
**[стенд]**. То же у `view.export_df()`. У `catalog.export_df()` проблемы нет.

Лишние kwargs передаются в `pd.read_csv`, поэтому обход простой **[стенд]**:

```python
df = await mc.export_df(sep=";")        # или delimiter=";"
# 14 колонок: 'Unnamed: 0' (Versions), 'Unnamed: 1' (Cubes), 'Jan 21' … 'Dec 21'
df = df.rename(columns={"Unnamed: 0": "Versions", "Unnamed: 1": "Cubes"})
```

Имена измерений по строкам в DataFrame теряются (`Unnamed: N`), их берите из второй строки файла.
Все значения приходят строками.

### Фильтры

- **В `export_om_file` оба фильтра работают:** с булевым кубом, равным TRUE только в январе и
  феврале, выгрузились только `Jan 21` и `Feb 21` **[стенд]**.
- **В `export_file` и `export_df` фильтры по проверенным измерениям не сработали** **[стенд]**:
  - `filter_boolean_cube` с измерением времени (13000000003) или версий (20000000020) файл не меняет;
  - `filter_formula` убирает **все** элементы указанного измерения: со временем остаются колонки без
    месяцев, с версиями — заголовки без строк;
  - с несуществующим id измерения — `AttributeError` из `dataclasses_json`.
  
  [?] Вероятно, фильтр ждёт булев куб или формулу, у которых нет других измерений, кроме
  фильтруемого (в примерах документации это измерение-справочник по строкам). Для фильтрации
  данных используйте `export_om_file`.

### Импорт, очистка, копирование

| Метод | Поведение **[стенд]** |
|---|---|
| `clear()` | 0,8 с. Значения становятся 0, строки в гриде остаются |
| `import_om_file(path)` | Круговой тест (экспорт ОМ → `clear` → импорт) вернул значения полностью |
| `import_file(path)` | Принимает CSV грида без ошибки |
| `import_om_file(файл грида)` | **Ошибки нет** при файле не того формата. Результат не проверялся — не полагайтесь на то, что SDK поймает неверный файл |
| `import_from_multicube(src_id)` | Скопировал значения `SRC_MC` в `DEST_MC` (одинаковые время, версии и имена кубов), 1,4 с |
| `export_to_multicube(dst_id)` | Отработал без ошибки, 1,4 с |
| `change_cell_value(value, cube_id, dimension_id)` | Не проверялся: смысл `dimension_id` не ясен |

`import_om_file` использует action `IMPORT_STORAGE_VIEW` с зашитыми `;`, десятичной точкой и UTF-8,
колонки сопоставляются по заголовкам **[код]**. `import_file` — `IMPORT_FILE_IN_COMMON` **[код]**.

### Вью мультикуба

`mc.get_views()` → `list[View]` (`name`, `id`, `model_id`, `multicube_id`). Вью умеет то же, что
мультикуб: `export_file`, `export_om_file`, `export_df` (с тем же багом), `import_file`,
`import_om_file` **[стенд]**. `mc.export_views(path)` пишет по файлу на вью: `SRC_MC view 1.csv`.

---

## 6. Справочники (catalogs)

### Чтение **[стенд]**

| Метод | Возвращает |
|---|---|
| `await catalog.info()` | `CatalogInfo`: name, id, model_id |
| `get_elements()` | `(список из 8 имён колонок, список строк)`. Колонки начинаются с `Item Name`, `List`, `Parent`. В строке 3 строки, 2 `None`, 3 `int`. **`row_count`/`column_count` не ограничивают:** с `row_count=3, column_count=2` пришли все 5 строк и 8 колонок |
| `export_df()` | DataFrame, 9 колонок: `Unnamed: 0` (ключ элемента), Item Name, List, Parent, Hierarchy Level, Dimension Rank, Code, Debug, Id |
| `export_file(path)` | По умолчанию **XLSX** (у мультикуба — CSV). CSV: заголовок `;"Item Name";List;Parent;…`, колонка `Debug` — многострочный JSON в кавычках (id, longId, parentLongId, nextLongId, lvl, rank, accessLevelId, isParent) |
| `export_om_file(path)` | **404 «не удалось получить идентификатор файла»** во всех форматах. Для справочника и его вью не работает |
| `get_views()`, `get_view_by_name()` | `CatalogView` (`id`, `name`, `model_id`, `catalog_id`) или `None` |
| `export_views(path)` | Файл на каждое вью, в XLSX |

Id элементов у каждого справочника со своим префиксом: 206000000001…, 207000000001…, 211000000001…
`Debug` в CSV ломает построчное чтение файла — читайте через `pandas` или берите `export_df`.

### Запись **[стенд]**

| Метод | Поведение |
|---|---|
| `model.create_catalogs(names=[...], position=, relative_to=)` | Создаёт, возвращает `None`. Объект берите через `get_catalog_by_name` |
| `catalog.create(model_id, names=[...])` | То же через объект справочника |
| `add_elements(names=[...])`, `add_elements(count=N)`, `position=BEFORE, relative_to=` | Работают, возвращают `None` (id новых элементов не возвращаются) |
| `rename_elements(names, elements_ids)` | Работает |
| `delete_elements(ids=[...])`, `delete_elements(names=[...])` | Работают. Несуществующее имя молча пропускается |
| `clear()` | Удаляет все элементы |
| **`import_file(path)`** | **Не добавляет элементы.** После `clear` импорт собственного CSV-экспорта, собственного XLSX-экспорта и простого CSV (`Item Name` + 3 имени) дал 0 элементов, ошибок нет |
| `export_catalog(dst_id)` / `import_catalog(src_id)` | Ошибок нет, но из 3 элементов источника в приёмник попал **1**. Какой и почему — не выяснено |
| `rename([...])`, `model.rename_catalogs(names, ids)` | Работают |
| `delete()`, `model.delete_catalogs(ids)` | Работают |

**Вывод:** надёжный способ наполнить справочник — `add_elements`. Импорт и копирование справочников
через SDK пока считать нерабочими.

---

## 7. Формулы

```python
from os_ws_sdk.objects.cube_formula_manager import CubeFormulaManager
formulas = CubeFormulaManager.from_model(model)
text = await formulas.get_formula(cube_id)              # = model.get_formula(cube_id)
result = await formulas.change_formula(cube_id, "3")     # = model.change_formula(...)
```

**[стенд]**:
- `get_formula` → `str`, ~1 с. id можно передать строкой.
- `change_formula` → `int`: `0`, если формула та же, `1`, если изменена. ~5 с.
- Синтаксическая ошибка → `WorkspaceException(400, "Не удалось изменить формулу куба id=…:
  CXP: Can't parse formula: … Expected formula format: Number")`.
- Параметра `conditional_formula` из документации нет **[код]**.
- `get_formula` при каждом вызове листает `GET_CUBES_INFO` до нужного куба: для массового чтения
  берите `CUBES_GRID` **[код]**.

---

## 8. Права, UAM, пользователи

### Матрицы прав **[стенд]**

| Метод | Грид |
|---|---|
| `get_multicube_permissions(id=None)` | 137000000019 |
| `get_version_permissions(id=None)` | 137000000015 |
| `get_list_permissions(id=None)` | 137000000012 |
| `get_macro_permissions(id=None)` (алиас `get_macros_permissions`) | 137000000045 |

Формат: DataFrame `rowEntityLongId`, `rowLabel`, затем по колонке на роль, **колонка называется
именем роли** (`SDK_PROBE_ROLE`). Значения `write` / `read` / `none`. С `id` — одна строка. Если в модели
нет ролей, остаются только две первые колонки. Матрица макросов включает системные действия
(`Open`, `Import File`, `Export`, `Create Item`, `Delete Item` …) и скрипты модели.

`PermissionGridExporter.from_model(model).export(grid_id)` → `PermissionGridData(dataframe, role_columns)`,
где `role_columns` — список `PermissionRoleColumn(role_id, role, column)`. Это способ узнать
**id роли по имени**.

`get_user_permissions()` (алиас `get_userpermisions`) → DataFrame `user`, `object_type`, `object_id`,
`access_type`, `role`; ~5 с. Администраторы (роль 19000000000) в него не попадают, с
`ignore_missing_roles=False` вызов падает на первом администраторе.

### Изменение прав **[стенд]**

```python
from os_ws_sdk.objects.permission_change_manager import PermissionChangeManager
pm = PermissionChangeManager.from_model(model)
await pm.change_multicube_permission(mc_id, role_id, "read")   # те же методы есть у model
```

- `change_multicube_permission`, `change_list_permission`, `change_version_permission` — `write` /
  `read` / `none` в любом регистре. ~2 с, возвращают `True`.
- **`True` возвращается и при установке того же значения повторно** — это не признак изменения.
- `role_id` — только число. Имя роли → `WorkspaceException(400, "ID роли должен быть целым")`.
- Неверное значение → `400 "permission должен быть одним из: write, read, none"` (на клиенте).
  Несуществующая роль → `404 "Роль с id=… не найдена в матрице доступов"`.
- **Макросы:** `change_macros_permission` (алиасы `change_macro_permission`, `change_mscros_permission`)
  принимает `"read"`, `"write"`, `"false"`, `True`, `False`. **Значение `"none"` SDK отвергает**
  с ошибкой «должен быть одним из: true, false, write, read, none», хотя `none` в этом списке есть.
  Снимайте доступ через `False`.

### UAM **[стенд]**

`UamManager.from_model(model)`: `get_enabled_list_ids()`, `get_uam_grid_mapping()`, `get_uam(list_id)`,
`change_uam(list_id, item_id, user, permission)`; у `model` есть `get_list_uam`, `get_uam`, `change_uam`.

- **`get_enabled_list_ids()` / `get_list_uam()` возвращают все справочники модели**, включая
  системные, а не только с включённым UAM. `get_uam` на справочнике без UAM →
  `400 "Dimension '…' user cube not found"`.
- `get_uam_grid_mapping()` → `dict` `{list_id: uam_grid_id}`.
- `get_uam(list_id)` → DataFrame: строки — элементы справочника (включая `TOTAL`), колонки —
  e-mail пользователей, значения `write` / `read` / `none`. ~2 с.
- `change_uam(list_id, item_id, "user@mail", "read")` → `True`, ~4,6 с. Пользователя можно указать
  e-mail. Несуществующий → `404 "Пользователь '…' не найден в UAM-grid"`.

### Пользователи **[стенд]**

`get_model_user_list()` / `get_ws_user_list()` → DataFrame с колонками `MODEL_USERS_GRID`.
`get_model_user_info(id)` / `get_ws_user_info(id)` → DataFrame из одной строки; `id` — `rowEntityLongId`
пользователя (111000000…). Без прав на пользователей WS — `Grid access denied` **[док]**.

---

## 9. Скрипты **[стенд]**

| Метод | Возвращает |
|---|---|
| `get_script_list()` | `list[int]` — id скриптов |
| `get_script_info(id)` | DataFrame из одной строки, 16 колонок: rowEntityLongId, rowLabel (имя), **Macros (исходный код скрипта)**, Folder, Scheduler, Lock Mode, Last Date, Last Time, Last Status, Memory Limit MB, Time Limit sec, Custom Offset, Comments, Production, Dev Complete, Id. Нет скрипта — `None` |
| `run_script(id)` | **Ждёт окончания скрипта** и возвращает его вывод строкой (`console.log("hi!")` → `"hi!"`). ~1 с на простом скрипте. Несуществующий id → `KeyError: 'output'` |
| `get_script_history(...)` | DataFrame: History Date, History Time, History Status, History Elapsed Time, History User, History Type, History Request Id; у непустого результата спереди ещё rowEntityLongId, rowLabel (`#299283 scr1`) |
| `get_script_chain_history(...)` | То же для цепочек |

- Фильтры истории `date_from`, `date_to`, `status`, `run_type`, `show_archive` работают.
  Без `date_from` SDK сам подставляет «сутки назад».
- **`identifiers=[...]` сервер отвергает:** `400 "The property identifiers is not defined"`. Для истории
  одного скрипта фильтруйте DataFrame по `rowLabel`.
- Запуск через `run_script` пишется в историю с типом `INTERFACE`, а не `API`.
- `build_scripts_history_filter(...)` показывает фильтр, который уйдёт на сервер:
  `{"type": "SCRIPTS_HISTORY_FILTER", "params": {"dateFrom": <unix сек>, "runType": 2, "status": 3, "showArchive": true}}`.

---

## 10. Дашборды **[стенд]**

| Метод | Возвращает |
|---|---|
| `get_dashboard_multicube_links(dashboard_ids=None, request_delay=0.5, ignore_errors=True)` | DataFrame: dashboardId, optionId, objectIdentifier, type (`cube`), name, multicubeId, gridId, savedViewId, positionX/Y/W/H. ~7 с на 4 дашборда |
| `get_dashboard_multicubes(dashboard_id)` | То же для одного дашборда, ~1,3 с |
| `export_dashboard_multicube_links_grid()` | Грид 137000000080 (см. раздел 4) — **не связи дашбордов** |

- По умолчанию `ignore_errors=True` молча пропускает дашборды без содержимого. С `False` вызов падает
  на первом пустом дашборде (`нет definitionJsonString`).
- **В модели без дашбордов** вызов без `dashboard_ids` бросает `404 "Не удалось автоматически найти
  dashboard longId… Передайте dashboard_ids явно"`, а не возвращает пустой DataFrame.

---

## 11. Оптимизация **[стенд]**

| Метод | Поведение |
|---|---|
| `get_optimizer_requests_grid()` и алиас `get_optimizer_requests()` | **Без параметра падают** (`400 viewDefinition not found`). Работают с `entity_default_view_entity_long_id=137000000051`: колонки optimizerRequestId, requestName, Optimizer Engine, Objectives, Variables, Constraints, Comments, Id |
| `get_optimizer_request_fields(id, entity_default_view_entity_long_id=137000000051)` | Тоже требует параметр. Колонки optimizerRequestId, requestName, goals, variables, constraints |
| `export_lp_mapping(id, path)` | На пустом запросе → `404 "не удалось получить идентификатор для скачивания LP файла"`. На заполненном запросе стендом не проверялся; по коду возвращает **имя** zip с `optimizer.lp` и `optimizer.map`, таймаут 5 мин **[код]** |
| `import_solution_mapping(zip)` | Не проверялся. **Формат CSV решения не описан нигде** — остаётся открытым вопросом |

LP из Optimacros содержит дефекты (голые переменные в Bounds, инвертированные границы, константа
в цели). Перед сторонним солвером — `lp_preprocess.py` (файлы 07–08 проекта) **[код]**.

---

## 12. Бэкапы и управление моделью **[стенд]**

| Метод | Поведение |
|---|---|
| `get_backup_list()` | DataFrame `BACKUPS_GRID`; `rowLabel` — короткий id (`#47329`), `rowEntityLongId` — длинный (41000047329) |
| `get_backup_info(long_id)` | DataFrame из одной строки |
| `get_backup("#47329")` | `bytes`, zip (130 КБ за 2 с). Короткий id со знаком `#` |
| **`backup()`** | **Возвращает `None` мгновенно, новый бэкап в списке не появляется.** Похоже, не работает |
| `recalculate()` | `True`, 0,7 с |
| `stop()` | `True`, 1,7 с. Повторно на офлайн-модели — `False` (6,7 с). `is_offline` обновляется и у того же объекта |
| Экспорт из остановленной модели | `AttributeError: 'NoneType' object has no attribute 'items'` |
| `start()` | `True`, **22,6 с**. На онлайн-модели тоже `True` |
| `restore()` | Не проверялся: без параметров неясно, из какого бэкапа он восстанавливает |

---

## 13. Чего в SDK нет или что не работает: обходы

| Задача | Как сделать |
|---|---|
| DataFrame из мультикуба или вью | `export_df(sep=";")` или `export_om_file` + `pd.read_csv(sep=";")` |
| Наполнить справочник | `add_elements(names=[...])` |
| Выгрузка справочника в формате ОМ | `export_file(format=CSV)` или `export_df()` |
| Все формулы модели | `CUBES_GRID`, колонка `Formula` |
| Исходный код скриптов | `get_script_info(id)["Macros"]` |
| История одного скрипта | Общая история + фильтр по `rowLabel` |
| Id роли по имени | `PermissionGridExporter.export(137000000019).role_columns` |
| Снять доступ к макросу | `change_macros_permission(id, role_id, False)` |
| Связи дашборд → мультикуб | `get_dashboard_multicube_links()` с `ignore_errors=True` |
| Отфильтрованная выгрузка | `export_om_file(boolean_cube_filter_long_id=…)` или `filter_formula="…"` |
| Свойства справочника, подмножества времени | Через SDK не работают (`LIST_PROPERTIES_GRID`, `TIME_*_SUBSETS_GRID`) |
| Создать мультикуб, куб, измерение, версию | Через SDK никак |
| Произвольный action API | Методы коннектора (раздел 1) — недокументированный путь |

---

## 14. Ловушки (сводка)

1. `mc.export_df()` / `view.export_df()` — одна колонка. Обход `sep=";"`.
2. `catalog.import_file()` молча не добавляет элементы.
3. `export_catalog` / `import_catalog` копируют не все элементы.
4. `export_om_file` у справочника и его вью — 404.
5. Экспорт в несуществующую папку — ни файла, ни ошибки.
6. «Не найдено»: `None` или исключение, зависит от метода (раздел 2).
7. Права на макрос: `"none"` отвергается, нужен `False`.
8. Роль — только числовой id.
9. `change_*_permission` и `start()` возвращают `True`, даже если ничего не поменялось.
10. `get_script_history(identifiers=…)` — ошибка сервера.
11. `get_optimizer_requests_grid()` и `get_optimizer_request_fields()` без `137000000051` падают.
12. `exporter.export_many()` без аргументов падает целиком.
13. `get_enabled_list_ids()` возвращает и справочники без UAM.
14. `get_dashboard_multicube_links()` по умолчанию прячет ошибки; без дашбордов — 404.
15. `run_script` с неверным id — `KeyError`, неверные id в фильтрах и запросы к остановленной
    модели — `AttributeError`. Ловите `Exception`.
16. `model.backup()` не создаёт бэкап.
17. `ws.info()` не содержит список моделей.
18. `OM_ZIP` — расширение `.gz`, внутри zip. `XLS` — на деле XLSX.
19. Экспорт ОМ включает агрегаты времени (`FY21`, `YTD`) — отфильтруйте их перед обработкой.
20. Пустой системный грид приходит без `rowEntityLongId` / `rowLabel`.
21. `row_count` у гридов и `get_elements` — не лимит.
22. Объекты SDK — снимки: после переименования старый объект отдаёт старое имя.
23. Формулы-фильтры в `.env` пишите в двойных кавычках — `python-dotenv` молча пропускает строку с
    одинарными кавычками внутри.

---

## 15. Что осталось непроверенным

- `export_lp_mapping` на заполненном оптимизационном запросе и `import_solution_mapping` (формат решения).
- `change_cell_value` — смысл `dimension_id`.
- `restore()`.
- Правильный `dimension_id` для фильтров `export_file`.
- Поведение на Windows и на больших моделях.

## 16. Как перепроверить: стенд `sdk_probe.py`

```bash
pip install -r requirements.txt
cp .env.example .env    # заполнить
python sdk_probe.py     # первый прогон: OM_SKIP_WRITE=1, OM_TEST_ADMIN=0
```

Результат в `OM_WORK_DIR`: `report.md` (таблица по всем вызовам), `report.json` (то же для машины),
`results/*.csv` (полученные DataFrame), `export/<группа>/<вызов>/` (файлы каждого экспорта).

| Переменная | По умолчанию | Что включает |
|---|---|---|
| `OM_SKIP_WRITE` | 1 | 0 = запись: временные справочники (удаляются), круговой импорт в мультикуб-приёмник с восстановлением, формулы, права и UAM с откатом, `run_script` |
| `OM_TEST_ADMIN` | 0 | 1 = `recalculate`, `backup`, `stop` / `start` |
| `OM_TEST_RESTORE` | 0 | 1 = `restore()` — откатывает модель |
| `OM_GROUPS` | все | Запуск части групп |

Запускать только на тестовой модели. При выходе новой версии SDK или Workspace прогон повторяется;
расхождения с этой статьёй — повод её обновить.
