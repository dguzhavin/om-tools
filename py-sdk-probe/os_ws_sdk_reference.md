# os_ws_sdk 1.0: как SDK работает на самом деле

Официальная документация к Python SDK Optimacros рассказывает, как SDK задуман. Мы захотели знать,
как он себя ведёт на живом Workspace, и прогнали через стенд все публичные методы. Здесь собрано то,
что получилось: что возвращает каждый метод, в каком формате, сколько времени занимает, где падает
и в чём расходится с документацией.

Версии: `os_ws_sdk` 1.0, Workspace 9.200 (storage 9.200.164.73), Python 3.13, pandas 3.0, macOS.
На Windows стенд пока не запускали.

Стенд, отчёты и тестовые настройки лежат в репозитории
[om-tools/py-sdk-probe](https://github.com/dguzhavin/om-tools/tree/main/py-sdk-probe). Прогонов было
четыре, 12 октября 2026, почти 700 вызовов на тестовой модели `PY_SDK ##1.0`: сначала только чтение,
потом запись с откатом изменений, потом остановка и запуск модели. Сырые результаты каждого прогона
лежат в [reports/](https://github.com/dguzhavin/om-tools/tree/main/py-sdk-probe/reports), их удобно
открывать, когда нужен точный текст ошибки или полный список колонок.

Всё, что написано ниже, мы видели своими глазами на сервере. Если вывод сделан по исходному коду SDK
или это догадка, так и сказано в тексте. Модель маленькая (три мультикуба по сотне ячеек), поэтому
время в таблицах показывает накладные расходы API. Скорость на больших объёмах мы не мерили.

## Сначала главное

Если прочитать только этот раздел, вот что сэкономит больше всего времени:

1. `mc.export_df()` и `view.export_df()` сломаны и возвращают DataFrame из одной колонки. Лечится
   одним аргументом: `await mc.export_df(sep=";")`.
2. `catalog.import_file()` не добавляет элементы в справочник. Ошибки нет, справочник остаётся пустым.
   Элементы создаём через `add_elements`.
3. `export_om_file` работает только у мультикубов и их вью. У справочника он отвечает 404.
4. Если папки для экспорта нет, SDK молча ничего не делает: ни файла, ни ошибки.
5. Когда объект не найден, одни методы возвращают `None`, другие бросают исключение. Таблица ниже,
   в разделе про общее поведение.
6. Права на макрос нельзя снять строкой `"none"`, только `False`.
7. Исходный код скрипта достаётся через `get_script_info(id)`, колонка `Macros`. В заметках
   раньше было написано, что такого метода нет. Есть.
8. Все формулы модели приходят одним запросом из системного грида `CUBES_GRID`. Вызывать
   `get_formula` в цикле не нужно.
9. `model.backup()` ничего не делает: возвращается мгновенно, нового бэкапа в списке нет.

## Установка и подключение

В PyPI пакета нет, ставим из wheel. Нужен Python 3.10 или новее. Готовый
[requirements.txt](https://github.com/dguzhavin/om-tools/blob/main/py-sdk-probe/requirements.txt)
есть в репозитории, wheel кладётся рядом с ним.

```bash
pip install ./os_ws_sdk-1.0-py3-none-any.whl pandas python-dotenv
```

Свои зависимости (aiohttp, dataclasses_json и прочее) wheel подтянет сам.

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

Что стоит знать про подключение:

- В `WsVersion` есть только `VERSION_9_200`.
- `ConnectorManager.create` и `Workspace(...)` в сеть не ходят. Неверный токен или URL всплывут
  на первом настоящем запросе.
- У логгера SDK (`os_ws_sdk.logging_sdk.logger`) нет своих хендлеров. Его сообщения появятся
  в консоли, только если в скрипте вызван `logging.basicConfig(...)`.
- Политику event loop для Windows ставим так же, как это сделано в тестах самого SDK.

### Импорты

Импортировать из корня пакета нельзя. `from os_ws_sdk import ScriptRunType` из документации,
как и `ScriptStatus`, `Workspace`, `ConnectorManager`, падает с `AttributeError`. Рабочие пути такие:

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

### Значения перечислений

| Enum | Значения |
|---|---|
| `DownloadFormat` | `CSV="csv"`, `TXT="txt"`, `XLS="xls"`, `XLSX="xlsx"` |
| `DownloadFormatOM` | `OM_CSV="csv"`, `OM_TXT="txt"`, `OM_ZIP="gz"` |
| `EncodingTypeExport` | 8 значений, в том числе `WINDOWS_1251="windows-1251"`, `WINDOWS_1251_UPPER="WINDOWS-1251"`, `WIN="win"`, `UTF`, `UTF_8` |
| `EncodingTypeImport` | 6 значений: `WINDOWS_1251="WINDOWS-1251"`, `WIN="win"`, `UTF="utf"`, `UTF_8_UPPER`, `AUTO`, `AUTO_UPPER` |
| `ScriptStatus` | `NONE=0`, `WAIT_IN_QUEUE=1`, `IN_PROGRESS=2`, `SUCCESS=3`, дальше `FAILED`, `ABORTED` |
| `ScriptRunType` | `INTERFACE=0`, `SCHEDULER=1`, `API=2` |
| `Position` | `START="Start"`, `END="End"`, `BEFORE="Before"`, `AFTER="After"` |
| `GridTypeIdentifiers` | 17 системных гридов, список в разделе про модель |

### Из чего сделаны объекты

`Workspace`, `Model`, `Multicube`, `View`, `Catalog` и `CatalogView` написаны как dataclasses на
`dataclasses_json`, поэтому у всех есть `to_dict`, `to_json`, `from_dict`, `from_json` и `schema`.
Объект `Model` из `get_models()` несёт гораздо больше полей, чем возвращает `model.info()`:
`storage_version`, `auto_calc`, `created_at` и `updated_at` в unix-времени, папку модели, режимы
чтения и записи хранилища, `restore_mode`.

У коннектора нашлось 17 методов, которых нет в документации: `get_system_grid_data`,
`get_view_definition`, `get_dashboard_definition`, `get_entity_types`, `download_dataframe`,
`download_backup`, `get_lp_hash`, `get_optimizer_requests_table` и другие. Есть даже
`download_milticube_data` с опечаткой в имени. Сигнатуры мы не разбирали. Это запасной путь на случай,
если публичного метода не хватит, но опираться на него стоит осторожно: в следующей версии его могут
переименовать без предупреждения.

## Общее поведение

### Что ходит в сеть, а что нет

`model.info()` синхронный. `mc.info()`, `view.info()`, `catalog.info()` и `catalog_view.info()`
вызываются через `await`, но на сервер не ходят и отдают то, что уже лежит в объекте. В сеть идёт
только `ws.info()`.

Отсюда следствие, на котором легко споткнуться. Объект SDK хранит снимок на момент получения.
Переименовали справочник через `catalog.rename(...)`, а старый объект по-прежнему отдаёт старое имя.
Свежие данные будут только у нового объекта из `get_catalog(...)`. Единственное исключение, которое
мы видели: `model.stop()` обновляет `is_offline` у того же объекта.

### Если объекта нет

| Вызов | Что вернётся |
|---|---|
| `ws.get_model(id)` | `ModelNotFoundError: Модель model='…' не найдена` |
| `ws.get_model_by_name(name)` | `None` |
| `model.get_multicube(id)`, `get_multicube_by_name(name)` | `None` |
| `model.get_id_multicube_by_name(name)` | `WorkspaceException(404, "Entity by query … not found")` |
| `model.get_catalog(id)`, `get_catalog_by_name(name)` | `WorkspaceException(404, "Каталог с id=… не найден")` |
| `model.get_catalog_view_by_name(...)`, `mc.get_view_by_name(...)`, `catalog.get_view_by_name(...)` | `None` |
| `model.get_script_info(id)` | `None` |
| `model.get_formula(id)` | `WorkspaceException(404, "Куб с id=… не найден")` |
| `model.get_dashboard_multicubes(id)` | `WorkspaceException(404, "В ответе GET_DASHBOARD_DEFINITION нет definitionJsonString…")` |
| `model.delete_catalogs([id])` | `WorkspaceException(404, …)` |
| `catalog.delete_elements(names=[...])` | ничего, несуществующие имена тихо пропускаются |
| `model.get_backup(id)` | `WorkspaceException(500, …)`, в тексте HTML-страница сервера |
| `model.run_script(id)` | `KeyError: 'output'` |

Мультикуб и справочник, которые ищутся одинаково, ведут себя по-разному. Поэтому после любого
`get_*` проверяем и на `None`, и на исключение.

### Ошибки

В `WorkspaceException` лежит пара: `e.args[0]` с кодом в духе HTTP (400, 404, 500) и `e.args[1]`
с текстом, часто по-русски.

Часть проверок SDK делает у себя, не отправляя запрос: недопустимое значение права, роль не числом,
несуществующий файл для импорта (тут будет `FileNotFoundError`). Такие ошибки прилетают за миллисекунды.

Неприятнее, что неверные входные данные иногда дают не `WorkspaceException`, а
`AttributeError: 'NoneType' object has no attribute 'items'` откуда-то из глубин `dataclasses_json`.
Мы ловили его на неверном id измерения в фильтре и при экспорте из остановленной модели. Так что
в скриптах ловим `Exception`, а не только `WorkspaceException`.

Ещё один случай: `action_info` для скрипта не бросает исключение, а возвращает обычный `dict`,
внутри которого лежит `modalInfo.isError=True` и сообщение «Action info disabled». Если в ответе
есть `modalInfo`, его стоит проверить руками.

### Сколько это длится

| Операция | Время |
|---|---|
| `get_models` (122 модели), `get_multicubes`, `get_catalogs`, любой `*_by_name` | 0,6–1 с |
| Системный грид | около 1 с |
| Любой экспорт в файл или DataFrame | 2,1–2,7 с |
| `import_om_file`, `import_file`, `clear` | 0,7–1 с |
| `get_formula` | около 1 с |
| `change_formula` | около 5 с |
| Изменение права | около 2 с |
| `change_uam` | около 4,6 с |
| `get_user_permissions` | около 5 с |
| `get_dashboard_multicube_links()` на четырёх дашбордах | около 7 с |
| `stop()` / `start()` | 1,7 с / 22,6 с |

Поиск по имени занимает столько же, сколько загрузка полного списка. Похоже, `*_by_name` каждый раз
тянет весь список заново (у `ws.get_model` это видно по коду). Если объект нужен несколько раз,
сохраняем его в переменную.

Параллельность работает. Два экспорта через `asyncio.gather` заняли 2,4 с, то есть столько же,
сколько один. Лимит Workspace видно в `ws.info()`: `concurrency_max=30`.

### Файлы

Во всех `export_*` аргумент `path` означает папку. Имя файла SDK придумывает сам по имени объекта:
`SRC_MC.csv`, `SRC_MC view 1.csv`. Путь к файлу метод не возвращает.

Если папки нет, SDK её не создаёт, файл не пишет и об ошибке не сообщает. Поэтому перед каждым
экспортом:

```python
Path(dir).mkdir(parents=True, exist_ok=True)
```

Два экспорта одного объекта в одну папку дадут файлы с одинаковым именем. Перезапись мы не
проверяли, но на всякий случай разводим такие экспорты по разным папкам.

## Workspace

| Метод | Что возвращает |
|---|---|
| `await ws.info()` | `WsInfo`: имя и id воркспейса, лимит и текущий размер (в байтах), `storage_version`, теги (`tag_summaries`), настройки аудита, `concurrency_max`, `check_license_status`. Поля `model_ids`, `model_names`, `model_summaries` приходят пустыми |
| `await ws.get_models()` | `list[Model]`, все модели воркспейса. 122 модели за 0,9 с |
| `await ws.get_model(id)` | `Model` или `ModelNotFoundError` |
| `await ws.get_model_by_name(name)` | `Model` или `None` |

Список моделей берём из `get_models()`. В `ws.info()` его нет, хотя поля для него там есть.

## Модель: структура и метаданные

| Метод | Что возвращает |
|---|---|
| `model.info()` | `ModelInfo`: model_name, cell_count, memory, model_id, model_definition_serial_number, is_offline. Синхронный |
| `get_multicubes()` | `list[Multicube]`, у каждого `name`, `id`, `model_id` |
| `get_multicube(id)`, `get_multicube_by_name(name)` | `Multicube` или `None` |
| `get_id_multicube_by_name(name)` | `int` или `WorkspaceException(404)` |
| `get_catalogs(start=0, count=-1)` | `list[Catalog]`, включая системные `Roles`, `Users`, `Access Levels`. Пагинация работает: `count=3` вернул три справочника из пяти |
| `get_catalog(id)`, `get_catalog_by_name(name)` | `Catalog`. В поле `unknown_things` лежит сырой ответ сервера: `itemCount`, `isFlat`, `isNumber`, `sortOrder`, `subsetEntityLongIds` и другое |
| `get_catalog_views(catalog_id)` | `list[CatalogView]`, около 1,3 с |
| `get_catalog_view_by_name(catalog_id, name)` | `CatalogView` или `None` |
| `action_info(action_id)` | `dict`. Для оптимизационного запроса в нём `name`, `savedActionLongId`, `objectives`, `variables`, `constraints`, `optimizerLongId`, `options`. Для скрипта приходит ошибка внутри `dict` |

### Системные гриды

Системные гриды выручают там, где отдельного метода нет. Например, все формулы модели проще всего
достать отсюда.

```python
from os_ws_sdk.objects.system_grid_exporter import SystemGridExporter, GridTypeIdentifiers
exporter = SystemGridExporter.from_model(model)
df = await exporter.export_df(GridTypeIdentifiers.CUBES_GRID)
```

| Грид | ID | Колонки |
|---|---|---|
| `MULTICUBES_GRID` | 137000000002 | rowEntityLongId, rowLabel, Code, Folder, Comments, User Lists, Time Scale, Versions, Cube Subset, Clean Cells, Multicube Type, Cell Count, Size (GB), Id, Dev Complete |
| `FOLDERS_GRID` | 137000000003 | Parent Folder, Comments, Id (плюс rowEntityLongId и rowLabel, если папки есть) |
| `CUBES_GRID` | 137000000004 | rowEntityLongId, rowLabel, Multicube, Formula, Format (JSON), Comments, Code, Summary, Time Summary, User Lists, Time Scale, Versions, Cube Subset, DCA Formula, Use Switchover, Use Workflow, Storage Format (JSON), Style, Referenced By, Cell Count, Size (GB), User Context Dimension, Omit Dimensions, Clean Cells, Repair Cells, Id, AutoCalculation, Last Update, Obfuscation Formula, Stable View Formula, User View Formula, Error Comment |
| `VERSIONS_GRID` | 137000000013 | rowEntityLongId, rowLabel, Item Name, Code, Debug (JSON), Id, Actual(s), SwitchOver, Comments |
| `VERSIONS_SUBSETS_GRID` | 137000000014 | Code, Comments, Element Count, Id, Production, Dev Complete |
| `DASHBOARDS_GRID` | 137000000022 | rowEntityLongId, rowLabel, Folder, Referred By, Comments, Production, Dev Complete, Id |
| `CONTEXT_TABLES_GRID` | 137000000024 | Folder, Comments, Production, Dev Complete, Id |
| `MODEL_USERS_GRID` | 137000000027 | rowEntityLongId, rowLabel (e-mail), First Name, Last Name, User Role, Modeler, Administrator, Service, Comments, Config, User Ban, User Ban Message, Id |
| `WORKSPACE_USERS_GRID` | 137000000028 | те же, что у `MODEL_USERS_GRID` |
| `BACKUPS_GRID` | 137000000030 | rowEntityLongId, rowLabel (`#47329`), User, Description, Date & Time, File Size MB, Memory GB, Comments, Model Version, Model Version Comment, Id |
| `DASHBOARD_MULTICUBES_GRID` | 137000000080 | rowEntityLongId, rowLabel, Folder, Comments, Namespace, Namespace Type, View Type, Saved View Owner, Referenced Objects Ids, Referenced Objects, View Id, Saved View Creator, Saved View Updater, Production, Dev Complete |
| `LIST_PROPERTIES_GRID` | 137000000032 | не работает: 400 `viewReference entity not valid`, с `context_entity_long_id` справочника тоже |
| `TIME_DAYS/WEEKS/MONTHS/PERIODS/YEARS_SUBSETS_GRID` | 137000000040–44 | не работают, та же ошибка, контекст не помогает |
| оптимизационные запросы (нет в enum) | 137000000051 | rowEntityLongId, rowLabel, Optimizer Engine, Objectives, Variables, Constraints, Comments, Id |

Несколько вещей, о которых документация молчит:

- У пустого грида нет колонок `rowEntityLongId` и `rowLabel`. Код, который на них опирается, должен
  сначала проверить, что они есть.
- В `CUBES_GRID` строки мультикубов идут вперемешку со строками кубов. У мультикуба колонки
  `Multicube` и `Formula` пустые, у куба в `Multicube` стоит имя родителя. Все формулы модели
  приходят одним запросом примерно за секунду.
- `DASHBOARD_MULTICUBES_GRID`, вопреки названию, не про дашборды. Там список сущностей модели
  и их вью: справочники, мультикубы, вью по умолчанию. Связи дашбордов с мультикубами отдаёт
  `get_dashboard_multicube_links()`, о нём ниже.
- `row_count` задаёт размер страницы, а не лимит. С `row_count=5` пришли все 11 строк.
- Вместо enum можно передать число: `export_df(137000000002)`.
- `exporter.export_many()` без аргументов падает всегда, примерно через 11 секунд. В список по
  умолчанию входят нерабочие гриды, и одной ошибки хватает, чтобы упал весь вызов. Список передаём
  явно: `export_many([GridTypeIdentifiers.MULTICUBES_GRID, GridTypeIdentifiers.CUBES_GRID])`.
  Вернётся `dict`, где ключи совпадают с именами enum (`"MULTICUBES_GRID"`).
- `export_many_as_one_df([...])` склеивает гриды в один DataFrame и добавляет колонки `_grid_name`
  и `_grid_id`.

## Мультикубы

### Два вида экспорта

У мультикуба два способа выгрузки, и разница между ними больше, чем кажется по названиям.

| | `export_file` / `export_df` | `export_om_file` |
|---|---|---|
| Что выгружает | Грид в раскладке сохранённого вью | Плоскую таблицу прямо из хранилища |
| Начало файла | Пустая строка, потом строка с именами измерений по осям (`Versions;Cubes;Months`), потом заголовки колонок (`;;"Jan 21";…`), потом данные | Одна строка заголовка: измерения и кубы (`Versions;Months;куб_1;куб_2;куб_3;bool_cube`) |
| Агрегаты времени | Как во вью (у нас были только месяцы) | Есть: строки `FY21` и `YTD` |
| Булевы значения | `true` / `false` | `1` / `0` |
| Форматы | `CSV` (по умолчанию), `TXT`, `XLSX`, `XLS` | `OM_CSV`, `OM_TXT`, `OM_ZIP` |
| Фильтры | `filter_boolean_cube=[cube_id, dim_id]`, `filter_formula=[formula, dim_id]`, у нас не заработали | `boolean_cube_filter_long_id=cube_id`, `filter_formula="formula"`, работают |

Для преобразований и обратного импорта берём `export_om_file`. Агрегаты `FY21` и `YTD` перед
обработкой отфильтровываем, иначе они попадут в данные.

Мелочи про форматы, которые всплыли по ходу:

- `DownloadFormat.XLS` на самом деле отдаёт XLSX, только с расширением `.xls`.
- `DownloadFormatOM.OM_ZIP` создаёт файл `SRC_MC.gz`, но внутри обычный zip, не gzip. В архиве лежит
  CSV с датой в имени: `SRC_MC--2026.10.12--01.52.csv`.
- `delimiter=","` и `encoding=EncodingTypeExport.WINDOWS_1251` работают как ожидается. BOM в UTF-8 нет.
- `TXT` по содержимому не отличается от `CSV`.

### `export_df` сломан, и как его починить

Это самая неприятная находка. `mc.export_df()` читает CSV грида через `pd.read_csv(skiprows=2)`
и не передаёт разделитель, а файл приходит с `;`. На выходе DataFrame из одной колонки, и в каждой
ячейке целая строка вида `Actual;куб_1;1;1;…`. У `view.export_df()` то же самое. У справочника
`export_df` работает нормально.

К счастью, лишние аргументы SDK передаёт прямо в `pd.read_csv`, поэтому лечится одним параметром:

```python
df = await mc.export_df(sep=";")        # delimiter=";" тоже работает
# колонки: 'Unnamed: 0' (Versions), 'Unnamed: 1' (Cubes), 'Jan 21' … 'Dec 21'
df = df.rename(columns={"Unnamed: 0": "Versions", "Unnamed: 1": "Cubes"})
```

Имена измерений по строкам теряются, поэтому колонки приходится переименовывать руками. Все значения
приходят строками.

### Фильтры

В `export_om_file` оба фильтра работают. Булев куб был равен TRUE только в январе и феврале,
и в выгрузку попали только `Jan 21` и `Feb 21`.

С `export_file` и `export_df` разобраться не получилось. Мы пробовали три измерения:

- `filter_boolean_cube` со временем (13000000003) или версиями (20000000020) никак не меняет файл.
- `filter_formula` выкидывает все элементы указанного измерения. Со временем остаются колонки
  без месяцев, с версиями остаются заголовки без строк.
- С несуществующим id измерения вызов падает с `AttributeError`.

Похоже, фильтру нужен булев куб или формула только по тому измерению, которое фильтруется. В
примерах документации это справочник по строкам. Проверить это на нашей модели было не на чем.
Пока для отфильтрованной выгрузки используем `export_om_file`.

### Импорт, очистка, копирование

| Метод | Как себя ведёт |
|---|---|
| `clear()` | 0,8 с. Значения обнуляются, строки в гриде остаются |
| `import_om_file(path)` | Экспорт ОМ, `clear` и импорт того же файла вернули все значения на место |
| `import_file(path)` | CSV грида принимается без ошибок |
| `import_om_file(файл грида)` | Файл не того формата SDK принимает без ошибки. Что при этом стало с данными, мы не проверяли. Рассчитывать на то, что SDK поймает неверный файл, не стоит |
| `import_from_multicube(src_id)` | Скопировал значения `SRC_MC` в `DEST_MC` (одинаковые время, версии и имена кубов) за 1,4 с |
| `export_to_multicube(dst_id)` | Отработал без ошибок за 1,4 с |
| `change_cell_value(value, cube_id, dimension_id)` | Не проверяли: непонятно, что SDK ждёт в `dimension_id` |

По коду SDK `import_om_file` отправляет action `IMPORT_STORAGE_VIEW`, в котором зашиты разделитель
`;`, десятичная точка и UTF-8. Колонки сопоставляются по заголовкам, поэтому у приёмника должны быть
те же имена измерений и кубов. `import_file` отправляет `IMPORT_FILE_IN_COMMON`.

### Вью мультикуба

`mc.get_views()` возвращает `list[View]` с полями `name`, `id`, `model_id`, `multicube_id`. Вью умеет
всё то же, что мультикуб: `export_file`, `export_om_file`, `export_df` (с тем же багом), `import_file`,
`import_om_file`. `mc.export_views(path)` пишет по файлу на каждое вью, например `SRC_MC view 1.csv`.

## Справочники

В SDK справочник называется `catalog`, потому что `list` в Python уже занято.

### Чтение

| Метод | Что возвращает |
|---|---|
| `await catalog.info()` | `CatalogInfo`: name, id, model_id |
| `get_elements()` | Кортеж: список из восьми имён колонок (начинается с `Item Name`, `List`, `Parent`) и список строк. Параметры `row_count` и `column_count` ничего не ограничивают: с `row_count=3, column_count=2` пришли все 5 строк и 8 колонок |
| `export_df()` | DataFrame из 9 колонок: `Unnamed: 0` (ключ элемента), Item Name, List, Parent, Hierarchy Level, Dimension Rank, Code, Debug, Id |
| `export_file(path)` | По умолчанию XLSX (у мультикуба по умолчанию CSV). В CSV колонка `Debug` содержит многострочный JSON в кавычках |
| `export_om_file(path)` | Не работает: 404 «не удалось получить идентификатор файла», в любом формате, у справочника и у его вью |
| `get_views()`, `get_view_by_name()` | `CatalogView` (`id`, `name`, `model_id`, `catalog_id`) или `None` |
| `export_views(path)` | По файлу XLSX на каждое вью |

Из-за многострочного `Debug` CSV справочника нельзя читать построчно. Читаем через pandas или
сразу берём `export_df`. У каждого справочника свой префикс id элементов: 206000000001…,
207000000001… и так далее.

### Запись

| Метод | Как себя ведёт |
|---|---|
| `model.create_catalogs(names=[...], position=, relative_to=)` | Создаёт справочники и возвращает `None`. Сам объект потом достаём через `get_catalog_by_name` |
| `catalog.create(model_id, names=[...])` | То же самое через объект справочника |
| `add_elements(names=[...])`, `add_elements(count=N)`, `position=BEFORE, relative_to=` | Работают. Id новых элементов не возвращаются |
| `rename_elements(names, elements_ids)` | Работает |
| `delete_elements(ids=[...])`, `delete_elements(names=[...])` | Работают, несуществующие имена тихо пропускаются |
| `clear()` | Удаляет все элементы |
| `import_file(path)` | Не добавляет элементы. Мы очищали справочник и импортировали три файла: его же CSV-выгрузку, его же XLSX-выгрузку и простой CSV с колонкой `Item Name`. Каждый раз 0 элементов и никакой ошибки |
| `export_catalog(dst_id)` / `import_catalog(src_id)` | Ошибок нет, но из трёх элементов источника в приёмник попал один. Какой и почему, не разобрались |
| `rename([...])`, `model.rename_catalogs(names, ids)` | Работают |
| `delete()`, `model.delete_catalogs(ids)` | Работают |

Итог простой: наполнять справочник через SDK можно только `add_elements`. Импорт и копирование
справочников пока считаем нерабочими.

## Формулы

```python
from os_ws_sdk.objects.cube_formula_manager import CubeFormulaManager
formulas = CubeFormulaManager.from_model(model)
text = await formulas.get_formula(cube_id)              # то же, что model.get_formula(cube_id)
result = await formulas.change_formula(cube_id, "3")     # то же, что model.change_formula(...)
```

`get_formula` возвращает строку примерно за секунду, id можно передать и строкой. `change_formula`
работает около пяти секунд и возвращает `0`, если формула не изменилась, или `1`, если изменилась.
Синтаксическую ошибку сервер описывает понятно:
`WorkspaceException(400, "Не удалось изменить формулу куба id=…: CXP: Can't parse formula: … Expected formula format: Number")`.

Параметра `conditional_formula`, который упоминается в документации, в коде SDK нет. По коду же видно,
что `get_formula` при каждом вызове листает `GET_CUBES_INFO` до нужного куба, так что для чтения всех
формул разом удобнее `CUBES_GRID`.

## Права, UAM, пользователи

### Матрицы прав

| Метод | Грид |
|---|---|
| `get_multicube_permissions(id=None)` | 137000000019 |
| `get_version_permissions(id=None)` | 137000000015 |
| `get_list_permissions(id=None)` | 137000000012 |
| `get_macro_permissions(id=None)` (алиас `get_macros_permissions`) | 137000000045 |

Все четыре метода возвращают DataFrame. В нём колонки `rowEntityLongId`, `rowLabel` и по колонке на
каждую роль, причём колонка называется именем роли. Значения `write`, `read` или `none`. Если передать
`id`, останется одна строка. Если ролей в модели нет, останутся только две первые колонки. В матрице
макросов кроме скриптов модели есть системные действия: `Open`, `Import File`, `Export`, `Create Item`,
`Delete Item` и другие.

Чтобы узнать id роли по имени, пригодится
`PermissionGridExporter.from_model(model).export(grid_id)`. Он возвращает
`PermissionGridData(dataframe, role_columns)`, где каждый `PermissionRoleColumn` хранит `role_id`,
`role` и `column`.

`get_user_permissions()` (алиас `get_userpermisions`) отдаёт DataFrame с колонками `user`,
`object_type`, `object_id`, `access_type`, `role` и работает около пяти секунд. Администраторов
с ролью 19000000000 в нём нет. С `ignore_missing_roles=False` вызов падает на первом администраторе.

### Изменение прав

```python
from os_ws_sdk.objects.permission_change_manager import PermissionChangeManager
pm = PermissionChangeManager.from_model(model)
await pm.change_multicube_permission(mc_id, role_id, "read")   # те же методы есть у model
```

- `change_multicube_permission`, `change_list_permission` и `change_version_permission` принимают
  `write`, `read` или `none` в любом регистре. Работают около двух секунд и возвращают `True`.
- `True` приходит и тогда, когда ставишь то же значение, что уже стоит. По нему нельзя понять,
  изменилось ли что-то.
- Роль передаётся только числом. На имя роли SDK отвечает
  `WorkspaceException(400, "ID роли должен быть целым")`.
- Неверное значение права SDK отбивает сам: `400 "permission должен быть одним из: write, read, none"`.
  На несуществующую роль отвечает сервер: `404 "Роль с id=… не найдена в матрице доступов"`.
- С макросами есть баг. `change_macros_permission` (и его алиасы `change_macro_permission`,
  `change_mscros_permission`) принимает `"read"`, `"write"`, `"false"`, `True` и `False`, а строку
  `"none"` отвергает. Текст ошибки при этом такой: «должен быть одним из: true, false, write, read,
  none». Доступ снимаем через `False`.

### UAM

`UamManager.from_model(model)` даёт `get_enabled_list_ids()`, `get_uam_grid_mapping()`,
`get_uam(list_id)` и `change_uam(list_id, item_id, user, permission)`. У модели есть короткие версии:
`get_list_uam`, `get_uam`, `change_uam`.

`get_enabled_list_ids()` и `get_list_uam()` возвращают все справочники модели, включая системные,
а не только те, где включён UAM. Если вызвать `get_uam` на справочнике без UAM, получим
`400 "Dimension '…' user cube not found"`.

`get_uam_grid_mapping()` отдаёт словарь `{id справочника: id UAM-грида}`. `get_uam(list_id)` отдаёт
DataFrame, где строки соответствуют элементам справочника (включая `TOTAL`), колонки соответствуют
e-mail пользователей, а в ячейках `write`, `read` или `none`. Работает около двух секунд.

`change_uam(list_id, item_id, "user@mail", "read")` возвращает `True` примерно за 4,6 секунды.
Пользователя можно указать по e-mail. На несуществующего SDK отвечает
`404 "Пользователь '…' не найден в UAM-grid"`.

### Пользователи

`get_model_user_list()` и `get_ws_user_list()` возвращают DataFrame с колонками `MODEL_USERS_GRID`.
`get_model_user_info(id)` и `get_ws_user_info(id)` возвращают одну строку, а `id` здесь означает
`rowEntityLongId` пользователя (вида 111000000…). По документации, без прав на пользователей
воркспейса сервер ответит `Grid access denied`.

## Скрипты

| Метод | Что возвращает |
|---|---|
| `get_script_list()` | `list[int]` с id скриптов |
| `get_script_info(id)` | Одну строку из 16 колонок: rowEntityLongId, rowLabel (имя скрипта), Macros (исходный код), Folder, Scheduler, Lock Mode, Last Date, Last Time, Last Status, Memory Limit MB, Time Limit sec, Custom Offset, Comments, Production, Dev Complete, Id. Если скрипта нет, `None` |
| `run_script(id)` | Ждёт, пока скрипт закончится, и возвращает его вывод строкой: `console.log("hi!")` превращается в `"hi!"`. На простом скрипте около секунды. С несуществующим id падает с `KeyError: 'output'` |
| `get_script_history(...)` | DataFrame: History Date, History Time, History Status, History Elapsed Time, History User, History Type, History Request Id. Если история не пустая, спереди добавляются rowEntityLongId и rowLabel (`#299283 scr1`) |
| `get_script_chain_history(...)` | То же для цепочек скриптов |

Фильтры истории `date_from`, `date_to`, `status`, `run_type` и `show_archive` работают. Если
`date_from` не передать, SDK сам подставит «сутки назад».

Фильтр `identifiers=[...]` сервер не принимает: `400 "The property identifiers is not defined"`.
Чтобы получить историю одного скрипта, берём общую историю и фильтруем DataFrame по `rowLabel`.

Запуск через `run_script` попадает в историю с типом `INTERFACE`, а не `API`.

Если интересно, что именно уходит на сервер, можно вызвать `build_scripts_history_filter(...)`:

```json
{"type": "SCRIPTS_HISTORY_FILTER", "params": {"dateFrom": 1791724634, "runType": 2, "status": 3, "showArchive": true}}
```

## Дашборды

| Метод | Что возвращает |
|---|---|
| `get_dashboard_multicube_links(dashboard_ids=None, request_delay=0.5, ignore_errors=True)` | DataFrame: dashboardId, optionId, objectIdentifier, type (`cube`), name, multicubeId, gridId, savedViewId, positionX/Y/W/H. На четырёх дашбордах около 7 с |
| `get_dashboard_multicubes(dashboard_id)` | То же для одного дашборда, около 1,3 с |
| `export_dashboard_multicube_links_grid()` | Грид 137000000080, который, как выяснилось, не про связи дашбордов |

С `ignore_errors=True` (это значение по умолчанию) пустые дашборды молча пропускаются. С `False`
вызов падает на первом пустом дашборде с ошибкой про `definitionJsonString`.

Если в модели вообще нет дашбордов, вызов без `dashboard_ids` не вернёт пустой DataFrame, а упадёт
с `404 "Не удалось автоматически найти dashboard longId… Передайте dashboard_ids явно"`.

## Оптимизация

| Метод | Как себя ведёт |
|---|---|
| `get_optimizer_requests_grid()` и алиас `get_optimizer_requests()` | Без параметра падают с `400 viewDefinition not found`. С `entity_default_view_entity_long_id=137000000051` работают и отдают колонки optimizerRequestId, requestName, Optimizer Engine, Objectives, Variables, Constraints, Comments, Id |
| `get_optimizer_request_fields(id, entity_default_view_entity_long_id=137000000051)` | Без того же параметра тоже падает. С ним отдаёт optimizerRequestId, requestName, goals, variables, constraints |
| `export_lp_mapping(id, path)` | На пустом запросе: `404 "не удалось получить идентификатор для скачивания LP файла"`. Заполненного запроса в тестовой модели не было. По коду метод возвращает имя zip-файла с `optimizer.lp` и `optimizer.map`, а таймаут выгрузки зашит в 5 минут |
| `import_solution_mapping(zip)` | Не проверяли. Формат CSV с решением нигде не описан, это главный открытый вопрос по оптимизации |

LP, который выгружает Optimacros, содержит дефекты: голые переменные в Bounds, перевёрнутые границы,
константу в цели. Перед сторонним солвером его нужно прогнать через `lp_preprocess.py`.

## Бэкапы и управление моделью

| Метод | Как себя ведёт |
|---|---|
| `get_backup_list()` | DataFrame `BACKUPS_GRID`. В `rowLabel` короткий id (`#47329`), в `rowEntityLongId` длинный (41000047329) |
| `get_backup_info(long_id)` | Одна строка того же грида |
| `get_backup("#47329")` | `bytes` с zip-архивом, 130 КБ за 2 с. Нужен короткий id со знаком `#` |
| `backup()` | Возвращает `None` мгновенно, а нового бэкапа в списке нет. Похоже, метод ничего не делает |
| `recalculate()` | `True` за 0,7 с |
| `stop()` | `True` за 1,7 с. Повторный вызов на остановленной модели возвращает `False` (6,7 с) |
| Экспорт из остановленной модели | `AttributeError: 'NoneType' object has no attribute 'items'` |
| `start()` | `True`, но ждать приходится 22,6 с. На работающей модели тоже возвращает `True` |
| `restore()` | Не проверяли. Параметров у метода нет, и непонятно, из какого бэкапа он восстанавливает |

## Обходные пути

| Задача | Как решаем |
|---|---|
| DataFrame из мультикуба или вью | `export_df(sep=";")` или `export_om_file` и `pd.read_csv(sep=";")` |
| Наполнить справочник | `add_elements(names=[...])` |
| Выгрузить справочник | `export_file(format=CSV)` или `export_df()`, формат ОМ не работает |
| Все формулы модели | `CUBES_GRID`, колонка `Formula` |
| Исходный код скриптов | `get_script_info(id)["Macros"]` |
| История одного скрипта | Общая история и фильтр по `rowLabel` |
| Id роли по имени | `PermissionGridExporter.export(137000000019).role_columns` |
| Снять доступ к макросу | `change_macros_permission(id, role_id, False)` |
| Связи дашбордов с мультикубами | `get_dashboard_multicube_links()` |
| Выгрузка с фильтром | `export_om_file` с `boolean_cube_filter_long_id` или `filter_formula` |
| Свойства справочника, подмножества времени | Через SDK не достать: `LIST_PROPERTIES_GRID` и `TIME_*_SUBSETS_GRID` не работают |
| Создать мультикуб, куб, измерение, версию | Через SDK никак |
| Action API, для которого нет метода | Методы коннектора, на свой страх и риск |

## Что осталось непроверенным

- `export_lp_mapping` на заполненном оптимизационном запросе и `import_solution_mapping` вместе
  с форматом решения.
- Что `change_cell_value` ждёт в `dimension_id`.
- `restore()`.
- Какой `dimension_id` нужен фильтрам `export_file`.
- Windows и большие модели.

Если кто-то закроет один из этих вопросов, допишите сюда.

## Как перепроверить

Стенд [sdk_probe.py](https://github.com/dguzhavin/om-tools/blob/main/py-sdk-probe/sdk_probe.py) лежит
в репозитории вместе с
[.env.example](https://github.com/dguzhavin/om-tools/blob/main/py-sdk-probe/.env.example) и готовыми
настройками для тестовой модели в
[env_examples/](https://github.com/dguzhavin/om-tools/tree/main/py-sdk-probe/env_examples). Какие
объекты нужно создать в тестовой модели, написано в
[README](https://github.com/dguzhavin/om-tools/blob/main/py-sdk-probe/README.md), бэкап модели
кладём в [model/](https://github.com/dguzhavin/om-tools/tree/main/py-sdk-probe/model).

```bash
pip install -r requirements.txt
cp .env.example .env    # вписать токен и тестовые объекты
python sdk_probe.py     # первый прогон: OM_SKIP_WRITE=1, OM_TEST_ADMIN=0
```

Результат появится в папке из `OM_WORK_DIR`: `report.md` с таблицей по всем вызовам, `report.json`
с тем же для машинной обработки, `results/*.csv` с полученными DataFrame и `export/<группа>/<вызов>/`
с файлами каждого экспорта.

| Переменная | По умолчанию | Что включает |
|---|---|---|
| `OM_SKIP_WRITE` | 1 | 0 включает запись: временные справочники (в конце удаляются), импорт в мультикуб-приёмник с восстановлением данных, формулы, права и UAM с откатом, `run_script` |
| `OM_TEST_ADMIN` | 0 | 1 включает `recalculate`, `backup`, `stop` и `start` |
| `OM_TEST_RESTORE` | 0 | 1 включает `restore()`, который откатывает модель |
| `OM_GROUPS` | все группы | Позволяет запустить только часть групп |

Гоняем стенд только на тестовой модели. Когда выйдет новая версия SDK или Workspace, стоит прогнать
его заново и сравнить отчёты с этой статьёй.
