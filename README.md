# om-tools
`Тут буду складывать разные полезные скрипты для упрощения работы с оптимакросом`

## Оглавление:

1. [SubsetInfoCollector](https://github.com/dguzhavin/om-tools/blob/main/SubsetInfoCollector)
    Скрипт позволяет **собрать информацию** о всех имеющихся **сабсетах** модели в виде CSV файла.
    Для удобного сравнения файлов можно использовать шаблон - [alcm-reports-analysis.xlsx](https://github.com/dguzhavin/om-tools/blob/main/alcm-reports-analysis.xlsx)

3. [ProductionStatusChanger](https://github.com/dguzhavin/om-tools/blob/main/ProductionStatusChanger.js)
  Скрипт предназначен для того, чтобы помочь вам **автомазировать процесс проставления признака Production** в ходе тестирования процесса миграции ALCM.

4. [ProductionSubsetDataManager](https://github.com/dguzhavin/om-tools/blob/main/ProductionSubsetDataManager.js)
    Скрипт позволяет бэкапировать и восстанавливать **данные продуктивных сабсетов** пользовательских справочников, версий и времени.

5. [ProductionSubsetDataChanger](https://github.com/dguzhavin/om-tools/blob/main/ProductionSubsetDataChanger.js)
   Скрипт изменяет текущее содержимое (галку) всех продуктивных сабсетов на противоположное для первых 10 элементов всех справочников.

6. [MulticubeCollector](https://github.com/dguzhavin/om-tools/blob/main/MulticubeCollector.js)
   Скрипт добавляет название всех мультикубов модели в указанный пользовательский справочник.

7. [CubesSumCollector](https://github.com/dguzhavin/om-tools/blob/main/CubesSumCollector.js)
   Скрипт собирает суммы по всем числовым кубам модели и выгружает в CSV файл.


---

Описание подключения гит репозитория к воркспейсу ОМ вынес в [отдельный репозиторий](https://github.com/dguzhavin/ext-scripts/tree/main).

