# om-tools
`Тут буду складывать разные полезные скрипты для упрощения работы с оптимакросом`

## Оглавление:

1. [SubsetInfoCollector](https://github.com/dguzhavin/om-tools/blob/main/SubsetInfoCollector)
    Скрипт позволяет **собрать информацию** о всех имеющихся **сабсетах** модели в виде CSV файла.

2. [ProductionStatusChanger](https://github.com/dguzhavin/om-tools/blob/main/ProductionStatusChanger.js)
  Скрипт предназначен для того, чтобы помочь вам **автомазировать процесс проставления признака Production** в ходе тестирования процесса миграции ALCM.

3. [ProductionSubsetDataManager](https://github.com/dguzhavin/om-tools/blob/main/ProductionSubsetDataManager.js)
    Скрипт позволяет бэкапировать и восстанавливать **данные продуктивных сабсетов** пользовательских справочников, версий и времени.

4. [ProductionSubsetDataChanger](https://github.com/dguzhavin/om-tools/blob/main/ProductionSubsetDataChanger.js)
   Скрипт изменяет текущее содержимое на противополеженное  для первых 10 элементов всех справочникв для всех продуктивных сабсетов. Если элемент был включен в сабсет - скрипт его исключает, если не был включен - включает.




---

Описание подключения гит репозитория к воркспейсу ОМ вынес в [отдельный репозиторий](https://github.com/dguzhavin/ext-scripts/tree/main).

