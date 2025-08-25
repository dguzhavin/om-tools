/*
    Скрипт позволяет бэкапировать и восстанавливать данные продуктивных сабсетов пользовательских справочников, версий и времени.

    Режим BACKUP предназначен для скачивания "резервной копии" данных продуктивных справочников. Результат работы скрипта - скачанный .txt файл с JSON-структурой.

    Режим RESTORE предназначен для восстановления данных продуктивных сабсетов. Для этого необходимо передать JSON из скачанного в режиме BACKUP файла в параметр DUMP_DATA.

*/

const ENV = {
    SCRIPT_MODE: "BACKUP", // BACKUP || RESTORE
    DUMP_DATA: null // Обязательный параметр для режима RESTORE
};

class ProductionSubsetDataManager {
    constructor(ENV) {
        this.VALID_SCRIPT_MODE = ["BACKUP", "RESTORE"];
        this.SCRIPT_MODE = ENV.SCRIPT_MODE;
        this.isBackupMode = this.SCRIPT_MODE === "BACKUP";
        this.isRestoreMode = this.SCRIPT_MODE === "RESTORE";
        this.SUBSET_GRID_COLUMN = "Production";
        this.versionsListName = "Versions";
        this.TIMESCALE_GRIDS = ['Days', 'Weeks', 'Periods', 'Months', 'Quarters', 'Half Years', 'Years']; // Измерения времени (можно не трогать, несуществующие будут пропущены скриптом)
        this.fileName = `model_${om.common.modelInfo().id()}_subsets_dump`;
        this.fileExtention = "txt"; // Расширение файла -- не менять!
        this.DATA = ENV.DUMP_DATA;
        this.actualLists = this.DATA ? Object.keys(this.DATA) : null;
        this.listsTab = om.lists.listsTab();
        this.writer = om.filesystems.filesDataManager().csvWriter();  
        this.localFileSystem = om.filesystems.local();  
        this.cb = om.common.createCellBuffer().canLoadCellsValues(false);    
    }

    // Получает данные пользовательских справочников
    getUserListsSubsetData() {
        print(`[~] Получаю список пользовательских справочников.\n`);

		const pivot = this.listsTab.pivot().withoutValues();
		const generator = pivot.create().range().generator();
		
		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                const listHasProdSubsets = this.getListSubsets(rowName);

                if (listHasProdSubsets) this.getListSubsetItems({listName: rowName, userListType:true});
			}
		}
    }

    // Собирает данные сабсетов пользовательских справочников
    getListSubsets(listName) {
        print(`[~] Получаю данные сабсетов пользовательских справочников...\n`);
        let status;
        const pivot = this.listsTab.open(listName).listSubsetTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMN);
        const generator = pivot.create().range().generator();

        for (const chunk of generator) {
            const rowLabels = chunk.rows();
            for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (cell.getValue() === "true") {
                        if (!this.DATA[listName]) {
                            this.DATA[listName] = {}
                        }
                        this.DATA[listName][rowName] = []
                        status = true;
                    }
                }
            }
        }
        return status;
    }

    // Получает содержимое сабсетов по условиям
    getListSubsetItems({listName, userListType = null, timeListType = null, versionListType = null}) {
        let pivot;
        const listSubsets = Object.keys(this.DATA[listName]);
        if (timeListType) {
            pivot = om.times.timePeriodTab(listName).pivot().columnsFilter(listSubsets);
        } else if (userListType) {
            pivot = this.listsTab.open(listName).pivot().columnsFilter(listSubsets);
        } else if (versionListType) {
            pivot = om.versions.versionsTab().pivot().columnsFilter(listSubsets);
        } else {
            this.print(`Ошибка, метод getListSubsetItems получил некорректный тип справочка.\n`);
            return;
        }
        const generator = pivot.create().range().generator();
		
		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (!cell.isEditable()) continue;
                    const colName = cell.columns().first().name();
                    if (cell.getValue() === "true") {
                        this.DATA[listName][colName].push(rowName);
                    }
                }
			}
		}
    }

    // Собирает данные сабсетов справочников времени
    getTimescaleSubsetData() {
        print(`[~] Получаю список справочников времени.\n`);
        print(`[~] Получаю данные сабсетов справочников времени...\n`);

        for (const listName of this.TIMESCALE_GRIDS) {
            let pivot;
            try {
                pivot = om.times.timePeriodTab(listName).subsetsTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMN);
            } catch {
                continue;
            }

            const generator = pivot.create().range().generator();
    
            for (const chunk of generator) {
                const rowLabels = chunk.rows();
                for (const rowLabelsGroup of rowLabels.all()) {
                    const rowName = rowLabelsGroup.first().name();
                    let status;
                    for (const cell of rowLabelsGroup.cells().all()) {
                        if (cell.getValue() === "true") {
                            if (!this.DATA[listName]) {
                                this.DATA[listName] = {}
                            }
                            this.DATA[listName][rowName] = []
                            status = true;
                        }
                    }
                    if (status) this.getListSubsetItems({listName: listName, timeListType:true});
                }
            }
        }
    }

    // Собирает данные сабсетов версий
    getVersionSubsetData() {
        print(`[~] Получаю данные сабсетов версий...\n`);

        const listName = this.versionsListName;
        const pivot = om.versions.versionSubsetsTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMN);
        const generator = pivot.create().range().generator();

        for (const chunk of generator) {
            const rowLabels = chunk.rows();
            for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                let status;
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (cell.getValue() === "true") {
                        if (!this.DATA[listName]) {
                            this.DATA[listName] = {}
                        }
                        this.DATA[listName][rowName] = []
                        status = true;
                    }
                }
                if (status) this.getListSubsetItems({listName: listName, versionListType:true});
            }
        }
    }

    // Скачивает файл с дампом содержимого продуктивных сабсетов
    downloadSavedData() {
        this.localFileSystem.write(`${this.fileName}.${this.fileExtention}`, JSON.stringify(this.DATA));

        const filePath = this.localFileSystem.getPathObj(`${this.fileName}.${this.fileExtention}`).getPath();
        const hash = this.localFileSystem.makeGlobalFile(this.fileName, this.fileExtention, filePath);

        om.common.resultInfo().addFileHash(hash)
        print(`[~] Скачивание бэкапа сабсетов модели.\n`)
    }

    // Запускает режим BACKUP
    saveProductionSubsetInfo() {
        print(`[ i ] Скрипт запущен в режиме ${this.SCRIPT_MODE}.\n`)
        this.DATA = {};
        this.getUserListsSubsetData();
        this.getTimescaleSubsetData();
        this.getVersionSubsetData();
        this.downloadSavedData();
    }

    // Выводит сообщение в лог
    print(msg) {
        console.log(msg);
    }

    // Проверяет наличие данных для восстановление в режиме RESTORE
    restoreValidation() {
        if (!this.DATA) {
            this.print(`[ ! ] Ошибка восстановления! Данные не получены.\n`);
            throw Error("Ошибка восстановления!");
        }        
    }

    // Возвращает массив уникальных элементов справочника, включенных хотя бы в один продуктивный сабсет
    getUniqueItemsList(data) {
        return [...new Set(Object.values(data).flat())];
    }

    // Задает значения нужных галок в буфере ячеек сабсетов по условиям
    restoreListSubsetItems({listName, userListType = null, timeListType = null, versionListType = null}) {
        let pivot;
        
        const listData = this.DATA[listName];
        const listSubsets = Object.keys(listData);
        const uniqueRows = this.getUniqueItemsList(listData);

        if (timeListType) {
            pivot = om.times.timePeriodTab(listName).pivot().columnsFilter(listSubsets)//.rowsFilter(uniqueRows);
        } else if (userListType) {
            try {
                pivot = this.listsTab.open(listName).pivot().columnsFilter(listSubsets)//.rowsFilter(uniqueRows);
            } catch {
                return
            }
        } else if (versionListType) {
            pivot = om.versions.versionsTab().pivot().columnsFilter(listSubsets)//.rowsFilter(uniqueRows);
        } else {
            this.print(`[ ! ] Ошибка, метод restoreListSubsetItems получил некорректный тип справочка.\n`);
            return;
        }
        const generator = pivot.create().range().generator();
		
		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                console.log(`rowName - ${rowName}\n`)
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (!cell.isEditable()) continue;
                    const colName = cell.columns().first().name();
                    console.log(`value - ${cell.getValue()}, includes - ${this.DATA[listName][colName].includes(rowName)}\n`)
                    if (cell.getValue() === "false" && this.DATA[listName][colName].includes(rowName) ||
                        cell.getValue() === "true" && !this.DATA[listName][colName].includes(rowName)) {
                        this.cb.set(cell, cell.getValue() === "false");
                    }
                }
			}
		}
    }

    // Применяет внесенные изменения (галки сабсетов)
    applyChanges() {
        console.log(`[~] Начинаю применять изменения.\n`);
        this.cb.apply();
        console.log(`[ i ] Изменения применены.\n`);
    }

    // Восстанавливает данные сабсетов версий
    restoreVersionSubsets() {   
        this.print(`[~] Восстановление сабсетов версий.\n`)     
        this.restoreListSubsetItems({listName: this.versionsListName, versionListType:true});
    }
    
    // Восстанавливает данные сабсетов справочников времени
    restoreTimescaleSubsets() {
        this.print(`[~] Восстановление сабсетов справочников времени.\n`)
        for (const list of this.TIMESCALE_GRIDS) {
            if (this.actualLists.includes(list)) {
                this.restoreListSubsetItems({listName: list, timeListType:true})
            }
        }
    }

    // Восстанавливает данные сабсетов пользовательских справочников
    restoreListSubsets() {
        this.print(`[~] Восстановление сабсетов пользовательских справочников.\n`)
        for (const list of this.actualLists) {
            this.restoreListSubsetItems({listName: list, userListType:true})
        }
    }

    // Запускает режим восстановления
    restoreProductionSubsetInfo() {
        print(`[ i ] Скрипт запущен в режиме ${this.SCRIPT_MODE}.\n`)
        
        this.restoreValidation();
        this.restoreVersionSubsets();
        this.restoreTimescaleSubsets()
        this.restoreListSubsets();
        this.applyChanges();
    }

    // Точка входа
    run() {
        print(`[~] Запуск скрипта...\n`);

        if (this.isBackupMode) {
            this.saveProductionSubsetInfo();
        } else if (this.isRestoreMode) {
            this.restoreProductionSubsetInfo();
        } else {
            print(`[ ! ] Ошибка! Параметр SCRIPT_MODE должен соответствовать одному из значений: ${this.VALID_SCRIPT_MODE.join(', ')}.\n`);
        }
    }
}

(new ProductionSubsetDataManager(ENV)).run();
