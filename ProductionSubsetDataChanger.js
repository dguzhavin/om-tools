/*
    Скрипт изменяет текущее содержимое на противополеженное  для первых 10 элементов всех справочникв для всех продуктивных сабсетов. Если элемент был включен в сабсет - скрипт его исключает, если не был включен - включает.
*/

class ProductionSubsetDataChanger {
    constructor() {
        this.SUBSET_GRID_COLUMN = "Production";
        this.versionsListName = "Versions";
        this.TIMESCALE_GRIDS = ['Days', 'Weeks', 'Periods', 'Months', 'Quarters', 'Half Years', 'Years']; // Измерения времени (можно не трогать, несуществующие будут пропущены скриптом)
        this.listsTab = om.lists.listsTab();
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
                this.getListSubsets(rowName);
			}
		}
    }

    // Собирает данные сабсетов пользовательских справочников
    getListSubsets(listName) {
        print(`[~] Получаю данные сабсетов справочника "${listName}"...\n`);
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
                            this.DATA[listName] = []
                        }
                        this.DATA[listName].push(rowName)
                        status = true;
                    }
                }
            }
        }
        return status;
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
                    for (const cell of rowLabelsGroup.cells().all()) {
                        if (cell.getValue() === "true") {
                            if (!this.DATA[listName]) {
                                this.DATA[listName] = []
                            }
                            this.DATA[listName].push(rowName)
                        }
                    }
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
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (cell.getValue() === "true") {
                        if (!this.DATA[listName]) {
                            this.DATA[listName] = []
                        }
                        this.DATA[listName].push(rowName)
                    }
                }
            }
        }
    }

    // Выводит сообщение в лог
    print(msg) {
        console.log(msg);
    }

    // Изменяет значение галок на обратное
    changeListSubsetItems({listName, userListType = null, timeListType = null, versionListType = null}) {
        let pivot;
        
        const listSubsets = this.DATA[listName];

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
        const generator = pivot.create().range(0, 10, 0, -1).generator();
		
		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
                for (const cell of rowLabelsGroup.cells().all()) {
                    if (!cell.isEditable()) continue;
                    
                    this.cb.set(cell, cell.getValue() === "false");
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

    // Изменяет данные сабсетов версий
    changeVersionSubsets() {   
        this.print(`[~] Восстановление сабсетов версий.\n`);
        if (Object.keys(this.DATA).includes(this.versionsListName)) {
            this.changeListSubsetItems({listName: this.versionsListName, versionListType:true});
        } 
    }
    
    // Изменяет данные сабсетов справочников времени
    changeTimescaleSubsets() {
        this.print(`[~] Восстановление сабсетов справочников времени.\n`)
        for (const list of this.TIMESCALE_GRIDS) {
            if (Object.keys(this.DATA).includes(list)) {
                this.changeListSubsetItems({listName: list, timeListType:true})
            }
        }
    }

    // Изменяет данные сабсетов пользовательских справочников
    changeListSubsets() {
        this.print(`[~] Восстановление сабсетов пользовательских справочников.\n`)
        for (const list of this.actualLists) {
            this.changeListSubsetItems({listName: list, userListType:true})
        }
    }

    // Точка входа
    run() {
        this.DATA = {};
        this.getUserListsSubsetData();
        this.getTimescaleSubsetData();
        this.getVersionSubsetData();

        this.actualLists = Object.keys(this.DATA);
    
        this.changeVersionSubsets();
        this.changeTimescaleSubsets()
        this.changeListSubsets();
        this.applyChanges();
    }
}

(new ProductionSubsetDataChanger()).run();
