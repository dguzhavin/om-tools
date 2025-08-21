class ModelChecker {

    constructor() {
        this.listsTab = om.lists.listsTab();
        this.localFileSystem = om.filesystems.local();
        this.writer = om.filesystems.filesDataManager().csvWriter();

        this.LIST_GRID_COLUMNS_LIST = ["Production"]; // Колонки грида справочников (чтобы добавить в файл, необходимо доработать скрипт)
        this.SUBSET_GRID_COLUMNS_LIST = ["Element Count", "Production", "Id"]; // Колонки грида сабсетов (при добавлении будут добавлены в файл)
        this.TIME_PERIOD_GRIDS = ['Days', 'Weeks', 'Periods', 'Months', 'Quarters', 'Half Years', 'Years']; // Измерения времени (можно не трогать, несуществующие будут пропущены скриптом)
        this.csvRows = [
            ["List", "Prod Status", "Subset", "Column", "Value"] // Первая строка файла с заголовками
        ];

        this.fileName = "report"; // Навзвание экспортируемого файла
        this.fileExtention = "csv"; // Расширение файла -- не менять!
    }

    // Получает список пользовательских справочников
    getLists() {
        this.listData = {};
        const pivot = this.listsTab.pivot().columnsFilter(this.LIST_GRID_COLUMNS_LIST);

        const generator = pivot.create().range().generator();

        for (const chunk of generator) {
            const rowLabels = chunk.rows();
            for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                this.listData[rowName] = {};
                for (const cell of rowLabelsGroup.cells().all()) {
                    const colName = cell.columns().first().name();
                    this.listData[rowName][colName] = cell.getValue();
                }
            }
        }
    }

    // Получает информацию о сабсетах полученного грида 
    getSubsetData(pivot, listName, systemrid = false) {
        const generator = pivot.create().range().generator();

        for (const chunk of generator) {
            const rowLabels = chunk.rows();
            for (const rowLabelsGroup of rowLabels.all()) {
                const rowName = rowLabelsGroup.first().name();
                for (const cell of rowLabelsGroup.cells().all()) {
                    const colName = cell.columns().first().name();
                    this.csvRows.push([listName, systemrid ? "-" : this.listData[listName]["Production"], rowName, colName, cell.getValue()]);
                }
            }
        }
    }

    // Собирает информацию о сабсетах всех пользовательских справочников
    getListSubsetData() {
        Object.keys(this.listData).forEach(item => {
            const pivot = this.listsTab.open(item).listSubsetTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMNS_LIST);
            this.getSubsetData(pivot, item)
        })
    }

    // Получает информацию о сабсетах всех указанных периодов времени
    getTimePeriodSubsets() {
        for (const item of this.TIME_PERIOD_GRIDS) {
            try {
                const pivot = om.times.timePeriodTab(item).subsetsTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMNS_LIST);
                this.getSubsetData(pivot, item, true)
            } catch {
                continue;
            }
        }
    }

    // Получает информацию о сабсетах версий
    getVersionSubsets() {
        const pivot = om.versions.versionSubsetsTab().pivot().columnsFilter(this.SUBSET_GRID_COLUMNS_LIST);
        this.getSubsetData(pivot, "Versions", true)
    }

    // Записывает всю полученную информацию в CSV файл и скачивает его
    writeToCSV() {

        this.csvRows.forEach(row => {
            this.writer.writeRow(row)
        })

        this.writer.save(this.fileName);

        const filePath = this.localFileSystem.getPathObj(`${this.fileName}.${this.fileExtention}`).getPath();
        const hash = this.localFileSystem.makeGlobalFile(this.fileName, this.fileExtention, filePath);

        om.common.resultInfo().addFileHash(hash)
    }

    // Точка входа
    run() {
        this.getLists();
        this.getListSubsetData();
        this.getTimePeriodSubsets();
        this.getVersionSubsets()
        this.writeToCSV();
    }
}

(new ModelChecker()).run()