const ENV = {
    LISTS: ["List #14", "List #15", "List #26"], // справочники, которым необходимо изменить статус
    //LISTS: [101000000001, 101000000005], // справочники, которым необходимо изменить статус
    SET_PRODUCTION_STATUS: true, // true - проставить Production статус справочникам, false - снять Production статус у справочников
    SET_ALL_SUBSET_STATUS: true, // проставить Production статус всем сабсетам справочников
    SET_TIMESCALE_SUBSET_STATUS: true, // проставить Production статус всем сабсетам времени
    SET_VERSION_SUBSET_STATUS: true, // проставить Production статус всем сабсетам версий

}

class ProdStatusChanger {
    constructor(ENV) {
        this.lists = ENV.LISTS;
        this.listsStatus = ENV.SET_PRODUCTION_STATUS;
        this.listSubsetStatus = ENV.SET_ALL_SUBSET_STATUS;
        this.timeSubsetStatus = ENV.SET_TIMESCALE_SUBSET_STATUS;
        this.versionSubsetStatus = ENV.SET_VERSION_SUBSET_STATUS;

        this.listAttr = "Production";

        this.listsTab = om.lists.listsTab();
        this.cb = om.common.createCellBuffer()

        //tmp
        this.listData = {}
    }

    // Проставляет указанный статус справочникам
    setListStatus() {
        const pivot = this.listsTab.pivot().columnsFilter(this.listAttr).rowsFilter(this.lists)
        const generator = pivot.create().range().generator();

        for (const chunk of generator) {
            const rowLabels = chunk.rows();
            for (const rowLabelsGroup of rowLabels.all()) {
                for (const cell of rowLabelsGroup.cells().all()) {
                    this.cb.set(cell, this.listsStatus);
                }
            }
        }
    }

    setSubsetStatus(pivot) {}

    setListSubsetStatus() {}

    setTimeSubsetStatus() {}

    setVersionSubsetStatus() {}

    apply() {
        this.cb.apply();
    }

    run() {
        this.setListStatus();

        this.apply();
        
        //tmp
    }
}

(new ProdStatusChanger(ENV)).run()