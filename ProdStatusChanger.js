/*
    Скрипт предназначен для того, чтобы помочь вам автомазировать процесс проставления признака Production в ходе тестирования процесса миграции ALCM.
    Необходимо передать список справочников в поле LISTS. Допускается только использование названий.
    
    Возможные дополнительные настройки:
      - присвоить признак Production ВСЕМ сабсетам справочников LISTS
      - присвоить признак Production ВСЕМ сабсетам версий
      - присвоить признак Production ВСЕМ сабсетам времени
        Список используемых шкал времени перечислен в переменной TIME_PERIOD_GRIDS в методе constructor класса ProdStatusChanger. По умолчанию список содержит все шкалы времени, неиспользуемые в модели шкалы игнорируются.
*/

const ENV = {
	LISTS: ['List #14', 'List #15', 'List #26'], // справочники, которым необходимо изменить статус
	SET_PRODUCTION_STATUS: true, // true - проставить Production статус справочникам, false - снять Production статус у справочников
	SET_ALL_SUBSET_STATUS: true, // проставить Production статус всем сабсетам справочников
	SET_TIMESCALE_SUBSET_STATUS: true, // проставить Production статус всем сабсетам времени
	SET_VERSION_SUBSET_STATUS: true, // проставить Production статус всем сабсетам версий
};

class ProdStatusChanger {
	constructor(ENV) {
		this.lists = ENV.LISTS;
		this.listsStatus = ENV.SET_PRODUCTION_STATUS;
		this.listSubsetStatus = ENV.SET_ALL_SUBSET_STATUS;
		this.timeSubsetStatus = ENV.SET_TIMESCALE_SUBSET_STATUS;
		this.versionSubsetStatus = ENV.SET_VERSION_SUBSET_STATUS;

		this.listAttr = 'Production';
		this.TIME_PERIOD_GRIDS = [
			'Days',
			'Weeks',
			'Periods',
			'Months',
			'Quarters',
			'Half Years',
			'Years',
		]; // Измерения времени (можно не трогать, несуществующие будут пропущены скриптом)

		this.listsTab = om.lists.listsTab();
		this.cb = om.common.createCellBuffer();
	}

	// Проставляет указанный статус справочникам
	setListStatus() {
		const pivot = this.listsTab
			.pivot()
			.columnsFilter(this.listAttr)
			.rowsFilter(this.lists);
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

	// Проставляет статус SET_PRODUCTION_STATUS сабсетам в полученном гриде
	setSubsetStatus(pivot, status) {
		const generator = pivot.create().range().generator();
		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
				for (const cell of rowLabelsGroup.cells().all()) {
					this.cb.set(cell, status);
				}
			}
		}
	}

	// Проставляет статус SET_PRODUCTION_STATUS справочникам LISTS, если активен флаг SET_ALL_SUBSET_STATUS
	setListSubsetStatus() {
		if (!this.listSubsetStatus) return;
		for (const list of this.lists) {
			const pivot = this.listsTab
				.open(list)
				.listSubsetTab()
				.pivot()
				.columnsFilter(this.listAttr);
			this.setSubsetStatus(pivot, this.listsStatus);
		}
	}

	// Проставляет статус SET_PRODUCTION_STATUS сабсетам версий, если активен флаг SET_VERSION_SUBSET_STATUS
	setVersionSubsetStatus() {
		if (!this.versionSubsetStatus) return;
		const pivot = om.versions
			.versionSubsetsTab()
			.pivot()
			.columnsFilter(this.listAttr);
		this.setSubsetStatus(pivot, this.listsStatus);
	}

	// Проставляет статус SET_PRODUCTION_STATUS сабсетам времени, если активен флаг SET_TIMESCALE_SUBSET_STATUS
	setTimeSubsetStatus() {
		if (!this.timeSubsetStatus) return;
		for (const item of this.TIME_PERIOD_GRIDS) {
			try {
				const pivot = om.times
					.timePeriodTab(item)
					.subsetsTab()
					.pivot()
					.columnsFilter(this.listAttr);
				this.setSubsetStatus(pivot, this.listsStatus);
			} catch {
				continue;
			}
		}
	}

	// Применяет все внесенные изменения (галки)
	apply() {
		this.cb.apply();
	}

	// Точка входа
	run() {
		this.setListStatus();
		this.setListSubsetStatus();
		this.setTimeSubsetStatus();
		this.setVersionSubsetStatus();
		this.apply();
	}
}

new ProdStatusChanger(ENV).run();
