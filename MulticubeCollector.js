/*
	Добавляет название всех мультикубов в указанный справочник LIST_NAME. 
	При добавлении заменяет символ "#" на "##" и исключает дубли. 
	Скрипт устроен довольно просто, поэтому мониторить неактуальные МК нужно самостоятельно. 
*/

const ENV = {
	LIST_NAME: 'MCs', // Название справочника для добавление списка мультикубов
};

class MulticubeCollector {
	constructor(ENV) {
		this.listName = ENV.LIST_NAME;

		this.multicubesList = [];
		this.cleanMulticubeList = [];
		this.listElements = [];
	}

	// Получает актуальный список мультикубов
	getMulticubes() {
		const pivot = om.multicubes
			.multicubesTab()
			.pivot()
			.withoutValues()
			.create();
		const generator = pivot.range().generator();

		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
				const rowName = rowLabelsGroup.first().name();
				this.multicubesList.push(rowName);
			}
		}
	}

	// Получает список элементов справочника ENV.LIST_NAME
	getListData() {
		const pivot = om.lists
			.listsTab()
			.open(this.listName)
			.pivot()
			.withoutValues()
			.create();
		const generator = pivot.range().generator();

		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
				const rowName = rowLabelsGroup.first().name();
				this.listElements.push(rowName);
			}
		}
	}
	// Добавляет мультикубы в справочник
	addToList() {
		try {
		const creator = om.lists
			.listsTab()
			.open(this.listName)
			.elementsCreator()
			.named()
			.setElementNames(this.cleanMulticubeList);
			creator.create();
		} catch(e) {
			console.log(`Новые мультикубы не найдены.`)
		}
	}

	// Очищает список мультикубов от дублей и заменяем спец. символ "#" на "##"
	clearMulticubeList() {
		this.cleanMulticubeList = this.multicubesList
			.map((str) => str.replaceAll('#', '##'))
			.filter((str) => !this.listElements.includes(str));
	}

	// Точка входа
	run() {
		this.getMulticubes();
		this.getListData();
		this.clearMulticubeList();
		this.addToList();
	}
}

new MulticubeCollector(ENV).run();
