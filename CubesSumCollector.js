/**
 * Скрипт позволяет выгрузить CVS файл, в котором отражается сумма значений по всем ячейкам числовых кубов модели.
 */

const ENV = {
	FILE_NAME: 'file', // Имя выгружаемого файла без расширения
};

class MulticubeDataCollector {
	constructor(ENV) {
		this.multicubesTab = om.multicubes.multicubesTab();
		this.writer = om.filesystems.filesDataManager().csvWriter();
		this.localFileSystem = om.filesystems.local();

		this.fileName = ENV.FILE_NAME;
		this.fileExtention = 'csv'; // Расширение файла -- не менять!
		this.csvHeaders = ['Multicube', 'Cube', 'Sum'];

		this.data = [];

		this.writeFirstRow();
	}

	// Итерируется по мультикубам модели
	getMulticubesInfo() {
		const pivot = this.multicubesTab.pivot();
		const generator = pivot
			.withoutValues()
			.create()
			.range(0, -1, 0, 0)
			.generator();

		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
				const rowName = rowLabelsGroup.first().name();

				this.getCubes(rowName);
				// првоерить количество элементов в data - если больше 1000 (задавать в констр) - записывать чанками
			}
		}
		this.writer.writeRows(this.data);
	}

	// Итерируется по кубам полученного мультикуба и получает по ним сумму
	getCubes(mcName) {
		const pivot = this.multicubesTab.open(mcName);
		const generator = pivot
			.cubesTab()
			.pivot()
			.create()
			.range(0, -1, 0, 0)
			.generator();

		for (const chunk of generator) {
			const rowLabels = chunk.rows();
			for (const rowLabelsGroup of rowLabels.all()) {
				const rowName = rowLabelsGroup.first().name();
				if (rowName === mcName) continue;

				const sum = this.getCubeSum(pivot, rowName);

				if (sum != null) {
					// this.writer.writeRow([mcName, rowName, sum]);
					this.data.push([mcName, rowName, sum]);
				}
			}
		}
	}

	// Получает сумму по числовым кубам, по остальным возвращает null
	getCubeSum(multicubePivot, cubeName) {
		const cubeFormat = multicubePivot
			.getCubeInfo(cubeName)
			.getFormatInfo()
			.getFormatTypeEntity()
			.name();
		if (cubeFormat !== 'Number') return;

		let sum = 0;
		const gen = multicubePivot.cubeCellSelector(cubeName).load().generator();

		for (const chunk of gen) sum += chunk.getValue();
		return sum;
	}

	// Записывает первую строку с заголовками в файл
	writeFirstRow() {
		this.writer.writeRow(this.csvHeaders);
	}

	// Сохраняет CSV файл и скачивает его
	downloadCSV() {
		this.writer.save(this.fileName);

		const filePath = this.localFileSystem
			.getPathObj(`${this.fileName}.${this.fileExtention}`)
			.getPath();
		const hash = this.localFileSystem.makeGlobalFile(
			this.fileName,
			this.fileExtention,
			filePath
		);

		om.common.resultInfo().addFileHash(hash);
	}

	// Точка входа
	run() {
		this.getMulticubesInfo();
		this.downloadCSV();
	}
}

new MulticubeDataCollector(ENV).run();
