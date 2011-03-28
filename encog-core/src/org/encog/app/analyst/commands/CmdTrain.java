package org.encog.app.analyst.commands;

import java.io.File;

import org.encog.app.analyst.AnalystError;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.script.prop.ScriptProperties;
import org.encog.ml.MLMethod;
import org.encog.ml.MLTrain;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.neural.data.NeuralDataSet;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.simple.EncogUtility;

public class CmdTrain extends Cmd {

	public final static String COMMAND_NAME = "TRAIN";

	public CmdTrain(EncogAnalyst analyst) {
		super(analyst);
	}

	private MLMethod obtainMethod() {
		String resourceID = getProp().getPropertyString(
				ScriptProperties.ML_CONFIG_machineLearningFile);
		String resourceFile = getProp().getFilename(resourceID);
		
		MLMethod method = (MLMethod)EncogDirectoryPersistence.loadObject(new File(resourceFile));

		if (!(method instanceof MLMethod)) {
			throw new AnalystError(
					"The object to be trained must be an instance of MLMethod. "
							+ method.getClass().getSimpleName());
		}

		return (MLMethod) method;
	}

	private NeuralDataSet obtainTrainingSet() {
		String trainingID = getProp().getPropertyString(
				ScriptProperties.ML_CONFIG_trainingFile);

		String trainingFile = getProp().getFilename(trainingID);

		NeuralDataSet trainingSet = EncogUtility.loadEGB2Memory(trainingFile);

		return trainingSet;
	}

	private MLTrain createTrainer(MLMethod method, NeuralDataSet trainingSet) {

		MLTrainFactory factory = new MLTrainFactory();
		
		String type = this.getProp().getPropertyString(ScriptProperties.ML_TRAIN_type);
		String args = this.getProp().getPropertyString(ScriptProperties.ML_TRAIN_arguments);
		
		MLTrain train = factory.create(method, trainingSet, type, args);

		return train;
	}

	private void performTraining(MLTrain train, MLMethod method,
			NeuralDataSet trainingSet) {

		double targetError = this.getProp().getPropertyDouble(ScriptProperties.ML_TRAIN_targetError);
		this.getAnalyst().reportTrainingBegin();
				
		do {
			train.iteration();
			this.getAnalyst().reportTraining(train);
		} while (train.getError() > targetError
				&& !this.getAnalyst().shouldStopCommand()
				&& !train.isTrainingDone() );
		train.finishTraining();

		this.getAnalyst().reportTrainingEnd();
	}

	public boolean executeCommand() {

		MLMethod method = obtainMethod();
		NeuralDataSet trainingSet = obtainTrainingSet();
		MLTrain trainer = createTrainer(method, trainingSet);

		performTraining(trainer, method, trainingSet);
		
		String resourceID = getProp().getPropertyString(
				ScriptProperties.ML_CONFIG_machineLearningFile);
		String resourceFile = getProp().getFilename(resourceID);
		EncogDirectoryPersistence.saveObject(new File(resourceFile), method);


		return this.getAnalyst().shouldStopCommand();
	}

	@Override
	public String getName() {
		return COMMAND_NAME;
	}

}
