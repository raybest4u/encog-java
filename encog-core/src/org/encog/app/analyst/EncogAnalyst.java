package org.encog.app.analyst;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URL;

import org.encog.app.analyst.analyze.PerformAnalysis;
import org.encog.app.analyst.evaluate.AnalystEvaluateCSV;
import org.encog.app.analyst.script.AnalystClassItem;
import org.encog.app.analyst.script.AnalystScript;
import org.encog.app.analyst.script.DataField;
import org.encog.app.analyst.script.EncogAnalystConfig;
import org.encog.app.analyst.script.segregate.AnalystSegregateTarget;
import org.encog.app.quant.evaluate.EvaluateCSV;
import org.encog.app.quant.normalize.ClassItem;
import org.encog.app.quant.normalize.NormalizationAction;
import org.encog.app.quant.normalize.NormalizationStats;
import org.encog.app.quant.normalize.NormalizeCSV;
import org.encog.app.quant.normalize.NormalizedField;
import org.encog.app.quant.segregate.SegregateCSV;
import org.encog.app.quant.segregate.SegregateTargetPercent;
import org.encog.app.quant.shuffle.ShuffleCSV;
import org.encog.bot.BotUtil;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.buffer.EncogEGBFile;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.persist.EncogMemoryCollection;
import org.encog.persist.EncogPersistedObject;
import org.encog.util.csv.CSVFormat;
import org.encog.util.file.FileUtil;
import org.encog.util.logging.Logging;
import org.encog.util.simple.EncogUtility;

public class EncogAnalyst {

	public final static String ACTION_ANALYZE = "ANALYZE";

	private AnalystScript script = new AnalystScript();

	public void analyze(File file, boolean headers, CSVFormat format) {
		script.getConfig().setFilename(EncogAnalystConfig.FILE_RAW,
				file.toString());
		script.getConfig().setCSVFormat(format);
		script.getConfig().setInputHeaders(headers);
		PerformAnalysis a = new PerformAnalysis(script, file.toString(),
				headers, CSVFormat.ENGLISH);
		a.process(this);

	}

	public void clear() {

	}

	public void load(String filename) {
		load(new File(filename));
	}

	public void save(String filename) {
		save(new File(filename));
	}

	public void save(File file) {
		OutputStream fos = null;

		try {
			fos = new FileOutputStream(file);
			save(fos);
		} catch (IOException ex) {
			throw new AnalystError(ex);
		} finally {
			if (fos != null)
				try {
					fos.close();
				} catch (IOException e) {
					throw new AnalystError(e);
				}
		}
	}

	public void load(File file) {
		InputStream fis = null;

		try {
			fis = new FileInputStream(file);
			load(fis);
		} catch (IOException ex) {
			throw new AnalystError(ex);
		} finally {
			if (fis != null)
				try {
					fis.close();
				} catch (IOException e) {
					throw new AnalystError(e);
				}
		}
	}

	public void save(OutputStream stream) {
		this.script.save(stream);

	}

	public void load(InputStream stream) {
		this.script.load(stream);
	}

	/**
	 * @return the script
	 */
	public AnalystScript getScript() {
		return script;
	}

	private void generateNormalizedFields(File file) {
		NormalizedField[] norm = new NormalizedField[this.script.getFields().length];
		DataField[] dataFields = this.getScript().getFields();

		for (int i = 0; i < this.script.getFields().length; i++) {
			DataField f = dataFields[i];
			NormalizationAction action;

			if (f.isInteger() || f.isReal() && !f.isClass()) {
				action = NormalizationAction.Normalize;
				norm[i] = new NormalizedField(f.getName(), action, 1, -1);
				norm[i].setActualHigh(f.getMax());
				norm[i].setActualLow(f.getMin());
			} else if( f.isClass() ) { 
				if( f.getClassMembers().size()>2)
					action = NormalizationAction.Equilateral;
				else
					action = NormalizationAction.OneOf;
				norm[i] = new NormalizedField(f.getName(), action, 1, -1);
				int index = 0;
				for(AnalystClassItem item : f.getClassMembers() )
				{
					norm[i].getClasses().add(new ClassItem(item.getName(),index++));
				}
			} else {
				action = NormalizationAction.PassThrough;
				norm[i] = new NormalizedField(action, f.getName());
			}
		}

		String train;
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_NORMALIZE,
				FileUtil.addFilenameBase(file, "_norm").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_RANDOM,
				FileUtil.addFilenameBase(file, "_random").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_OUTPUT,
				FileUtil.addFilenameBase(file, "_output").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_TRAIN,
				train = FileUtil.addFilenameBase(file, "_train").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_EVAL,
				FileUtil.addFilenameBase(file, "_eval").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_TRAINSET,
				FileUtil.forceExtension(train, "egb"));
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_EG,
				FileUtil.forceExtension(file.toString(), "eg"));

		this.script.getNormalize().setNormalizedFields(norm);
		this.script.getRandomize().setSourceFile(EncogAnalystConfig.FILE_RAW);
		this.script.getRandomize().setTargetFile(EncogAnalystConfig.FILE_RANDOM);
		this.script.getSegregate().setSourceFile(EncogAnalystConfig.FILE_RANDOM);		
		this.script.getNormalize().setSourceFile(EncogAnalystConfig.FILE_TRAIN);
		this.script.getNormalize().setTargetFile(EncogAnalystConfig.FILE_NORMALIZE);
		this.script.getGenerate().setSourceFile(EncogAnalystConfig.FILE_NORMALIZE);
		this.script.getGenerate().setTargetFile(EncogAnalystConfig.FILE_TRAINSET);
		this.script.getMachineLearning().setTrainingFile(EncogAnalystConfig.FILE_TRAINSET);
		this.script.getMachineLearning().setResourceFile(EncogAnalystConfig.FILE_EG);
		this.script.getMachineLearning().setOutputFile(EncogAnalystConfig.FILE_OUTPUT);
		this.script.getMachineLearning().setEvalFile(EncogAnalystConfig.FILE_EVAL);
		this.script.getMachineLearning().setMLType("feedforward");
		this.script.getMachineLearning().setMLArchitecture("TANH(?)->TANH(10)->TANH(?)");
		this.script.getMachineLearning().setResourceName("ml");
	}


	private void generateRandomize(File file) {

	}

	private void generateSegregate(File file) {
		AnalystSegregateTarget[] array = new AnalystSegregateTarget[2];
		array[0] = new AnalystSegregateTarget(EncogAnalystConfig.FILE_TRAIN, 75);
		array[1] = new AnalystSegregateTarget(EncogAnalystConfig.FILE_EVAL, 25);
		this.script.getSegregate().setSegregateTargets(array);
	}
	
	private NormalizedField getTargetField() {
		NormalizedField[] fields = this.script.getNormalize().getNormalizedFields();
		
		
		// first try to find a classify field
		for(int i=0;i<fields.length;i++) {
			DataField df = this.script.findDataField(fields[i].getName());
			if( fields[i].getAction().isClassify() && df.isClass() ) {
				return fields[i];
			}
		}
		
		// otherwise, just return the first regression field
		for(int i=0;i<fields.length;i++) {
			DataField df = this.script.findDataField(fields[i].getName());
			if( !df.isClass() && (df.isReal() || df.isInteger()) ) {
				return fields[i];
			}
		}
		
		// can't find anything useful!
		return null;
	}

	private void generateGenerate(File file) {
		NormalizedField targetField = getTargetField();

		int inputColumns = this.script.getNormalize().calculateInputColumns(targetField);
		int idealColumns = this.script.getNormalize().calculateOutputColumns(targetField);

		this.script.getGenerate().setInput(inputColumns);
		this.script.getGenerate().setIdeal(idealColumns);

	}

	public void wizard(File saveFile, File analyzeFile, boolean b,
			CSVFormat english) {
		analyze(analyzeFile, b, english);
		generateNormalizedFields(analyzeFile);
		generateRandomize(analyzeFile);
		generateSegregate(analyzeFile);
		generateGenerate(analyzeFile);
	}

	public void wizard(URL url, File saveFile, File analyzeFile, boolean b,
			CSVFormat format) {
		this.script.getInformation().setDataSource(url.toExternalForm());
		this.script.getInformation().setDataSourceFormat(format);
		this.script.getInformation().setDataSourceHeaders(b);
		BotUtil.downloadPage(url, analyzeFile);
		analyze(analyzeFile, b, format);
		generateNormalizedFields(analyzeFile);
		generateRandomize(analyzeFile);
		generateSegregate(analyzeFile);
		generateGenerate(analyzeFile);
	}

	public void normalize() {
		// mark generated
		this.script.markGenerated(this.script.getNormalize().getTargetFile());

		// get filenames
		String sourceFile = this.script.getConfig().getFilename(
				this.script.getNormalize().getSourceFile());
		String targetFile = this.script.getConfig().getFilename(
				this.script.getNormalize().getTargetFile());

		// prepare to normalize
		NormalizeCSV norm = new NormalizeCSV();
		NormalizedField[] normFields = this.script.getNormalize()
				.getNormalizedFields();
		NormalizationStats stats = new NormalizationStats(normFields);

		boolean headers = this.script.expectInputHeaders(this.script
				.getNormalize().getSourceFile());
		norm.analyze(sourceFile, headers, this.script.getConfig()
				.getCSVFormat(), stats);
		norm.setProduceOutputHeaders(this.script.getConfig().isOutputHeaders());
		norm.normalize(targetFile);
	}

	public void randomize() {
		// mark generated
		this.script.markGenerated(this.script.getRandomize().getTargetFile());

		// get filenames
		String sourceFile = this.script.getConfig().getFilename(
				this.script.getRandomize().getSourceFile());
		String targetFile = this.script.getConfig().getFilename(
				this.script.getRandomize().getTargetFile());

		// prepare to normalize
		ShuffleCSV norm = new ShuffleCSV();
		boolean headers = this.script.expectInputHeaders(this.script
				.getRandomize().getSourceFile());
		norm.analyze(sourceFile, headers, this.script.getConfig()
				.getCSVFormat());
		norm.process(targetFile);
	}

	public void segregate() {
		// get filenames		
		String inputFile = this.script.getConfig().getFilename(
				this.script.getSegregate().getSourceFile());

		// prepare to segregate
		boolean headers = this.script.expectInputHeaders(this.script
				.getSegregate().getSourceFile());
		SegregateCSV seg = new SegregateCSV();
		for (AnalystSegregateTarget target : this.script.getSegregate()
				.getSegregateTargets()) {
			String filename = this.script.getConfig().getFilename(
					target.getFile());
			seg.getTargets().add(
					new SegregateTargetPercent(filename, target.getPercent()));
			// mark generated
			this.script.markGenerated(target.getFile());
		}
		seg.analyze(inputFile, headers, this.script.getConfig().getCSVFormat());

		seg.process();
	}

	public void generate() {
		// mark generated
		this.script.markGenerated(this.script.getNormalize().getTargetFile());

		// get filenames
		String sourceFile = this.script.getConfig().getFilename(
				this.script.getGenerate().getSourceFile());
		String targetFile = this.script.getConfig().getFilename(
				this.script.getGenerate().getTargetFile());
		int input = this.script.getGenerate().getInput();
		int ideal = this.script.getGenerate().getIdeal();

		boolean headers = this.script.expectInputHeaders(this.script
				.getGenerate().getSourceFile());
		EncogUtility.convertCSV2Binary(sourceFile, targetFile, input, ideal,
				headers);
	}
	
	private EncogPersistedObject createML(String type, String arch,int input,int ideal) {
		
		if( type.equalsIgnoreCase("feedforward") ) {
			BasicNetwork network = new BasicNetwork();
			network.addLayer(new BasicLayer(new ActivationSigmoid(), true,input));
			network.addLayer(new BasicLayer(new ActivationSigmoid(), true,4));
			network.addLayer(new BasicLayer(null, false,ideal));
			network.getStructure().finalizeStructure();
			network.reset();
			return network;
		}
		
		return null;
	}
	
	public void create()
	{
		// get filenames
		String trainingFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getTrainingFile());
		String resourceFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getResourceFile());
		String resource = this.script.getMachineLearning().getResourceName();
		String type = this.script.getMachineLearning().getMLType();
		String arch = this.script.getMachineLearning().getMLArchitecture();
		
		EncogEGBFile egb = new EncogEGBFile(new File(trainingFile));
		egb.open();
		int input = egb.getInputCount();
		int ideal = egb.getIdealCount();
		egb.close();
		
		EncogMemoryCollection encog = new EncogMemoryCollection();
		if( new File(resourceFile).exists()) {
			encog.load(resourceFile);
		}
		EncogPersistedObject obj = createML(type,arch,input, ideal);
		encog.add(resource, obj);
		encog.save(resourceFile);
	}
	
	public void train()
	{
		// get filenames
		String trainingFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getTrainingFile());
		String resourceFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getResourceFile());
		String resource = this.script.getMachineLearning().getResourceName();
		
		NeuralDataSet trainingSet = EncogUtility.loadEGB2Memory(trainingFile);
		
		EncogMemoryCollection encog = new EncogMemoryCollection();
		encog.load(resourceFile);
				
		EncogPersistedObject method = encog.find(resource);
		EncogUtility.trainToError((MLMethod)method, trainingSet, 0.01);					
		encog.save(resourceFile);
	}
	
	public void evaluateRaw()
	{
		// get filenames
		String evalFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getEvalFile());
		String resourceFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getResourceFile());
		String resource = this.script.getMachineLearning().getResourceName();
		
		String outputFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getOutputFile());
		
		EncogMemoryCollection encog = new EncogMemoryCollection();
		encog.load(resourceFile);
		MLRegression method = (MLRegression) encog.find(resource);
		
		boolean headers = this.script.expectInputHeaders(this.script
				.getNormalize().getSourceFile());
		

		EvaluateCSV eval = new EvaluateCSV();
		eval.analyze(evalFile, headers, this.script.getConfig().getCSVFormat());
		eval.process(outputFile, method);

	}
	
	public void evaluate()
	{
		// get filenames
		String evalFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getEvalFile());
		String resourceFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getResourceFile());
		String resource = this.script.getMachineLearning().getResourceName();
		
		String outputFile = this.script.getConfig().getFilename(
				this.script.getMachineLearning().getOutputFile());
		
		EncogMemoryCollection encog = new EncogMemoryCollection();
		encog.load(resourceFile);
		MLRegression method = (MLRegression) encog.find(resource);
		
		boolean headers = this.script.expectInputHeaders(this.script
				.getNormalize().getSourceFile());
		

		AnalystEvaluateCSV eval = new AnalystEvaluateCSV();
		eval.analyze(evalFile, headers, this.script.getConfig().getCSVFormat());
		eval.process(outputFile, this, method);

	}
	


	public static void main(String[] args) {
		
		Logging.stopConsoleLogging();
		
		EncogAnalyst a = new EncogAnalyst();

		//a.wizard(new File("d:\\data\\iris.txt"), new File(
		//		"d:\\data\\iris_raw.csv"), false, CSVFormat.ENGLISH);

		URL url = null;
		try {
			url = new URL(
					"http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}

		a.wizard(url, new File("c:\\iris\\iris.txt"), new File(
				"c:\\iris\\iris_raw.csv"), false, CSVFormat.ENGLISH);

		a.randomize();
		a.segregate();
		a.normalize();
		a.generate();
		a.create();
		a.train();
		a.evaluate();
		a.save("c:\\iris\\iris.txt");

		/*
				a.analyze(
						new File("d:\\data\\forest.csv"), 
						false, 
						CSVFormat.ENGLISH);
				a.save("d:\\data\\forest.txt");*/

		System.out.println("Done");
	}

}
