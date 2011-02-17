package org.encog.app.analyst;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.encog.app.analyst.analyze.PerformAnalysis;
import org.encog.app.analyst.script.AnalystScript;
import org.encog.app.analyst.script.DataField;
import org.encog.app.analyst.script.EncogAnalystConfig;
import org.encog.app.analyst.script.classify.ClassifyField;
import org.encog.app.analyst.script.normalize.NormalizedField;
import org.encog.app.analyst.script.segregate.AnalystSegregateTarget;
import org.encog.app.quant.classify.ClassifyCSV;
import org.encog.app.quant.classify.ClassifyMethod;
import org.encog.app.quant.normalize.NormalizationDesired;
import org.encog.app.quant.normalize.NormalizationStats;
import org.encog.app.quant.normalize.NormalizeCSV;
import org.encog.app.quant.normalize.NormalizedFieldStats;
import org.encog.app.quant.segregate.SegregateCSV;
import org.encog.app.quant.segregate.SegregateTargetPercent;
import org.encog.app.quant.shuffle.ShuffleCSV;
import org.encog.util.csv.CSVFormat;
import org.encog.util.file.FileUtil;
import org.encog.util.simple.EncogUtility;

public class EncogAnalyst {
	
	public final static String ACTION_ANALYZE = "ANALYZE";
		
	private AnalystScript script = new AnalystScript();

	public void analyze(File file, boolean headers, CSVFormat format)
	{
		script.getConfig().setFilename(EncogAnalystConfig.FILE_RAW,file.toString());
		script.getConfig().setCSVFormat(format);
		script.getConfig().setInputHeaders(headers);
		PerformAnalysis a = new PerformAnalysis(script, file.toString(),headers,CSVFormat.ENGLISH);
		a.process(this);
		
	}
	
	public void clear()
	{
		
	}

	public void load(String filename)
	{
		load(new File(filename));
	}
	
	public void save(String filename)
	{
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
		
		for(int i=0;i<this.script.getFields().length;i++)
		{
			DataField f = dataFields[i];
			NormalizationDesired action;
			
			if( f.isInteger() || f.isReal() && !f.isClass() ) {
				action = NormalizationDesired.Normalize;
				norm[i] = new NormalizedField(f.getName(),action,1,-1); 
			} else {				
				action = NormalizationDesired.PassThrough;
				norm[i] = new NormalizedField(f.getName(),action);
			}			
		}
		
		String s;
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_NORMALIZE, FileUtil.addFilenameBase(file, "_norm").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_CLASSIFY, FileUtil.addFilenameBase(file, "_class").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_RANDOM, FileUtil.addFilenameBase(file, "_random").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_TRAIN, s= FileUtil.addFilenameBase(file, "_train").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_EVAL, FileUtil.addFilenameBase(file, "_eval").toString());
		this.script.getConfig().setFilename(EncogAnalystConfig.FILE_TRAINSET, FileUtil.forceExtension(s, "egb"));
		
		this.script.getNormalize().setNormalizedFields(norm);
		this.script.getNormalize().setSourceFile(EncogAnalystConfig.FILE_RAW);
		this.script.getNormalize().setTargetFile(EncogAnalystConfig.FILE_NORMALIZE);
		this.script.getClassify().setSourceFile(EncogAnalystConfig.FILE_NORMALIZE);
		this.script.getClassify().setTargetFile(EncogAnalystConfig.FILE_CLASSIFY);
		this.script.getRandomize().setSourceFile(EncogAnalystConfig.FILE_CLASSIFY);
		this.script.getRandomize().setTargetFile(EncogAnalystConfig.FILE_RANDOM);
		this.script.getSegregate().setSourceFile(EncogAnalystConfig.FILE_RANDOM);
		this.script.getGenerate().setSourceFile(EncogAnalystConfig.FILE_TRAIN);
		this.script.getGenerate().setTargetFile(EncogAnalystConfig.FILE_TRAINSET);
	}
	
	private void generateClassifiedFields(File file) {

		List<ClassifyField> classifyFields = new ArrayList<ClassifyField>();
		DataField[] dataFields = this.getScript().getFields();
		
		for(int i=0;i<this.script.getFields().length;i++)
		{
			DataField f = dataFields[i];
			
			if( f.isClass() ) {
				ClassifyMethod method;
				
				if( f.getClassMembers().size()>=3 )
					method = ClassifyMethod.Equilateral;
				else
					method = ClassifyMethod.OneOf;
				
				ClassifyField cField = new ClassifyField(f.getName(),method,1,-1);
				classifyFields.add(cField);
			}
		}
		
		ClassifyField[] array = new ClassifyField[classifyFields.size()];
		for(int i=0;i<array.length;i++) {
			array[i] = classifyFields.get(i);
		}
		
		this.script.getClassify().setClassifiedFields(array);		
	}
	
	private void generateRandomize(File file) {

	}
	
	private void generateSegregate(File file) {
		AnalystSegregateTarget[] array = new AnalystSegregateTarget[2];
		array[0] = new AnalystSegregateTarget(EncogAnalystConfig.FILE_TRAIN,75);
		array[1] = new AnalystSegregateTarget(EncogAnalystConfig.FILE_EVAL,25);
		this.script.getSegregate().setSegregateTargets(array);
	}
	
	private void generateGenerate(File file) {
		//int inputColumns = this.script.getFields().length - this.script.getClassify().getClassifiedFields().length;
		//int idealColumns = this.script.getClassify().
		
	}
	
	public void wizard(File file, boolean b, CSVFormat english) {
		analyze(file, b, english);
		generateNormalizedFields(file);
		generateClassifiedFields(file);
		generateRandomize(file);
		generateSegregate(file);
		generateGenerate(file);
	}	
	
	public void classify()
	{
		// mark generated
		this.script.markGenerated(this.script.getClassify().getTargetFile());
		
		// get filenames
		String sourceFile = this.script.getConfig().getFilename( this.script.getClassify().getSourceFile() );
		String targetFile = this.script.getConfig().getFilename( this.script.getClassify().getTargetFile() );
		
		// prepare to classify
		boolean headers = this.script.expectInputHeaders(this.script.getClassify().getSourceFile());
		ClassifyCSV classify = new ClassifyCSV();
		classify.setProduceOutputHeaders(this.script.getConfig().isOutputHeaders());
		classify.analyze(sourceFile, headers, CSVFormat.ENGLISH);
		
		for( ClassifyField field : this.script.getClassify().getClassifiedFields() ) {
			
			classify.addTarget(field.getName(), field.getMethod(),field.getHigh(),field.getLow(), -1, null);	
		}
		
		
		classify.process(targetFile);
	}
	
	public void normalize()
	{
		// mark generated
		this.script.markGenerated(this.script.getNormalize().getTargetFile());
				
		// get filenames
		String sourceFile = this.script.getConfig().getFilename( this.script.getNormalize().getSourceFile() );
		String targetFile = this.script.getConfig().getFilename( this.script.getNormalize().getTargetFile() );
		
		// prepare to normalize
		NormalizeCSV norm = new NormalizeCSV();
		NormalizedField[] normFields = this.script.getNormalize().getNormalizedFields();
		NormalizationStats stats = new NormalizationStats(normFields.length);
		
		int index = 0;
		for(NormalizedField normField: this.script.getNormalize().getNormalizedFields())
		{
			NormalizedFieldStats nfs = new NormalizedFieldStats(); 
			DataField dataField = this.script.findDataField(normField.getName());
			stats.getStats()[index++] = nfs;
			nfs.setName(normField.getName());
			nfs.setAction(normField.getAction());
			nfs.setNormalizedHigh(normField.getNormalizedHigh());
			nfs.setNormalizedLow(normField.getNormalizedLow());
			nfs.setActualHigh(dataField.getMax());
			nfs.setActualLow(dataField.getMin());
		}
		
		boolean headers = this.script.expectInputHeaders(this.script.getNormalize().getSourceFile());
		norm.analyze(sourceFile,headers,this.script.getConfig().getCSVFormat(), stats);
		norm.setProduceOutputHeaders(this.script.getConfig().isOutputHeaders());
		norm.normalize(targetFile);
	}
	
	public void randomize()
	{
		// mark generated
		this.script.markGenerated(this.script.getRandomize().getTargetFile());
				
		// get filenames
		String sourceFile = this.script.getConfig().getFilename( this.script.getRandomize().getSourceFile() );
		String targetFile = this.script.getConfig().getFilename( this.script.getRandomize().getTargetFile() );
		
		// prepare to normalize
		ShuffleCSV norm = new ShuffleCSV();
		boolean headers = this.script.expectInputHeaders(this.script.getRandomize().getSourceFile());
		norm.analyze(sourceFile, headers ,this.script.getConfig().getCSVFormat());
		norm.process(targetFile);
	}
	
	public void segregate()
	{	
		// get filenames		
		String inputFile = this.script.getConfig().getFilename( this.script.getSegregate().getSourceFile() );
		
		// prepare to segregate
		boolean headers = this.script.expectInputHeaders(this.script.getSegregate().getSourceFile());
		SegregateCSV seg = new SegregateCSV();		
		for( AnalystSegregateTarget target: this.script.getSegregate().getSegregateTargets() )
		{
			String filename = this.script.getConfig().getFilename( target.getFile() );
			seg.getTargets().add(new SegregateTargetPercent(filename,target.getPercent()));
			// mark generated
			this.script.markGenerated(target.getFile());
		}
		seg.analyze(inputFile, headers, this.script.getConfig().getCSVFormat());

		seg.process();		
	}
	
	public void generate()
	{
		// mark generated
		this.script.markGenerated(this.script.getNormalize().getTargetFile());
				
		// get filenames
		String sourceFile = this.script.getConfig().getFilename( this.script.getGenerate().getSourceFile() );
		String targetFile = this.script.getConfig().getFilename( this.script.getGenerate().getTargetFile() );
		int input = this.script.getGenerate().getInput();
		int ideal = this.script.getGenerate().getIdeal();
		
		boolean headers = this.script.expectInputHeaders(this.script.getGenerate().getSourceFile());
		EncogUtility.convertCSV2Binary(sourceFile, targetFile, input, ideal, headers);
	}


	public static void main(String[] args)
	{
		EncogAnalyst a = new EncogAnalyst();
	
		a.load("d:\\data\\iris.txt");
		a.wizard(
				new File("d:\\data\\iris_raw.csv"), 
				false, 
				CSVFormat.ENGLISH);
		a.normalize();
		a.classify();
		a.randomize();
		a.segregate();
		a.generate();
		a.save("d:\\data\\iris.txt");
		
/*
		a.analyze(
				new File("d:\\data\\forest.csv"), 
				false, 
				CSVFormat.ENGLISH);
		a.save("d:\\data\\forest.txt");*/
		
		System.out.println("Done");
	}

	

	

}
